"""Small, from-scratch RGB pixel DDPM with learned Arabic/English word conditioning.

Unlike the layout DDPM, it diffuses actual 32x32 RGB pixels. Training data is
procedural single-object imagery only. It is NOT a general-purpose image model.
"""
from dataclasses import dataclass
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

COLORS = {"red": (230,45,50), "green": (36,190,85), "blue": (45,90,225)}
COLOR_NAMES = ("red", "green", "blue")
SHAPES = ("circle", "square")
AR_FEM = ("حمراء","خضراء","زرقاء")
AR_MASC = ("أحمر","أخضر","أزرق")
AR_SHAPES = ("دائرة","مربع")
WORDS = (
    "<pad>", "a", "red", "green", "blue", "circle", "square", "on", "white",
    "background", "simple", "shape", "دائرة", "مربع", "حمراء", "خضراء",
    "زرقاء", "أحمر", "أخضر", "أزرق", "على", "خلفية", "بيضاء", "شكل", "بسيط",
)
VOCAB = {word:i for i,word in enumerate(WORDS)}


def vocabulary_for(dataset_version="v1"):
    if dataset_version == "v1":
        return WORDS
    if dataset_version == "v2":
        from src.pixel_dataset_v2 import WORDS_V2
        return WORDS_V2
    raise ValueError("Unsupported dataset vocabulary version")


def tokenize(text, max_tokens=12, dataset_version="v1"):
    if not isinstance(text,str) or not text.strip():
        raise ValueError("Prompt must be nonempty text")
    words = re.findall(r"[\w\u0621-\u064a]+",text.lower(),flags=re.UNICODE)
    vocabulary = {word: i for i, word in enumerate(vocabulary_for(dataset_version))}
    if not words or len(words)>max_tokens or any(word not in vocabulary for word in words):
        raise ValueError("Unsupported vocabulary; this MVP only supports bilingual shape prompts")
    return torch.tensor([vocabulary[word] for word in words]+[0]*(max_tokens-len(words)), dtype=torch.long)


def caption(color,shape,language="en",variant=0):
    if color not in COLORS or shape not in SHAPES: raise ValueError("Unsupported synthetic concept")
    if language=="en":
        return (f"{color} {shape}",f"a {color} {shape} on white background",f"simple {color} {shape}")[variant%3]
    if language=="ar":
        sh = AR_SHAPES[SHAPES.index(shape)]
        colors = AR_FEM if shape=="circle" else AR_MASC
        ar_color=colors[COLOR_NAMES.index(color)]
        return (f"{sh} {ar_color}",f"{sh} {ar_color} على خلفية بيضاء",f"شكل {sh} {ar_color} بسيط")[variant%3]
    raise ValueError("Language must be en or ar")


def synthetic_example(seed,color=None,shape=None,language=None):
    """Ground-truth image creation: called only when building TRAINING examples."""
    rng=random.Random(int(seed))
    color=color or rng.choice(COLOR_NAMES)
    shape=shape or rng.choice(SHAPES)
    language=language or rng.choice(("en","ar"))
    prompt=caption(color,shape,language,rng.randrange(3))
    image=Image.new("RGB",(32,32),(246,246,244))
    draw=ImageDraw.Draw(image)
    side=rng.randint(14,22)
    x,y=rng.randint(3,29-side),rng.randint(3,29-side)
    bounds=(x,y,x+side,y+side)
    if shape=="circle": draw.ellipse(bounds,fill=COLORS[color])
    else: draw.rectangle(bounds,fill=COLORS[color])
    array=np.asarray(image,dtype=np.float32).copy()
    return (torch.from_numpy(array).permute(2,0,1)/127.5-1,
            tokenize(prompt), prompt,(color,shape))


@dataclass
class PixelConfig:
    image_size:int=32
    channels:int=16
    text_dim:int=64
    diffusion_steps:int=100
    max_tokens:int=12
    architecture:str="rgb-unet-text-v1"
    dataset_version:str="v1"
    def __post_init__(self):
        if (self.image_size!=32 or self.max_tokens!=12 or self.channels<16
                or self.channels%8 or self.text_dim<16 or self.diffusion_steps<8
                or self.architecture not in ("rgb-unet-text-v1", "rgb-unet-spatial-text-v2")
                or self.dataset_version not in ("v1", "v2")
                or (self.architecture=="rgb-unet-spatial-text-v2"
                    and self.dataset_version!="v2")):
            raise ValueError("Unsupported Sora pixel diffusion configuration")


class TextEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token=nn.Embedding(len(vocabulary_for(cfg.dataset_version)),cfg.text_dim,padding_idx=0)
        self.position=nn.Embedding(cfg.max_tokens,cfg.text_dim)
        self.project=nn.Sequential(nn.LayerNorm(cfg.text_dim),nn.Linear(cfg.text_dim,cfg.text_dim),nn.SiLU())
    def forward(self,tokens):
        if tokens.ndim!=2 or tokens.shape[1]!=12: raise ValueError("Expected [batch,12] tokens")
        mask=(tokens!=0).float()
        pos=torch.arange(12,device=tokens.device)
        x=(self.token(tokens)+self.position(pos)[None])*mask[...,None]
        return self.project(x.sum(1)/mask.sum(1).clamp(min=1)[:,None])


class ResBlock(nn.Module):
    def __init__(self,ci,co,d):
        super().__init__()
        self.norm1=nn.GroupNorm(8,ci)
        self.conv1=nn.Conv2d(ci,co,3,padding=1)
        self.norm2=nn.GroupNorm(8,co)
        self.conv2=nn.Conv2d(co,co,3,padding=1)
        self.condition=nn.Linear(d,2*co)
        self.skip=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
    def forward(self,x,context):
        h=self.conv1(F.silu(self.norm1(x)))
        scale,bias=self.condition(context).chunk(2,dim=-1)
        h=self.norm2(h)*(1+scale[:,:,None,None])+bias[:,:,None,None]
        return (self.skip(x)+self.conv2(F.silu(h)))/math.sqrt(2)


class SpatialTextAttention(nn.Module):
    """Trainable per-pixel attention over individual prompt words.

    Original pooled v1 text loses word identity. Attention lets RGB image
    features attend to shape and color tokens at 32x32 and 16x16. A zero-
    initialized output permits exact initial v1-denoiser weight transfer.
    """
    def __init__(self, channels, text_dim):
        super().__init__()
        self.query=nn.Conv2d(channels,channels,1)
        self.key=nn.Linear(text_dim,channels)
        self.value=nn.Linear(text_dim,channels)
        self.output=nn.Conv2d(channels,channels,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, tokens, encoder):
        batch,channels,height,width=features.shape
        mask=tokens.ne(0)
        positions=torch.arange(tokens.shape[1],device=tokens.device)
        words=encoder.token(tokens)+encoder.position(positions)[None,:,:]
        words=words*mask[...,None]
        queries=self.query(features).flatten(2).transpose(1,2)
        keys=self.key(words)
        values=self.value(words)
        scores=torch.bmm(queries,keys.transpose(1,2)) / math.sqrt(channels)
        scores=scores.masked_fill(~mask[:,None,:],-1e4)
        weights=scores.softmax(dim=-1)
        weighted=torch.bmm(weights,values).transpose(1,2).reshape(
            batch,channels,height,width
        )
        delta=self.output(weighted)
        # Empty/null CFG prompt cannot accidentally receive spatial text.
        valid=mask.any(dim=1).float()[:,None,None,None]
        return features+delta*valid


class PixelDenoiser(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or PixelConfig()
        c,d=self.config.channels,self.config.text_dim
        self.text=TextEncoder(self.config)
        self.time=nn.Sequential(nn.Linear(64,d),nn.SiLU(),nn.Linear(d,d))
        self.context=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d))
        self.input=nn.Conv2d(3,c,3,padding=1)
        self.block32=ResBlock(c,c,d)
        self.down16=nn.Conv2d(c,2*c,3,stride=2,padding=1)
        self.block16=ResBlock(2*c,2*c,d)
        self.mid=ResBlock(2*c,2*c,d)
        self.up32=ResBlock(3*c,c,d)
        self.output=nn.Sequential(nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,3,3,padding=1))
        if self.config.architecture=="rgb-unet-spatial-text-v2":
            self.spatial32=SpatialTextAttention(c,d)
            self.spatial16=SpatialTextAttention(2*c,d)
    def forward(self,noisy,timestep,tokens):
        if (noisy.ndim!=4 or noisy.shape[1:]!=(3,32,32)
                or timestep.shape!=(len(noisy),) or tokens.shape!=(len(noisy),12)):
            raise ValueError("Expected [B,3,32,32] RGB, [B] steps, [B,12] tokens")
        frequencies=torch.exp(torch.arange(32,device=noisy.device).float()*(-math.log(10000)/31))
        time=timestep.float()[:,None]/self.config.diffusion_steps
        periodic=torch.cat((torch.sin(time*frequencies[None]*1000),
                            torch.cos(time*frequencies[None]*1000)),dim=-1)
        context=self.context(self.text(tokens)+self.time(periodic))
        x32=self.block32(self.input(noisy),context)
        if self.config.architecture=="rgb-unet-spatial-text-v2":
            x32=self.spatial32(x32,tokens,self.text)
        x16=self.block16(self.down16(x32),context)
        if self.config.architecture=="rgb-unet-spatial-text-v2":
            x16=self.spatial16(x16,tokens,self.text)
        mid=self.mid(x16,context)
        merged=torch.cat((F.interpolate(mid,scale_factor=2,mode="nearest"),x32),dim=1)
        return self.output(self.up32(merged,context))


def noise_schedule(config,device):
    t=torch.linspace(0,1,config.diffusion_steps+1,dtype=torch.float64,device=device)
    cum=torch.cos((t+0.008)/1.008*math.pi/2).square()
    cum=cum/cum[0]
    beta=(1-cum[1:]/cum[:-1]).clamp(0.0001,0.999).float()
    alpha=1-beta
    return beta,alpha,alpha.cumprod(0)


def apply_text_dropout(tokens, probability):
    """Classifier-free training: replace entire prompts with null condition."""
    if not 0 <= probability <= 1 or not math.isfinite(probability):
        raise ValueError("Text dropout probability must be between 0 and 1")
    if tokens.ndim != 2 or tokens.shape[1] != 12:
        raise ValueError("Expected [B,12] token batch")
    mask = torch.rand(len(tokens), device=tokens.device) < probability
    result = tokens.clone()
    result[mask] = 0
    return result, mask


def guided_noise(model, noisy, times, tokens, guidance_scale=1.0):
    """Epsilon-space CFG, scale 1 = conditional only; scale 0 = unconditional."""
    if not math.isfinite(guidance_scale) or not 0 <= guidance_scale <= 20:
        raise ValueError("guidance_scale must be between 0 and 20")
    if guidance_scale == 1.0:
        return model(noisy, times, tokens)
    prediction = model(
        torch.cat([noisy, noisy], dim=0),
        torch.cat([times, times], dim=0),
        torch.cat([torch.zeros_like(tokens), tokens], dim=0),
    )
    unconditional, conditioned = prediction.chunk(2, dim=0)
    return unconditional + guidance_scale * (conditioned - unconditional)


def training_loss(model,images,tokens,cfg_drop_prob=0.0):
    if images.ndim!=4 or images.shape[1:]!=(3,32,32): raise ValueError("Expected 32x32 RGB")
    _,_,cumulative=noise_schedule(model.config,images.device)
    times=torch.randint(0,model.config.diffusion_steps,(len(images),),device=images.device)
    eps=torch.randn_like(images)
    a=cumulative[times][:,None,None,None]
    noisy=a.sqrt()*images+(1-a).sqrt()*eps
    conditioned, _ = apply_text_dropout(tokens, cfg_drop_prob)
    return F.mse_loss(model(noisy,times,conditioned),eps)


def shape_aware_training_objective(model,images,tokens,cfg_drop_prob=0.15,
                                   silhouette_weight=0.15):
    """Experimental pixel-DDPM with an extra learned silhouette objective.

    The model always predicts real RGB pixel noise, never a shape template.
    For lower-noise timesteps, estimate x0 and compare the continuous RGB
    saturation foreground against the observed TRAINING image. Gradients
    flow through learned RGB denoising. Inference stays Gaussian -> pixels.
    """
    if model.config.architecture!="rgb-unet-spatial-text-v2":
        raise ValueError("Shape objective requires the spatial-text architecture")
    if not 0 <= silhouette_weight <= 2 or not math.isfinite(silhouette_weight):
        raise ValueError("Invalid silhouette_weight")
    if images.ndim!=4 or images.shape[1:]!=(3,32,32) or tokens.shape!=(len(images),12):
        raise ValueError("Expected 32x32 RGB pixels and matching text tokens")
    _,_,cumulative=noise_schedule(model.config,images.device)
    times=torch.randint(0,model.config.diffusion_steps,(len(images),),
                        device=images.device)
    eps=torch.randn_like(images)
    a=cumulative[times][:,None,None,None]
    noisy=a.sqrt()*images+(1-a).sqrt()*eps
    conditioned,_=apply_text_dropout(tokens,cfg_drop_prob)
    predicted_eps=model(noisy,times,conditioned)
    noise_mse=F.mse_loss(predicted_eps,eps)
    reconstructed=((noisy-(1-a).sqrt()*predicted_eps)/a.sqrt()).clamp(-1.25,1.25)
    reconstructed_saturation=reconstructed.amax(dim=1)-reconstructed.amin(dim=1)
    true_saturation=images.amax(dim=1)-images.amin(dim=1)
    soft_mask=torch.sigmoid((reconstructed_saturation-0.45)*8.0)
    target_mask=(true_saturation>0.45).float()
    intersection=(soft_mask*target_mask).sum(dim=(-1,-2))
    foreground_sum=soft_mask.sum(dim=(-1,-2))+target_mask.sum(dim=(-1,-2))
    dice_loss=1-(2*intersection+1)/(foreground_sum+1)
    # x0 is less stable when alpha is very small: train mask only at t<=T/2.
    valid=(times <= model.config.diffusion_steps//2).float()
    silhouette_loss=(dice_loss*valid).sum()/valid.sum().clamp(min=1)
    total=noise_mse+silhouette_weight*silhouette_loss
    return {"total_loss":total,"noise_mse":noise_mse.detach(),
            "silhouette_loss":silhouette_loss.detach()}


def clip_predicted_x0(x0,policy="fixed"):
    """Optional inference-only clipping policy.

    'fixed': historical clipping to [-1,1], unchanged.
    'dynamic': samplewise 99.5th percentile dynamic thresholding.
    Both produce bounded x0; dynamic retains more relative contrast
    when nearly all estimated x0 values are out of bounds at high noise.
    This is a sampling hypothesis, not a learned geometry guarantee.
    """
    if policy=="fixed":
        return x0.clamp(-1,1)
    if policy=="dynamic":
        if x0.ndim!=4 or x0.shape[1:]!=(3,32,32):
            raise ValueError("Dynamic clipping expects [B,3,32,32] RGB")
        bound=torch.quantile(
            x0.detach().abs().flatten(1).float(),0.995,dim=1
        ).clamp(min=1).to(dtype=x0.dtype).view(-1,1,1,1)
        return x0.clamp(-bound,bound)/bound
    raise ValueError("Unsupported x0 clipping policy")


@torch.no_grad()
def generate(model,prompt,seed=42,device="cpu",steps=None,sampler="posterior_mean",guidance_scale=1.0,x0_clipping="fixed"):
    """Generate pixels from learned diffusion.

    'posterior_mean': full-step reverse posterior mean without noise.
    'ancestral': full-step stochastic DDPM posterior sampling, with learned
    RGB-noise prediction, standard posterior variance and x0 clipping.
    'ddim': deterministic DDIM with optional timestep skipping.
    x0_clipping: fixed (historical) or dynamic (inference-only ablation).
    """
    if not math.isfinite(guidance_scale) or not 0 <= guidance_scale <= 20:
        raise ValueError("guidance_scale must be between 0 and 20")
    if x0_clipping not in ("fixed","dynamic"):
        raise ValueError("x0_clipping must be fixed or dynamic")
    total=model.config.diffusion_steps
    if sampler not in ("posterior_mean","ancestral","ddim"):
        raise ValueError("Unsupported diffusion sampler")
    if sampler in ("posterior_mean","ancestral"):
        if steps not in (None,total):
            raise ValueError(f"{sampler} requires all diffusion timesteps")
        steps=total
    else:
        steps=total if steps is None else steps
        if steps<2 or steps>total: raise ValueError("Invalid inference steps")
    tokens=tokenize(prompt, dataset_version=model.config.dataset_version).unsqueeze(0).to(device)
    generator=torch.Generator(device=device).manual_seed(int(seed))
    x=torch.randn((1,3,32,32),device=device,generator=generator)
    beta,alpha,cum=noise_schedule(model.config,device)
    model.eval()
    if sampler in ("posterior_mean","ancestral"):
        # IMPORTANT: descending range needs a NEGATIVE step. Without -1
        # this loop is empty and the sampler returns clipped initial noise.
        for t in range(total-1,-1,-1):
            eps=guided_noise(model,x,torch.tensor([t],device=device,dtype=torch.long),tokens,guidance_scale)
            a=cum[t]
            x0=clip_predicted_x0((x-(1-a).sqrt()*eps)/a.sqrt(),x0_clipping)
            if t==0: x=x0
            else:
                prev=cum[t-1]
                mean_x0=beta[t]*prev.sqrt()/(1-a)
                mean_xt=alpha[t].sqrt()*(1-prev)/(1-a)
                x=mean_x0*x0+mean_xt*x
                if sampler=="ancestral":
                    # True reverse-process variance (not deterministic mean).
                    posterior_variance=(beta[t]*(1-prev)/(1-a)).clamp(min=0)
                    x=x+posterior_variance.sqrt()*torch.randn(
                        x.shape,device=device,dtype=x.dtype,generator=generator
                    )
    else:
        timesteps=torch.linspace(total-1,0,steps).round().long().unique_consecutive()
        for i,t in enumerate(timesteps.tolist()):
            eps=guided_noise(model,x,torch.tensor([t],device=device,dtype=torch.long),tokens,guidance_scale)
            a=cum[t]
            x0=clip_predicted_x0((x-(1-a).sqrt()*eps)/a.sqrt(),x0_clipping)
            if i+1==len(timesteps): x=x0
            else:
                prev=cum[timesteps[i+1].item()]
                x=prev.sqrt()*x0+(1-prev).sqrt()*eps
    return x[0].clamp(-1,1).cpu()


def save_image(tensor,path):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    if tensor.shape!=(3,32,32) or not torch.isfinite(tensor).all(): raise ValueError("Invalid RGB tensor")
    path.parent.mkdir(parents=True,exist_ok=True)
    rgb=((tensor.permute(1,2,0).clamp(-1,1).numpy()+1)*127.5).round().astype("uint8")
    Image.fromarray(rgb).save(path)

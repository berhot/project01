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


def tokenize(text, max_tokens=12):
    if not isinstance(text,str) or not text.strip():
        raise ValueError("Prompt must be nonempty text")
    words = re.findall(r"[\w\u0621-\u064a]+",text.lower(),flags=re.UNICODE)
    if not words or len(words)>max_tokens or any(word not in VOCAB for word in words):
        raise ValueError("Unsupported vocabulary; this MVP only supports simple bilingual shape prompts")
    return torch.tensor([VOCAB[word] for word in words]+[0]*(max_tokens-len(words)), dtype=torch.long)


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
    def __post_init__(self):
        if (self.image_size!=32 or self.max_tokens!=12 or self.channels<16
                or self.channels%8 or self.text_dim<16 or self.diffusion_steps<8
                or self.architecture!="rgb-unet-text-v1"):
            raise ValueError("Unsupported Sora pixel diffusion configuration")


class TextEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token=nn.Embedding(len(WORDS),cfg.text_dim,padding_idx=0)
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
        x16=self.block16(self.down16(x32),context)
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


def training_loss(model,images,tokens):
    if images.ndim!=4 or images.shape[1:]!=(3,32,32): raise ValueError("Expected 32x32 RGB")
    _,_,cumulative=noise_schedule(model.config,images.device)
    times=torch.randint(0,model.config.diffusion_steps,(len(images),),device=images.device)
    eps=torch.randn_like(images)
    a=cumulative[times][:,None,None,None]
    noisy=a.sqrt()*images+(1-a).sqrt()*eps
    return F.mse_loss(model(noisy,times,tokens),eps)


@torch.no_grad()
def generate(model,prompt,seed=42,device="cpu",steps=None):
    """Deterministic DDIM over RGB pixels, starting from independent Gaussian noise."""
    total=model.config.diffusion_steps
    steps=total if steps is None else steps
    if steps<2 or steps>total: raise ValueError("Invalid inference steps")
    tokens=tokenize(prompt).unsqueeze(0).to(device)
    generator=torch.Generator(device=device).manual_seed(int(seed))
    x=torch.randn((1,3,32,32),device=device,generator=generator)
    _,_,cum=noise_schedule(model.config,device)
    timesteps=torch.linspace(total-1,0,steps).round().long().unique_consecutive()
    model.eval()
    for i,t in enumerate(timesteps.tolist()):
        eps=model(x,torch.tensor([t],device=device,dtype=torch.long),tokens)
        alpha=cum[t]
        x0=((x-(1-alpha).sqrt()*eps)/alpha.sqrt()).clamp(-1,1)
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

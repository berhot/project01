"""Experimental learned shape-first text-to-image pipeline for Sora Nano.

Stage 1: a conditional mask VAE learns 32x32 silhouettes from labeled
RGB training images and bilingual text embeddings. At inference the decoder
uses a sampled Gaussian latent, NEVER a drawn ellipse/polygon/rectangle.
Stage 2: an independently trained neural appearance network maps the
generated mask + text into 32x32 RGB pixels. Neither stage imports PIL or
the synthetic shape drawing code for inference.

Only the synthetic v2 (4 colors x 3 shapes x 2 languages) domain is supported.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.pixel_diffusion import tokenize, vocabulary_for


@dataclass(frozen=True)
class ShapeFirstConfig:
    mask_width: int=32
    rgb_width: int=32
    text_dim: int=48
    latent_dim: int=12
    image_size: int=32

    def __post_init__(self):
        if (self.image_size!=32 or self.mask_width not in (16,24,32,48,64)
                or self.rgb_width not in (16,24,32,48,64)
                or self.text_dim<24 or self.text_dim>256
                or self.latent_dim<4 or self.latent_dim>64):
            raise ValueError("Unsupported shape-first 32x32 architecture")


class BilingualCondition(nn.Module):
    """Trainable Arabic/English token embeddings, with explicit null padding."""
    def __init__(self,dim):
        super().__init__()
        self.embedding=nn.Embedding(len(vocabulary_for("v2")),dim,padding_idx=0)
        self.project=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),
                                   nn.Linear(dim,dim))
    def forward(self,tokens):
        if tokens.ndim!=2 or tokens.shape[1]!=12 or tokens.dtype!=torch.long:
            raise ValueError("Expected v2 integer tokens [B,12]")
        valid=(tokens!=0)
        x=self.embedding(tokens)
        mean=(x*valid[...,None]).sum(dim=1)/valid.sum(dim=1).clamp(min=1)[:,None]
        return self.project(mean)


class MaskVAE(nn.Module):
    """Learned conditional binary-mask distribution, not a hand-drawn renderer."""
    def __init__(self,config=None):
        super().__init__()
        self.config=config or ShapeFirstConfig()
        w=self.config.mask_width
        d=self.config.text_dim
        z=self.config.latent_dim
        self.text=BilingualCondition(d)
        self.encoder=nn.Sequential(
            nn.Conv2d(1,w,3,stride=2,padding=1),nn.SiLU(),  # 16x16
            nn.Conv2d(w,2*w,3,stride=2,padding=1),nn.SiLU(), # 8x8
            nn.Conv2d(2*w,2*w,3,stride=2,padding=1),nn.SiLU(), # 4x4
            nn.Flatten(),nn.Linear(2*w*4*4,2*z),
        )
        self.seed=nn.Sequential(nn.Linear(z+d,2*w*4*4),nn.SiLU())
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(2*w,2*w,4,stride=2,padding=1),nn.SiLU(),
            nn.ConvTranspose2d(2*w,w,4,stride=2,padding=1),nn.SiLU(),
            nn.ConvTranspose2d(w,w,4,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(w,1,3,padding=1),
        )

    def encode(self,mask):
        if mask.ndim!=4 or mask.shape[1:]!=(1,32,32):
            raise ValueError("Expected foreground mask [B,1,32,32]")
        return self.encoder(mask.float()).chunk(2,dim=1)

    def decode(self,z,tokens):
        if z.ndim!=2 or z.shape[1]!=self.config.latent_dim:
            raise ValueError("Invalid mask latent")
        context=self.text(tokens)
        if len(context)!=len(z):
            raise ValueError("Latent and tokens batch mismatch")
        x=self.seed(torch.cat((z,context),dim=1))
        w=self.config.mask_width
        return self.decoder(x.reshape(-1,2*w,4,4))

    def forward(self,mask,tokens,sample=True):
        mu,logvar=self.encode(mask)
        logvar=logvar.clamp(-8,5)
        if sample:
            z=mu+torch.randn_like(mu)*torch.exp(0.5*logvar)
        else:
            z=mu
        return self.decode(z,tokens),mu,logvar

    @torch.no_grad()
    def generate(self,tokens,seed=42,latent_scale=1.0):
        if not math.isfinite(latent_scale) or not 0<latent_scale<=2:
            raise ValueError("Invalid latent scale")
        self.eval()
        device=next(self.parameters()).device
        rng=torch.Generator(device="cpu").manual_seed(int(seed))
        z=torch.randn(len(tokens),self.config.latent_dim,
                      generator=rng,dtype=torch.float32)
        z=z.to(device=device)*latent_scale
        logits=self.decode(z,tokens.to(device))
        return (logits.sigmoid()>=0.5).float(),logits.sigmoid()


class RGBAppearance(nn.Module):
    """Learns background and shape color directly from RGB target images."""
    def __init__(self,config=None):
        super().__init__()
        self.config=config or ShapeFirstConfig()
        w=self.config.rgb_width
        d=self.config.text_dim
        self.text=BilingualCondition(d)
        self.condition=nn.Linear(d,w)
        self.network=nn.Sequential(
            nn.Conv2d(w+1,w,3,padding=1),nn.SiLU(),
            nn.Conv2d(w,w,3,padding=1),nn.SiLU(),
            nn.Conv2d(w,w,3,padding=1),nn.SiLU(),
            nn.Conv2d(w,3,1),
        )
        self.input=nn.Conv2d(3,w,3,padding=1)

    def forward(self,mask,tokens):
        if mask.ndim!=4 or mask.shape[1:]!=(1,32,32):
            raise ValueError("Expected [B,1,32,32] binary/soft masks")
        if tokens.shape!=(len(mask),12):
            raise ValueError("Expected matching text tokens")
        coords=torch.linspace(-1,1,32,device=mask.device)
        yy,xx=torch.meshgrid(coords,coords,indexing="ij")
        grid=torch.stack((xx,yy),dim=0).unsqueeze(0).expand(len(mask),-1,-1,-1)
        features=self.input(torch.cat((mask,grid),dim=1))
        context=self.condition(self.text(tokens)).unsqueeze(-1).unsqueeze(-1)
        features=features+context
        return torch.tanh(self.network(torch.cat((features,mask),dim=1)))


def mask_from_rgb(images):
    """Training/validation only: derive labels from real synthetic RGB pixels."""
    if images.ndim!=4 or images.shape[1:]!=(3,32,32):
        raise ValueError("Expected [B,3,32,32] RGB")
    if not torch.isfinite(images).all():
        raise ValueError("Nonfinite training RGB pixels")
    saturation=images.amax(dim=1,keepdim=True)-images.amin(dim=1,keepdim=True)
    return (saturation>0.45).float()


def mask_reconstruction_loss(logits,target):
    if logits.shape!=target.shape or target.ndim!=4:
        raise ValueError("Mask logits and target shapes differ")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite mask logits")
    bce=F.binary_cross_entropy_with_logits(logits,target)
    probability=logits.sigmoid()
    dice=1-(2*(probability*target).sum(dim=(1,2,3))+1)/(
        probability.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))+1)
    return bce+0.7*dice.mean(),bce.detach(),dice.mean().detach()


def mask_training_objective(model,images,tokens,beta=0.01,
                            contrast_weight=0.5,paired=None):
    """Conditional VAE + optionally same-layout counterfactual shape supervision.

    `paired`: (source_square_masks, target_masks, target_tokens). The
    source mask is encoded once; decoders of circle/square/triangle
    receive the same latent z and their own target caption. This tests
    whether the word controls geometry, not just the encoder.
    """
    if not (math.isfinite(beta) and math.isfinite(contrast_weight)
            and 0<=beta<=1 and 0<=contrast_weight<=2):
        raise ValueError("Invalid mask objective weights")
    truth=mask_from_rgb(images)
    reconstruction,mu,logvar=model(truth,tokens)
    reconstruction_loss,bce,dice=mask_reconstruction_loss(reconstruction,truth)
    kl=-0.5*(1+logvar-mu.square()-logvar.exp()).sum(dim=1).mean()
    total=reconstruction_loss+beta*kl
    counter_loss=torch.zeros((),device=images.device)
    if contrast_weight>0 and paired is not None:
        src_mask,targets,words=paired
        if (src_mask.shape!=targets.shape or src_mask.ndim!=4
                or src_mask.shape[1:]!=(1,32,32)
                or words.shape!=(len(src_mask),12)):
            raise ValueError("Invalid same-layout shape pair")
        src_mu,_=model.encode(src_mask)
        logits=model.decode(src_mu,words)
        counter_loss,_,_=mask_reconstruction_loss(logits,targets)
        total=total+contrast_weight*counter_loss
    return {
        "loss":total,
        "mask_bce":bce,
        "mask_dice_loss":dice,
        "latent_kl":kl.detach(),
        "counterfactual_mask_loss":counter_loss.detach(),
    }


def appearance_training_objective(model,images,tokens,mask=None):
    true_mask=mask_from_rgb(images) if mask is None else mask
    predicted=model(true_mask,tokens)
    weight=1+2*true_mask
    l1=((predicted-images).abs()*weight).mean()
    mse=((predicted-images).square()*weight).mean()
    return {"loss":l1+0.25*mse,"rgb_l1":l1.detach(),
            "rgb_mse":mse.detach()}


@torch.no_grad()
def generate_pipeline(mask_model,appearance_model,prompt,seed=42,
                      latent_scale=1.0,device="cpu"):
    """Complete learned inference: text+Gaussian -> learned mask -> learned RGB."""
    if mask_model.config!=appearance_model.config:
        raise ValueError("Incompatible mask and RGB model architectures")
    tokens=tokenize(prompt,dataset_version="v2").unsqueeze(0).to(device)
    mask_model.eval()
    appearance_model.eval()
    hard,soft=mask_model.generate(tokens,seed=seed,latent_scale=latent_scale)
    rgb=appearance_model(hard.to(device),tokens)
    return rgb[0].clamp(-1,1).cpu(),hard[0].cpu(),soft[0].cpu()

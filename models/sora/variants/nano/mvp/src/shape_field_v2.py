"""Sora Shape Field V2: experimentally learned shape geometry and color.

This is deliberately a NEW architecture, NOT a mutation of shape-first V1.
Limited to v2's 3 shape/4 color vocabulary in Arabic and English.

The neural implicit field learns foreground occupancy from real training
masks using normalized spatial coordinates and a learnable shape-word
embedding. At inference a seeded Gaussian chooses LOCATION/SCALE and
ORIENTATION; a TRAINED MLP predicts every foreground pixel. No ellipse,
triangle, rectangle, PIL drawing, or template lookup is used to generate
a shape.

The separately learned colorizer maps a learned COLOR-word embedding to
foreground RGB, and learns background RGB. Compositing the learned mask
with learned colors is appropriate to this single-object toy domain,
but this is NOT an open-domain photorealistic image model.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.pixel_diffusion import tokenize, vocabulary_for
from src.shape_first import mask_from_rgb

SHAPE_WORDS=("circle","square","triangle","دائرة","مربع","مثلث")
COLOR_WORDS=("red","green","blue","yellow",
             "حمراء","خضراء","زرقاء","صفراء",
             "أحمر","أخضر","أزرق","أصفر")


@dataclass(frozen=True)
class ShapeFieldConfig:
    image_size:int=32
    hidden:int=64
    word_dim:int=32

    def __post_init__(self):
        if self.image_size!=32 or self.hidden not in (16,32,48,64,96,128) or not 8<=self.word_dim<=128:
            raise ValueError("Unsupported shape-field geometry or text dimension")


def keyword_ids(tokens,words):
    """Isolate a single learned semantic keyword, independent of caption length."""
    if tokens.ndim!=2 or tokens.shape[1]!=12 or tokens.dtype!=torch.long:
        raise ValueError("Expected bilingual v2 token IDs [B,12]")
    vocabulary=vocabulary_for("v2")
    valid_ids=torch.tensor([vocabulary.index(word) for word in words],
                           device=tokens.device,dtype=torch.long)
    matched=(tokens[:,:,None]==valid_ids[None,None,:]).any(dim=2)
    counts=matched.sum(dim=1)
    if not bool(torch.all(counts==1)):
        raise ValueError("Expected exactly one supported concept word per caption")
    slot=matched.long().argmax(dim=1)
    return tokens.gather(1,slot[:,None]).squeeze(1)


def layout_from_mask(masks):
    """TRAINING-ONLY numerical bbox and triangle orientation measurement.

    This extracts POSITION/SCALE from pixels; it does not draw the shape.
    Orientation is inferred from mask's top vs bottom density, not from
    a hard-coded triangle renderer. The neural field must learn the shape.
    """
    if masks.ndim!=4 or masks.shape[1:]!=(1,32,32):
        raise ValueError("Expected [B,1,32,32] masks")
    if not torch.isfinite(masks).all():
        raise ValueError("Nonfinite masks")
    locations=[]
    for i in range(len(masks)):
        ys,xs=torch.where(masks[i,0]>0.5)
        if len(xs)<20:
            raise ValueError("Cannot infer location from empty mask")
        left,right=xs.min().float(),xs.max().float()
        top,bottom=ys.min().float(),ys.max().float()
        x=(left+right)/2
        y=(top+bottom)/2
        rx=(right-left+1)/2
        ry=(bottom-top+1)/2
        midpoint=(top+bottom)/2
        top_mass=(ys.float()<midpoint).sum().float()
        bottom_mass=(ys.float()>midpoint).sum().float()
        orientation=torch.where(top_mass<bottom_mass,1.0,-1.0)
        locations.append(torch.stack((x,y,rx,ry,orientation)))
    return torch.stack(locations).to(masks.device)


def draw_layout_from_noise(batch,seed,device="cpu"):
    """Seeded Gaussian -> *layout only*. Geometry remains entirely neural."""
    if batch<1:
        raise ValueError("Batch size must be positive")
    generator=torch.Generator(device="cpu").manual_seed(int(seed))
    z=torch.randn((batch,5),generator=generator)
    cx=16+4*z[:,0].tanh()
    cy=16+4*z[:,1].tanh()
    radius_x=8+2*z[:,2].tanh()
    radius_y=8+2*z[:,3].tanh()
    orientation=torch.where(z[:,4]>=0,1.0,-1.0)
    return torch.stack((cx,cy,radius_x,radius_y,orientation),dim=1).to(device)


def coordinate_features(layout,size=32):
    if layout.ndim!=2 or layout.shape[1]!=5 or size!=32:
        raise ValueError("Expected [B,5] location/scale/orientation")
    if not torch.isfinite(layout).all() or bool((layout[:,2:4]<=1).any()):
        raise ValueError("Invalid geometry layout")
    x=torch.arange(size,device=layout.device,dtype=layout.dtype)[None,None,:]
    y=torch.arange(size,device=layout.device,dtype=layout.dtype)[None,:,None]
    u=((x-layout[:,0,None,None])/layout[:,2,None,None]).expand(-1,size,-1)
    v=((y-layout[:,1,None,None])/layout[:,3,None,None]).expand(-1,-1,size)
    orient=layout[:,4,None,None].expand(-1,size,size)
    # Generic polynomial and Fourier coordinate channels; no class-specific
    # analytic circle/triangle/square tests or shape templates.
    return torch.stack((u,v,u.square(),v.square(),u*v,
                        torch.sin(math.pi*u),torch.sin(math.pi*v),
                        orient),dim=1)


class NeuralMaskField(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or ShapeFieldConfig()
        width=self.config.hidden
        dimension=self.config.word_dim
        self.shape_words=nn.Embedding(len(vocabulary_for("v2")),dimension,padding_idx=0)
        self.net=nn.Sequential(
            nn.Conv2d(8+dimension,width,1),nn.SiLU(),
            nn.Conv2d(width,width,1),nn.SiLU(),
            nn.Conv2d(width,width,1),nn.SiLU(),
            nn.Conv2d(width,1,1)
        )

    def forward(self,layout,tokens):
        if layout.ndim!=2 or tokens.shape!=(len(layout),12):
            raise ValueError("Batch mismatch for spatial shape field")
        selected=keyword_ids(tokens,SHAPE_WORDS)
        text=self.shape_words(selected)
        space=coordinate_features(layout,self.config.image_size)
        text_maps=text[:,:,None,None].expand(-1,-1,32,32)
        return self.net(torch.cat((space,text_maps),dim=1))

    @torch.no_grad()
    def sample(self,tokens,seed=42):
        self.eval()
        device=next(self.parameters()).device
        layout=draw_layout_from_noise(len(tokens),seed,device=device)
        probabilities=self(layout,tokens.to(device)).sigmoid()
        return (probabilities>=0.5).float(),probabilities,layout


class NeuralColorizer(nn.Module):
    """Explicit color-word neural grounding; learned colors, no RGB lookup."""
    def __init__(self,config=None):
        super().__init__()
        self.config=config or ShapeFieldConfig()
        d=self.config.word_dim
        self.colors=nn.Embedding(len(vocabulary_for("v2")),d,padding_idx=0)
        self.foreground=nn.Sequential(
            nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,3),nn.Sigmoid())
        self.background=nn.Parameter(torch.tensor((2.0,2.0,2.0)))

    def forward(self,mask,tokens):
        if mask.ndim!=4 or mask.shape[1:]!=(1,32,32) or tokens.shape!=(len(mask),12):
            raise ValueError("Expected [B,1,32,32] masks and [B,12] captions")
        selected=keyword_ids(tokens,COLOR_WORDS)
        fg=self.foreground(self.colors(selected))[:,:,None,None]
        bg=self.background.sigmoid().reshape(1,3,1,1)
        # Learned foreground/background RGB; single-object alpha composite.
        rgb=mask.clamp(0,1)*fg+(1-mask.clamp(0,1))*bg
        return rgb*2-1


def geometry_objective(model,images,tokens):
    masks=mask_from_rgb(images)
    geometry=layout_from_mask(masks)
    logits=model(geometry,tokens)
    bce=F.binary_cross_entropy_with_logits(logits,masks)
    soft=logits.sigmoid()
    dice=1-((2*(soft*masks).sum(dim=(1,2,3))+1) /
            (soft.sum(dim=(1,2,3))+masks.sum(dim=(1,2,3))+1)).mean()
    loss=bce+dice
    return {
        "loss":loss,
        "bce":bce.detach(),
        "dice_loss":dice.detach(),
        "mask_mean":soft.mean().detach()
    }


def color_objective(model,images,tokens):
    masks=mask_from_rgb(images)
    prediction=model(masks,tokens)
    fg_error=((prediction-images).abs()*masks).sum()/(masks.sum()*3).clamp(min=1)
    bg_error=((prediction-images).abs()*(1-masks)).sum() / (
        ((1-masks).sum()*3).clamp(min=1))
    loss=fg_error+0.5*bg_error
    return {
        "loss":loss,
        "foreground_l1":fg_error.detach(),
        "background_l1":bg_error.detach()
    }


@torch.no_grad()
def generate(mask_model,color_model,prompt,seed=42,device="cpu"):
    if mask_model.config!=color_model.config:
        raise ValueError("Incompatible shape-field models")
    tokens=tokenize(prompt,dataset_version="v2").unsqueeze(0).to(device)
    mask_model.eval()
    color_model.eval()
    binary,probabilities,_=mask_model.sample(tokens,seed)
    rgb=color_model(binary,tokens)
    return rgb[0].clamp(-1,1).cpu(),binary[0].cpu(),probabilities[0].cpu()

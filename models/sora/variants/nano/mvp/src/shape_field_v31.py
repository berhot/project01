"""Sora Nano V3.1: learned continuous corner-radius square geometry.

A separate experimental 32x32 neural field. PIL draws TRAINING labels only;
inference uses learned network logits, not a rounded-rectangle renderer.
This is deliberately not merged with V2/V2.1/V2.2 checkpoints.
"""
from dataclasses import asdict, dataclass
import random

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RadiusConfig:
    size: int = 32
    hidden: int = 64

    def __post_init__(self):
        if self.size != 32 or self.hidden not in (16, 32, 64, 96):
            raise ValueError("Unsupported radius field configuration")


def training_mask(seed, radius, size=32):
    """Ground truth for training/evaluation, NEVER called by inference."""
    if not 0 <= radius <= 1 or size != 32:
        raise ValueError("Invalid radius target")
    rng = random.Random(int(seed))
    side = rng.randint(12, 23)
    x = rng.randint(3, 29-side)
    y = rng.randint(3, 29-side)
    # Canvas bounds inclusive. radius represents fraction of half side.
    corner = int(round(radius * side / 2))
    canvas = Image.new("L", (size, size), 0)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (x, y, x+side, y+side), radius=corner, fill=255)
    target = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()/255)
    layout = torch.tensor([
        (2*x + side)/2, (2*y + side)/2,
        (side+1)/2, (side+1)/2], dtype=torch.float32)
    return target.unsqueeze(0), layout


def training_batch(step, count=16, seed=42):
    if step < 1 or count < 2:
        raise ValueError("Invalid training batch")
    layouts, radii, masks = [], [], []
    for index in range(count):
        # Diverse continuous rounding and deterministic extreme examples.
        r = (0.0, 0.25, 0.5, 0.75, 1.0)[index % 5] if index % 2 == 0 else (
            random.Random(seed+step*761+index*17).random())
        mask, layout = training_mask(seed+step*397+index*31, r)
        layouts.append(layout)
        radii.append(r)
        masks.append(mask)
    return (torch.stack(layouts), torch.tensor(radii, dtype=torch.float32)[:,None],
            torch.stack(masks))


class RadiusMaskField(nn.Module):
    """Continuous radius conditioned implicit mask, sampled only from weights."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RadiusConfig()
        h = self.config.hidden
        self.net = nn.Sequential(
            nn.Conv2d(8, h, 1), nn.SiLU(),
            nn.Conv2d(h, h, 1), nn.SiLU(),
            nn.Conv2d(h, h, 1), nn.SiLU(),
            nn.Conv2d(h, 1, 1))

    def forward(self, layout, radius):
        if (layout.ndim != 2 or layout.shape[1] != 4
                or radius.shape != (len(layout), 1)
                or not torch.isfinite(layout).all()
                or not torch.isfinite(radius).all()
                or bool((layout[:,2:] <= 0).any())
                or bool(((radius < 0) | (radius > 1)).any())):
            raise ValueError("Expected layout [B,4], radius [B,1] in [0,1]")
        n = self.config.size
        yy, xx = torch.meshgrid(
            torch.arange(n, device=layout.device, dtype=layout.dtype),
            torch.arange(n, device=layout.device, dtype=layout.dtype),
            indexing="ij")
        u = (xx[None]-layout[:,0,None,None])/layout[:,2,None,None]
        v = (yy[None]-layout[:,1,None,None])/layout[:,3,None,None]
        radius_map = radius[:,0,None,None].expand(-1,n,n)
        features = torch.stack((
            u, v, u.square(), v.square(), u*v,
            torch.abs(u), torch.abs(v), radius_map), dim=1)
        return self.net(features)

    @torch.no_grad()
    def predict(self, layout, radius, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("Invalid threshold")
        self.eval()
        probability = self(layout, radius).sigmoid()
        return (probability >= threshold).float(), probability


def radius_objective(model, layouts, radii, truth):
    if truth.shape != (len(layouts),1,32,32):
        raise ValueError("Expected [B,1,32,32] target")
    logits = model(layouts, radii)
    bce = F.binary_cross_entropy_with_logits(logits, truth)
    probs = logits.sigmoid()
    dice = 1 - ((2*(probs*truth).sum((1,2,3))+1) /
                (probs.sum((1,2,3))+truth.sum((1,2,3))+1)).mean()
    # Paired supervision: same layout, two radii, must lead to distinct masks.
    # Direct pixel supervision is retained; no geometric inference shortcut.
    edge_h = (probs[:,:,:,1:]-probs[:,:,:,:-1]).abs()
    edge_v = (probs[:,:,1:,:]-probs[:,:,:-1,:]).abs()
    truth_h = (truth[:,:,:,1:]-truth[:,:,:,:-1]).abs()
    truth_v = (truth[:,:,1:,:]-truth[:,:,:-1,:]).abs()
    edge = (edge_h-truth_h).abs().mean()+(edge_v-truth_v).abs().mean()
    loss = bce+dice+0.25*edge
    return {"loss":loss, "bce":bce.detach(), "dice":dice.detach(),
            "edge":edge.detach()}


def config_dict(model):
    return asdict(model.config)

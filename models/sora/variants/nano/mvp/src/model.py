"""Tiny caption-conditioned DDPM for pipeline verification (32x32 RGB).
No pretrained generators, external image APIs or borrowed model weights.
"""
from dataclasses import dataclass, asdict
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

CAPTIONS = (
    "red circle", "green circle", "blue circle",
    "red square", "green square", "blue square",
)
ALIASES = {
    "دائرة حمراء": "red circle", "دائرة خضراء": "green circle",
    "دائرة زرقاء": "blue circle", "مربع أحمر": "red square",
    "مربع اخضر": "green square", "مربع أخضر": "green square",
    "مربع أزرق": "blue square",
}


def caption_id(text):
    normalized = " ".join(text.lower().strip().split())
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in CAPTIONS:
        raise ValueError("Unsupported caption in Nano MVP. Choose one of: " + ", ".join(CAPTIONS))
    return CAPTIONS.index(normalized)


def synthetic_image(caption, seed, size=32):
    """Generate deterministic training RGB image; no downloads or licenses needed."""
    if size != 32:
        raise ValueError("Nano MVP supports 32x32 only")
    rng = random.Random(seed)
    label = CAPTIONS[caption]
    color, shape = label.split()
    rgb = {"red": (230, 45, 50), "green": (36, 190, 85), "blue": (45, 90, 225)}[color]
    image = Image.new("RGB", (size, size), (246, 246, 244))
    draw = ImageDraw.Draw(image)
    width = rng.randint(12, 21)
    x = rng.randint(3, size - width - 3)
    y = rng.randint(3, size - width - 3)
    bounds = (x, y, x + width, y + width)
    if shape == "circle":
        draw.ellipse(bounds, fill=rgb)
    else:
        draw.rectangle(bounds, fill=rgb)
    pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


@dataclass
class Config:
    size: int = 32
    channels: int = 32
    diffusion_steps: int = 50
    def __post_init__(self):
        if self.size != 32 or self.channels < 8 or self.diffusion_steps < 2:
            raise ValueError("Unsupported image diffusion configuration")


class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.channels
        self.label = nn.Embedding(len(CAPTIONS), c)
        self.time = nn.Sequential(nn.Linear(1, c), nn.SiLU(), nn.Linear(c, c))
        self.first = nn.Conv2d(3, c, 3, padding=1)
        self.down = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GroupNorm(4, c * 2), nn.SiLU(), nn.Conv2d(c * 2, c * 2, 3, padding=1))
        self.up = nn.Conv2d(c * 3, c, 3, padding=1)
        self.out = nn.Sequential(nn.GroupNorm(4, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, noisy, t, labels):
        if noisy.ndim != 4 or noisy.shape[1:] != (3, 32, 32):
            raise ValueError("Expected [batch,3,32,32] images")
        if t.shape != labels.shape or t.ndim != 1 or len(t) != len(noisy):
            raise ValueError("Batch conditioning mismatch")
        emb = (self.label(labels) + self.time(t.float()[:, None] / self.config.diffusion_steps))[:, :, None, None]
        skip = F.silu(self.first(noisy) + emb)
        hidden = self.down(skip)
        hidden = hidden + self.mid(hidden)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = F.silu(self.up(torch.cat([hidden, skip], dim=1)) + emb)
        return self.out(hidden)


def schedule(config, device):
    beta = torch.linspace(0.0001, 0.02, config.diffusion_steps, device=device)
    alpha = 1 - beta
    return beta, alpha, alpha.cumprod(dim=0)


def training_loss(model, images, labels):
    device = images.device
    _, _, cumulative = schedule(model.config, device)
    t = torch.randint(0, model.config.diffusion_steps, (len(images),), device=device)
    eps = torch.randn_like(images)
    a = cumulative[t][:, None, None, None]
    noisy = a.sqrt() * images + (1 - a).sqrt() * eps
    return F.mse_loss(model(noisy, t, labels), eps)


@torch.no_grad()
def sample(model, label, seed=42, device="cpu"):
    if label < 0 or label >= len(CAPTIONS):
        raise ValueError("Invalid caption ID")
    generator = torch.Generator(device=device).manual_seed(seed)
    beta, alpha, cumulative = schedule(model.config, device)
    x = torch.randn((1, 3, 32, 32), generator=generator, device=device)
    labels = torch.tensor([label], device=device)
    model.eval()
    for i in range(model.config.diffusion_steps - 1, -1, -1):
        t = torch.tensor([i], device=device)
        estimate = model(x, t, labels)
        x = (x - beta[i] / (1 - cumulative[i]).sqrt() * estimate) / alpha[i].sqrt()
        if i:
            x = x + beta[i].sqrt() * torch.randn(x.shape, generator=generator, device=device)
    return x.clamp(-1, 1).cpu()[0]


def save_png(tensor, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if tensor.shape != (3, 32, 32) or not torch.isfinite(tensor).all():
        raise ValueError("Invalid RGB output")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = ((tensor.permute(1, 2, 0).clamp(-1, 1).numpy() + 1) * 127.5).round().astype("uint8")
    Image.fromarray(array, mode="RGB").save(path)

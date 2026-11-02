"""Learned spatial neural field for the six Sora Nano shape classes.

This is NOT open-ended text-to-image generation. It receives an explicit
(x-center, y-center, half-size) layout as well as a discrete class label.
Training images come from the procedural synthetic dataset; in inference
the foreground mask and RGB pixels come exclusively from trained weights.
"""
from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.nn import functional as F

from src.model import CAPTIONS, synthetic_image, save_png


@dataclass
class SpatialConfig:
    hidden: int = 48
    embedding_dim: int = 8
    size: int = 32

    def __post_init__(self):
        if self.hidden < 8 or self.embedding_dim < 2 or self.size != 32:
            raise ValueError("Unsupported spatial field config")


class SpatialField(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SpatialConfig()
        h, d = self.config.hidden, self.config.embedding_dim
        self.embedding = nn.Embedding(len(CAPTIONS), d)
        self.field = nn.Sequential(
            nn.Linear(d + 2, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )
        # Learned colors, not a fixed palette in the renderer.
        self.foreground = nn.Embedding(len(CAPTIONS), 3)
        self.background = nn.Parameter(torch.zeros(3))
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy], dim=-1).float())

    def forward(self, labels, layouts):
        if labels.ndim != 1 or layouts.shape != (len(labels), 3):
            raise ValueError("Expected labels [B], layouts [B,3]")
        b = len(labels)
        coord = self.coords[None].expand(b, -1, -1, -1)
        centers = layouts[:, None, None, :2]
        half_size = layouts[:, None, None, 2:]
        if torch.any(half_size <= 0):
            raise ValueError("Half size must be positive")
        relative = (coord - centers) / half_size
        conditioning = self.embedding(labels)[:, None, None].expand(-1, 32, 32, -1)
        logits = self.field(torch.cat([relative, conditioning], dim=-1)).squeeze(-1)
        color = torch.sigmoid(self.foreground(labels))
        background = torch.sigmoid(self.background)
        rgb = background + (color[:, None, None, :] - background) * logits.sigmoid()[..., None]
        return logits, rgb


def layout_for_seed(seed):
    """Sample placement/scale; does not draw image pixels."""
    rng = random.Random(int(seed))
    width = rng.randint(12, 21)
    left = rng.randint(3, 32 - width - 3)
    top = rng.randint(3, 32 - width - 3)
    return [left + width / 2, top + width / 2, width / 2]


def synthetic_batch(labels, step, seed=2026):
    """Training ONLY: compute supervision using existing synthetic_image."""
    if labels.ndim != 1:
        raise ValueError("Expected [B] labels")
    images, masks, layouts = [], [], []
    bg = torch.tensor([246, 246, 244], dtype=torch.float32) / 255
    for j, label in enumerate(labels.tolist()):
        sample_seed = seed + step * 7919 + j * 13
        image = (synthetic_image(int(label), sample_seed) + 1) / 2
        rgb = image.permute(1, 2, 0)
        mask = ((rgb - bg).abs().amax(dim=-1) > 0.03).float()
        images.append(rgb)
        masks.append(mask)
        layouts.append(layout_for_seed(sample_seed))
    return (torch.stack(images), torch.stack(masks),
            torch.tensor(layouts, dtype=torch.float32))


def spatial_loss(logits, predicted, target_rgb, target_mask):
    segmentation = F.binary_cross_entropy_with_logits(logits, target_mask)
    color = F.l1_loss(predicted, target_rgb)
    return segmentation + 2 * color, segmentation, color


@torch.no_grad()
def render(model, label, seed=42, device="cpu"):
    """Only learned network weights produce output pixels."""
    if not (0 <= label < len(CAPTIONS)):
        raise ValueError("Invalid class label")
    labels = torch.tensor([label], dtype=torch.long, device=device)
    # Independent deterministic layouts per class.
    layout = torch.tensor([layout_for_seed(seed + 1000 * label)],
                          dtype=torch.float32, device=device)
    model.eval()
    logits, rgb = model(labels, layout)
    return rgb[0].cpu(), layout[0].cpu(), logits[0].cpu()


def save_field_png(rgb, path):
    if rgb.shape != (32, 32, 3):
        raise ValueError("Expected HWC RGB image")
    save_png(rgb.permute(2, 0, 1) * 2 - 1, path)

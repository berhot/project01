"""Tiny learned DDPM prior over image layout, not image pixels.

Generates 3D geometry internally from learned weights and seeded Gaussian noise.
Requires a separately trained spatial neural renderer for RGB pixels.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LayoutConfig:
    steps: int = 64
    hidden: int = 64
    label_count: int = 6

    def __post_init__(self):
        if self.steps < 4 or self.hidden < 16 or self.label_count != 6:
            raise ValueError("Unsupported layout diffusion configuration")


class LayoutDenoiser(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LayoutConfig()
        h = self.config.hidden
        self.embedding = nn.Embedding(self.config.label_count, h)
        self.network = nn.Sequential(
            nn.Linear(3 + h + h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3),
        )
        self.timestep = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, noisy, steps, labels):
        if (noisy.ndim != 2 or noisy.shape[1] != 3
                or steps.shape != (len(noisy),)
                or labels.shape != (len(noisy),)):
            raise ValueError("Expected noisy [B,3], timesteps [B], labels [B]")
        v = steps.float() / self.config.steps * (2 * math.pi)
        encoding = torch.stack([v.sin(), v.cos()], dim=-1)
        return self.network(torch.cat((
            noisy, self.timestep(encoding), self.embedding(labels)
        ), dim=-1))


def noise_schedule(config, device):
    t = torch.linspace(0, 1, config.steps + 1, dtype=torch.float64, device=device)
    cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(0.0001, 0.999).float()
    alpha = 1 - beta
    return beta, alpha, alpha.cumprod(dim=0)


def encode_layout(layout):
    """Encode center x/y and half-width, not pixels, to bounded DDPM coordinates."""
    if layout.ndim != 2 or layout.shape[1] != 3:
        raise ValueError("Expected [B,3] layouts")
    width = layout[:, 2] * 2
    min_xy = 3 + width / 2
    span = 26 - width
    if ((span <= 0).any() or (width < 12 - 1e-4).any()
            or (width > 21 + 1e-4).any()):
        raise ValueError("Layout width outside training range")
    fractions = (layout[:, :2] - min_xy[:, None]) / span[:, None]
    fraction_w = (width - 12) / 9
    values = torch.cat((fractions, fraction_w[:, None]), dim=-1)
    if (values < -1e-4).any() or (values > 1 + 1e-4).any():
        raise ValueError("Layout center outside allowed bounds")
    return values.clamp(0, 1) * 2 - 1


def decode_layout(encoded):
    if encoded.ndim != 2 or encoded.shape[1] != 3:
        raise ValueError("Expected [B,3] diffusion samples")
    fractions = (encoded.clamp(-1, 1) + 1) / 2
    width = 12 + fractions[:, 2] * 9
    min_xy = 3 + width / 2
    centers = min_xy[:, None] + (26 - width)[:, None] * fractions[:, :2]
    return torch.cat((centers, (width / 2)[:, None]), dim=1)


def training_loss(model, labels, layouts):
    device = next(model.parameters()).device
    _, _, cumulative = noise_schedule(model.config, device)
    target = encode_layout(layouts.to(device))
    ts = torch.randint(0, model.config.steps, (len(labels),), device=device)
    noise = torch.randn_like(target)
    a = cumulative[ts][:, None]
    noisy = a.sqrt() * target + (1 - a).sqrt() * noise
    return F.mse_loss(model(noisy, ts, labels.to(device)), noise)


@torch.no_grad()
def sample_layout(model, labels, seed=42):
    device = next(model.parameters()).device
    labels = labels.to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    beta, alpha, cumulative = noise_schedule(model.config, device)
    latent = torch.randn((len(labels), 3), generator=generator, device=device)
    model.eval()
    for i in range(model.config.steps - 1, -1, -1):
        ts = torch.full((len(labels),), i, dtype=torch.long, device=device)
        noise = model(latent, ts, labels)
        latent = (latent - beta[i] / (1 - cumulative[i]).sqrt() * noise) / alpha[i].sqrt()
        if i:
            posterior = beta[i] * (1 - cumulative[i - 1]) / (1 - cumulative[i])
            latent += posterior.sqrt() * torch.randn(
                latent.shape, generator=generator, device=device
            )
    return decode_layout(latent).cpu()

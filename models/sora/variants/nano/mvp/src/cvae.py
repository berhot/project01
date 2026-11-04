"""Experimental conditional variational autoencoder for six 32x32 shape prompts.

Separate from the original DDPM, which failed visual evaluation. This network
generates pixels from learned parameters; the procedural drawing code is used
only for training examples, never inside its decoder or sampler.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from src.model import CAPTIONS, synthetic_image, save_png


@dataclass
class CVAEConfig:
    channels: int = 32
    latent_dim: int = 8
    beta: float = 0.05

    def __post_init__(self):
        if self.channels < 8 or self.latent_dim < 1 or not (0 < self.beta <= 1):
            raise ValueError("Invalid CVAE configuration")


class ShapeCVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CVAEConfig()
        c, z = self.config.channels, self.config.latent_dim
        self.embedding = nn.Embedding(len(CAPTIONS), 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(19, c, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c, 2*c, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(2*c, 4*c, 4, 2, 1), nn.SiLU(), nn.Flatten(),
        )
        self.stats = nn.Linear(4*c*4*4, 2*z)
        self.project = nn.Linear(z + 16, c*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 2*c, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(2*c, c, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c, c, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def encode(self, images, labels):
        conditioning = self.embedding(labels)
        extended = conditioning[:, :, None, None].expand(-1, -1, 32, 32)
        mean, logvar = self.stats(self.encoder(torch.cat((images, extended), dim=1))).chunk(2, dim=1)
        return mean, logvar

    def decode(self, z, labels):
        conditioning = self.embedding(labels)
        x = self.project(torch.cat((z, conditioning), dim=1))
        return self.decoder(x.reshape(-1, self.config.channels, 4, 4))

    def forward(self, images, labels):
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32) or labels.shape != (len(images),):
            raise ValueError("Expected [batch,3,32,32] images and [batch] labels")
        mean, logvar = self.encode(images, labels)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return self.decode(z, labels), mean, logvar


def batch_from_synthetic(labels, step):
    """Training data only: do not call this in any inference function."""
    return torch.stack([
        (synthetic_image(int(label), step * 31 + i) + 1) / 2
        for i, label in enumerate(labels.tolist())
    ])


def cvae_loss(predicted, target, mean, logvar, beta=0.05):
    # Colored foreground is rarer than the off-white background.
    foreground_weight = 1 + 3 * (target.amin(dim=1, keepdim=True) < 0.7).float()
    reconstruction = ((predicted - target).abs() * foreground_weight).mean()
    divergence = (-0.5 * (1 + logvar - mean.square() - logvar.exp())).mean()
    return reconstruction + beta * divergence, reconstruction, divergence


@torch.no_grad()
def generate(model, label, seed=42, device="cpu"):
    if not (0 <= label < len(CAPTIONS)):
        raise ValueError("Unsupported prompt label")
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn((1, model.config.latent_dim), generator=gen, device=device)
    labels = torch.tensor([label], device=device)
    return model.decode(latent, labels)[0].cpu()


def save_rgb_png(rgb01, path):
    """Existing PNG writer accepts [-1,1], CVAE decoder emits [0,1]."""
    save_png(rgb01 * 2 - 1, path)

"""Experimental V2.2 geometry supervision for the existing Shape Field MLP.

Training-only differentiable losses; no geometric drawing at inference.
The model, vocabulary and existing V2/V2.1 checkpoint formats are unchanged.
"""
import math

import torch
from torch.nn import functional as F

from src.shape_field_v2 import keyword_ids, SHAPE_WORDS
from src.pixel_diffusion import vocabulary_for


def _edges(tensor):
    if tensor.ndim != 4 or tensor.shape[1:] != (1, 32, 32):
        raise ValueError("Expected shape [B,1,32,32]")
    dx = F.pad((tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs(),
               (0, 1, 0, 0))
    dy = F.pad((tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs(),
               (0, 0, 0, 1))
    return torch.maximum(dx, dy)


def boundary_centroid_objective(logits, targets, tokens,
                                boundary_weight=0.5, centroid_weight=0.05):
    """Shape-balanced differentiable edge and vertical-position penalties.

    Edge L1 is computed at exact training target pixels, not a postprocessing
    renderer. Centroid loss uses expected foreground distribution; its
    normalization and spatial coordinates are resolution-independent.

    Circles receive zero auxiliary gradient to avoid degrading the existing
    strong circle behavior. Their baseline BCE/Dice still train normally.
    """
    if (logits.shape != targets.shape or logits.ndim != 4
            or logits.shape[1:] != (1, 32, 32)
            or tokens.shape != (len(logits), 12)
            or not torch.isfinite(logits).all()
            or not torch.isfinite(targets).all()
            or bool(((targets < 0) | (targets > 1)).any())):
        raise ValueError("Invalid V2.2 logits, targets or tokens")
    if not (math.isfinite(boundary_weight) and math.isfinite(centroid_weight)
            and 0 <= boundary_weight <= 5 and 0 <= centroid_weight <= 5):
        raise ValueError("V2.2 loss weights must be finite and within [0,5]")
    vocab = vocabulary_for("v2")
    word_ids = keyword_ids(tokens, SHAPE_WORDS)
    noncircles = (
        (word_ids != vocab.index("circle")) &
        (word_ids != vocab.index("دائرة"))
    )
    probs = logits.sigmoid()
    edge_loss_per = (_edges(probs) - _edges(targets)).abs().mean((1,2,3))
    # Probability mass centers for differentiable centroid alignment.
    y = torch.linspace(-1, 1, 32, dtype=probs.dtype, device=probs.device)
    true_mass = targets.sum((1,2,3)).clamp_min(1)
    pred_mass = probs.sum((1,2,3)).clamp_min(1e-6)
    true_y = (targets * y[None,None,:,None]).sum((1,2,3))/true_mass
    pred_y = (probs * y[None,None,:,None]).sum((1,2,3))/pred_mass
    center_loss_per = (pred_y - true_y).square()
    weight = noncircles.to(dtype=probs.dtype)
    denominator = weight.sum().clamp_min(1)
    edge_loss = (edge_loss_per * weight).sum()/denominator
    center_loss = (center_loss_per * weight).sum()/denominator
    total = boundary_weight*edge_loss + centroid_weight*center_loss
    return {
        "loss": total,
        "edge_loss": edge_loss.detach(),
        "centroid_loss": center_loss.detach(),
        "supervised_noncircles": int(noncircles.sum()),
    }

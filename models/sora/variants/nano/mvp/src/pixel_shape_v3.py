"""Sora Nano Shape Learning V3: valid-q high-noise bilingual conditioning.

NO synthetic image drawing at inference. Unlike the failed midpoint-noise
pair objective, the correct and wrong shape captions here see exactly the
SAME valid q(x_t | real training x0) and predict the SAME actual epsilon.
A margin ranks the correct text above the swapped shape word, while the
standard epsilon MSE still trains accurate denoising.

This is an opt-in research objective, not a guaranteed triangle fix or an
architectural foundation model upgrade. Keep original weights immutable.
"""
import math

import torch
from torch.nn import functional as F

from src.pixel_diffusion import noise_schedule, vocabulary_for

SHAPE_WORD_PAIRS = (
    ("circle","square"),
    ("square","triangle"),
    ("triangle","square"),
    ("دائرة","مربع"),
    ("مربع","مثلث"),
    ("مثلث","مربع"),
)


def swapped_shape_tokens(tokens):
    """Change ONLY the shape word. All color and language tokens stay fixed."""
    if tokens.ndim!=2 or tokens.shape[1]!=12 or tokens.dtype!=torch.long:
        raise ValueError("Expected integer [B,12] v2 captions")
    words=vocabulary_for("v2")
    indices={w:words.index(w) for pair in SHAPE_WORD_PAIRS for w in pair}
    candidates=[(indices[src],indices[dst]) for src,dst in SHAPE_WORD_PAIRS]
    original=tokens
    changed=original.clone()
    count=torch.zeros(len(tokens),dtype=torch.long,device=tokens.device)
    for source,target in candidates:
        present=(original==source)
        count+=present.sum(dim=1)
        changed[present]=target
    if not bool(torch.all(count==1)):
        raise ValueError("Each V3 caption must contain exactly one known shape word")
    return changed


def high_noise_v3_objective(model,images,tokens,noise_weight=0.15,
                            contrast_weight=0.1,margin=0.003,
                            low_fraction=0.55,high_fraction=0.85):
    """Valid-forward-diffusion shape contrast at timesteps with low shape signal.

    Noise regression on real x_t stays anchored to epsilon; the contrast
    term cannot replace conventional DDPM training. Comparisons use
    the exact SAME x_t and epsilon, differing only in caption shape tokens.
    """
    cfg=model.config
    if cfg.dataset_version!="v2" or cfg.architecture!="rgb-unet-spatial-text-v2":
        raise ValueError("V3 requires the existing bilingual spatial-v2 network")
    if (images.ndim!=4 or images.shape[1:]!=(3,32,32)
            or tokens.shape!=(len(images),12) or len(images)<2):
        raise ValueError("Expected matched [B,3,32,32] RGB and [B,12] captions")
    if (not all(math.isfinite(v) for v in (
            noise_weight,contrast_weight,margin,low_fraction,high_fraction))
            or not (0<=noise_weight<=1 and 0<=contrast_weight<=1
                    and 0<margin<=0.1 and 0<=low_fraction<high_fraction<1)):
        raise ValueError("Invalid V3 high-noise contrast hyperparameters")
    wrong=swapped_shape_tokens(tokens)
    start=int(math.floor(cfg.diffusion_steps*low_fraction))
    end=int(math.floor(cfg.diffusion_steps*high_fraction))
    if end<start or end>=cfg.diffusion_steps:
        raise ValueError("V3 timestep window too narrow for diffusion schedule")
    times=torch.randint(start,end+1,(len(images),),device=images.device)
    true_eps=torch.randn_like(images)
    _,_,cum=noise_schedule(cfg,images.device)
    a=cum[times][:,None,None,None]
    x_t=a.sqrt()*images+(1-a).sqrt()*true_eps
    prediction=model(torch.cat((x_t,x_t)),torch.cat((times,times)),
                     torch.cat((tokens,wrong)))
    actual,swapped=prediction.chunk(2,dim=0)
    true_error=(actual-true_eps).square().mean(dim=(1,2,3))
    wrong_error=(swapped-true_eps).square().mean(dim=(1,2,3))
    ranking=F.relu(margin+true_error-wrong_error)
    noise_mse=true_error.mean()
    contrast=ranking.mean()
    total=noise_weight*noise_mse+contrast_weight*contrast
    return {
        "total_loss":total,
        "high_noise_mse":noise_mse.detach(),
        "contrast_loss":contrast.detach(),
        "correct_shape_word_win_rate":(true_error<wrong_error).float().mean().detach(),
        "correct_shape_word_mse_advantage":(wrong_error-true_error).mean().detach(),
        "sampled_times":times.detach(),
        "correct_tokens":tokens.detach(),
        "wrong_tokens":wrong.detach(),
    }

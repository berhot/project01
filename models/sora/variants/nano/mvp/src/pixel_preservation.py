"""Optional old-concept teacher preservation for Sora Nano v2 fine-tuning.

A frozen copy of the trained source checkpoint provides a denoising
reference for original CIRCLE/SQUARE prompts, including null CFG prompts.
This regularization is opt-in; it does not alter generation, datasets,
checkpoints, or the legacy trainer by default.

The teacher and student see the SAME ordinary q(x_t | x_0) noise sample.
This is distinct from the special midpoint-noise counterfactual task.
"""
import math

import torch
from torch.nn import functional as F

from src.pixel_diffusion import noise_schedule, vocabulary_for


def old_shape_mask(tokens):
    """Select rows containing an original English/Arabic shape word."""
    if tokens.ndim != 2 or tokens.shape[1] != 12:
        raise ValueError("Expected [B,12] text tokens")
    words=vocabulary_for("v2")
    ids=(words.index("circle"),words.index("square"),
         words.index("دائرة"),words.index("مربع"))
    return torch.stack([(tokens==key).any(dim=1) for key in ids]).any(dim=0)


def preservation_objective(student, teacher, images, tokens, null_weight=0.5):
    """Mean squared teacher-student epsilon differences on old concepts.

    The teacher must be the SAME architecture/config as the student and
    frozen. Only original circle/square samples are protected, to permit
    learning new triangular geometry. Both conditional and unconditional
    paths are anchored so CFG behavior cannot drift without penalty.
    """
    if student.config != teacher.config:
        raise ValueError("Teacher and student configs must match exactly")
    if not 0 <= null_weight <= 2 or not math.isfinite(null_weight):
        raise ValueError("Invalid null-prompt preservation weight")
    if (images.ndim != 4 or images.shape[1:] != (3,32,32)
            or tokens.shape != (len(images),12)):
        raise ValueError("Expected RGB images and corresponding [B,12] tokens")
    selected=old_shape_mask(tokens)
    if not bool(selected.any()):
        raise ValueError("No circle/square examples to protect in batch")
    x0=images[selected]
    text=tokens[selected]
    device=x0.device
    t=torch.randint(0,student.config.diffusion_steps,(len(x0),),device=device)
    eps=torch.randn_like(x0)
    _,_,cumulative=noise_schedule(student.config,device)
    a=cumulative[t][:,None,None,None]
    xt=a.sqrt()*x0+(1-a).sqrt()*eps
    zeros=torch.zeros_like(text)
    x_two=torch.cat((xt,xt))
    time_two=torch.cat((t,t))
    text_two=torch.cat((text,zeros))
    with torch.no_grad():
        teacher.eval()
        reference=teacher(x_two,time_two,text_two).detach()
    predicted=student(x_two,time_two,text_two)
    count=len(x0)
    conditional=F.mse_loss(predicted[:count],reference[:count])
    unconditional=F.mse_loss(predicted[count:],reference[count:])
    return {
        "total_loss":conditional+null_weight*unconditional,
        "conditional_loss":conditional.detach(),
        "null_loss":unconditional.detach(),
        "protected_examples":int(count),
    }

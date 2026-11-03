"""Counterfactual square-vs-triangle pixel diffusion research (v2 only).

A matched pair shares a procedural *training* image's color/background/
location/size. Only the filled silhouette and shape word differ.
At inference there is no procedural renderer or shape supplied: the RGB
diffusion model still denoises Gaussian pixels using learned weights.
"""
import math

import torch
from torch.nn import functional as F

from src.pixel_dataset_v2 import COLORS, synthetic_example_v2
from src.pixel_diffusion import noise_schedule


def matched_shape_batch(step, pairs, seed=42, split="train"):
    """[P,2,3,32,32] images and [P,2,12] tokens; order square, triangle.

    Same sample seed+color+language+variant ensures the PIL generator draws
    matching background/box/size for both training targets. This is a
    supervised training-data construction ONLY.
    """
    if step < 1 or pairs < 1:
        raise ValueError("step and pairs must be positive")
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    colors=tuple(COLORS)
    images,tokens,details=[],[],[]
    for i in range(pairs):
        color=colors[(step*3+i)%len(colors)]
        # Rotate language INDEPENDENTLY of color parity across steps.
        # Using (step+i)%2 here would confound colors with languages,
        # because color index is (3*step+i)%4.
        language=("en","ar")[((step//2)+i)%2]
        sample_seed=seed+step*5419+i*47
        pair=[]
        pair_tokens=[]
        for shape in ("square","triangle"):
            image,word_ids,caption,concept=synthetic_example_v2(
                sample_seed,color=color,shape=shape,language=language,
                split=split,variant=0,
            )
            pair.append(image)
            pair_tokens.append(word_ids)
        images.append(torch.stack(pair))
        tokens.append(torch.stack(pair_tokens))
        details.append({"color":color,"language":language,"sample_seed":sample_seed})
    return torch.stack(images),torch.stack(tokens),details


def _silhouette(image):
    """Continuous foreground mask from RGB color saturation [-1,1]."""
    saturation=image.amax(dim=1)-image.amin(dim=1)
    return torch.sigmoid((saturation-0.45)*8.0)


def make_counterfactual_noise(config, paired_images, generator=None,
                              low_frac=0.35, high_frac=0.70):
    """Generate SAME x_t for square and triangle, with distinct valid eps targets.

    For each x0_i, eps_i=(x_t-sqrt(a_t)*x0_i)/sqrt(1-a_t).
    The paired x_t is centered on their image midpoint, so it is
    plausible under both q(x_t | x0_i) at a moderate noise level.
    """
    if (paired_images.ndim!=5 or paired_images.shape[1:]!=(2,3,32,32)):
        raise ValueError("Expected matched [pairs,2,3,32,32] samples")
    if not 0 < low_frac < high_frac < 1:
        raise ValueError("Invalid noise interval")
    device=paired_images.device
    total=config.diffusion_steps
    low=max(1,int(total*low_frac))
    high=min(total-2,max(low+1,int(total*high_frac)))
    if high < low:
        raise ValueError("Insufficient diffusion timesteps for paired objective")
    n=len(paired_images)
    if generator is None:
        timesteps=torch.randint(low,high+1,(n,),device=device)
        eps=torch.randn((n,3,32,32),device=device,dtype=paired_images.dtype)
    else:
        timesteps=torch.randint(low,high+1,(n,),device=device,generator=generator)
        eps=torch.randn((n,3,32,32),device=device,dtype=paired_images.dtype,
                        generator=generator)
    _,_,cumulative=noise_schedule(config,device)
    a=cumulative[timesteps][:,None,None,None]
    midpoint=paired_images.mean(dim=1)
    xt=a.sqrt()*midpoint+(1-a).sqrt()*eps
    noise_targets=(xt[:,None]-a[:,None].sqrt()*paired_images)/(
        (1-a)[:,None].sqrt()
    )
    return xt,timesteps,noise_targets


def matched_shape_objective(model,paired_images,paired_tokens,
                            silhouette_weight=0.5,generator=None):
    """Train both captions against one noisy RGB input and different shapes.

    MSE teaches prompt-specific epsilon; silhouette-difference loss focuses
    precisely on pixels in which square/triangle foreground differs.
    Does not replace normal denoising/CFG training; add as a weighted
    auxiliary loss. There is no procedural geometry at inference.
    """
    if model.config.dataset_version!="v2":
        raise ValueError("Requires bilingual dataset v2")
    if not math.isfinite(silhouette_weight) or not 0 <= silhouette_weight <= 2:
        raise ValueError("Invalid silhouette_weight")
    if paired_tokens.shape!=(len(paired_images),2,12):
        raise ValueError("Expected paired token shapes")
    xt,timesteps,targets=make_counterfactual_noise(
        model.config,paired_images,generator=generator
    )
    n=len(paired_images)
    predicted_eps=model(
        torch.cat((xt,xt),dim=0),
        torch.cat((timesteps,timesteps),dim=0),
        torch.cat((paired_tokens[:,0],paired_tokens[:,1]),dim=0),
    )
    predicted_eps=torch.stack(predicted_eps.split(n,dim=0),dim=1)
    # Matched-pair epsilon targets are for *identical* noisy pixels, so a
    # shape-blind denoiser cannot satisfy both targets simultaneously.
    noise_loss=F.mse_loss(predicted_eps,targets)
    _,_,cumulative=noise_schedule(model.config,paired_images.device)
    a=cumulative[timesteps][:,None,None,None,None]
    reconstruct=(
        xt[:,None]-(1-a).sqrt()*predicted_eps
    )/a.sqrt()
    predict_masks=_silhouette(reconstruct.flatten(0,1)).reshape(
        n,2,32,32
    )
    # Build target from the training images themselves, not class templates.
    ground_masks=(
        paired_images.amax(dim=2)-paired_images.amin(dim=2)>0.45
    ).float()
    # Give more importance to triangle-cutout corners than equal background.
    unequal=(ground_masks[:,0]-ground_masks[:,1]).abs()
    difference=(predict_masks[:,0]-predict_masks[:,1])
    target_difference=(ground_masks[:,0]-ground_masks[:,1])
    mask_difference_loss=(
        ((difference-target_difference).square()*(0.25+3*unequal)).mean()
    )
    # Also discourage individual conditional foreground being misplaced.
    individual_mask_loss=F.binary_cross_entropy(
        predict_masks.clamp(1e-5,1-1e-5),ground_masks
    )
    shape_loss=mask_difference_loss+0.25*individual_mask_loss
    return {
        "total_loss":noise_loss+silhouette_weight*shape_loss,
        "pair_noise_mse":noise_loss.detach(),
        "pair_shape_loss":shape_loss.detach(),
        "pair_difference_loss":mask_difference_loss.detach(),
    }


@torch.no_grad()
def paired_prompt_probe(model, seed=42, pairs=8, step=1):
    """Read-only diagnostic: how well do prompts separate matched silhouettes?

    Uses validation synthetic pairs with true targets solely to *measure*
    conditional denoising; never trains weights or draws images at inference.
    """
    model.eval()
    images,tokens,details=matched_shape_batch(
        step,pairs,seed,split="validation"
    )
    device=next(model.parameters()).device
    images,tokens=images.to(device),tokens.to(device)
    rng=torch.Generator(device=device).manual_seed(seed)
    xt,times,targets=make_counterfactual_noise(
        model.config,images,generator=rng
    )
    n=len(images)
    predicted=model(
        torch.cat([xt,xt]),
        torch.cat([times,times]),
        torch.cat([tokens[:,0],tokens[:,1]]),
    )
    predicted=torch.stack(predicted.split(n),dim=1)
    expected_delta=targets[:,0]-targets[:,1]
    predicted_delta=predicted[:,0]-predicted[:,1]
    delta_norm=float(predicted_delta.square().mean().sqrt())
    expected_norm=float(expected_delta.square().mean().sqrt())
    alignment=float(F.cosine_similarity(
        predicted_delta.flatten(1),expected_delta.flatten(1),dim=1
    ).mean())
    return {
        "pairs":pairs,"step":step,"seed":seed,
        "expected_epsilon_difference_rms":round(expected_norm,6),
        "predicted_epsilon_difference_rms":round(delta_norm,6),
        "prompt_difference_ratio":round(
            delta_norm/max(expected_norm,1e-6),6
        ),
        "shape_difference_alignment":round(alignment,6),
        "scope":"no-retrain synthetic matched validation pairs; not a generated-image metric",
        "pair_details":details,
    }

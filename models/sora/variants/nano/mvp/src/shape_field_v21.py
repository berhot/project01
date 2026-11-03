"""Opt-in V2.1 shape-word separation for the existing Sora neural mask field.

The V2 80-step pilot learned circles but generated 0/8 squares and 0/8
triangles. This recipe isolates an *experimental* fix: the SAME observed
bbox/layout must correspond to three DIFFERENT training masks when ONLY
the caption's shape token changes.

Training masks are synthesized in the on-demand TRAIN split and never
used during model inference. This is not a procedural inference renderer.
"""
import math

import torch
from torch.nn import functional as F

from src.pixel_dataset_v2 import (
    COLORS,SHAPES,caption_v2,synthetic_example_v2,tokenize_v2
)
from src.shape_field_v2 import (layout_from_mask,draw_layout_from_noise,keyword_ids,SHAPE_WORDS)
from src.pixel_diffusion import vocabulary_for
from src.shape_first import mask_from_rgb


def matched_layout_shape_batch(step,examples_per_step,seed=42,device="cpu"):
    """Build matching-layout 3-shape / 2-language targets per base example.

    The shared bbox comes from a square training image. Orientation is
    extracted from a triangle training image with the exact same seed;
    it is repeated for all 3 shapes. Thus *layout alone cannot reveal
    shape class*, and only learned text conditioning can disambiguate it.
    """
    if step<1 or not 1<=examples_per_step<=32:
        raise ValueError("Expected positive step and 1..32 paired examples")
    layouts,truths,captions,concepts=[],[],[],[]
    color_names=tuple(COLORS)
    for index in range(examples_per_step):
        color=color_names[(step+index)%len(color_names)]
        sample_seed=seed+step*8111+index*137
        training_images={}
        for shape in SHAPES:
            image,_,_,_=synthetic_example_v2(
                sample_seed,color=color,shape=shape,language="en",
                split="train",variant=0)
            training_images[shape]=image
        squares=mask_from_rgb(training_images["square"].unsqueeze(0))
        triangles=mask_from_rgb(training_images["triangle"].unsqueeze(0))
        same_layout=layout_from_mask(squares)[0].clone()
        same_layout[4]=layout_from_mask(triangles)[0,4]
        for shape in SHAPES:
            target_mask=mask_from_rgb(
                training_images[shape].unsqueeze(0))[0]
            for language in ("en","ar"):
                layouts.append(same_layout)
                truths.append(target_mask)
                captions.append(tokenize_v2(caption_v2(
                    color,shape,language,variant=0)))
                concepts.append((color,shape,language))
    return (torch.stack(layouts).to(device),
            torch.stack(truths).to(device),
            torch.stack(captions).to(device),concepts)


def shape_word_separation_objective(model,layouts,targets,tokens,
                                    ranking_margin=0.03,
                                    ranking_weight=0.1):
    """Supervise three shape outputs for each layout, not shape averaging.

    Each consecutive six examples have order circle EN/AR, square EN/AR,
    triangle EN/AR. Correct mask BCE+Dice is trained for all. A separate
    ranking term asks correct token masks to fit target better than
    counterfactual wrong-token masks at the SAME exact layout.

    Margin uses per-example pixel BCE; no output shapes are manufactured
    from templates. The pair target images are only TRAINING data.
    """
    if (layouts.ndim!=2 or layouts.shape[1]!=5 or
            targets.ndim!=4 or targets.shape[1:]!=(1,32,32) or
            len(targets)==0 or len(targets)%6!=0 or
            tokens.shape!=(len(targets),12)):
        raise ValueError("Expected complete same-layout 3-shape x 2-language groups")
    if (not math.isfinite(ranking_margin) or not math.isfinite(ranking_weight)
            or not 0<=ranking_margin<=1 or not 0<=ranking_weight<=1):
        raise ValueError("Invalid V2.1 shape ranking settings")
    correct_logits=model(layouts,tokens)
    bce=F.binary_cross_entropy_with_logits(correct_logits,targets,
                                            reduction="none").mean((1,2,3))
    probability=correct_logits.sigmoid()
    dice=1-(2*(probability*targets).sum((1,2,3))+1)/(
        probability.sum((1,2,3))+targets.sum((1,2,3))+1)
    recon=bce.mean()+dice.mean()

    # Shape label rotates within each 6-entry same-layout group;
    # color and language stay identical.
    shape_group=len(targets)//6
    indices=torch.arange(len(targets),device=targets.device).reshape(shape_group,3,2)
    wrong_indices=indices.roll(shifts=1,dims=1).reshape(-1)
    wrong_tokens=tokens[wrong_indices]
    wrong_logits=model(layouts,wrong_tokens)
    wrong_bce=F.binary_cross_entropy_with_logits(
        wrong_logits,targets,reduction="none").mean((1,2,3))
    ranking=F.relu(ranking_margin+bce-wrong_bce).mean()
    result=recon+ranking_weight*ranking
    return {
        "loss":result,
        "reconstruction_loss":recon.detach(),
        "shape_rank_loss":ranking.detach(),
        "correct_vs_wrong_word_bce_advantage":
            (wrong_bce-bce).mean().detach(),
        "correct_shape_word_win_rate":
            (bce<wrong_bce).float().mean().detach(),
        "correct_logits":correct_logits,
        "wrong_tokens":wrong_tokens.detach(),
    }


def circle_teacher_preservation(model,teacher,images,tokens,seed):
    """Protect V2 circles at both training bboxes and new random layouts.

    Teacher is the immutable frozen V2 learned mask model. Shapes other than
    circle are learned via the V2.1 paired-shape objective. This does NOT
    guarantee preserving independent generated image quality: actual
    gallery and strict shape gate must still be checked.
    """
    if model.config!=teacher.config:
        raise ValueError("Circle-preservation teacher must match configuration")
    shapes=keyword_ids(tokens,SHAPE_WORDS)
    vocab=vocabulary_for("v2")
    selected=(shapes==vocab.index("circle"))|(shapes==vocab.index("دائرة"))
    if int(selected.sum())<1:
        raise ValueError("Circle-preservation batch needs real circle examples")
    circle_tokens=tokens[selected]
    truth=mask_from_rgb(images[selected])
    observed=layout_from_mask(truth)
    random_layouts=draw_layout_from_noise(
        len(circle_tokens),seed,device=images.device)
    layouts=torch.cat((observed,random_layouts),dim=0)
    words=torch.cat((circle_tokens,circle_tokens),dim=0)
    with torch.no_grad():
        expected=teacher(layouts,words).sigmoid()
    predicted=model(layouts,words).sigmoid()
    difference=F.mse_loss(predicted,expected)
    return {"loss":difference,"examples":len(words)}

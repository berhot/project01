"""Read-only stage isolation for an existing Sora shape-first checkpoint run.

Compares (1) held-out true masks vs posterior reconstructions, (2) masks
generated from the unconditional Gaussian prior, (3) RGB appearance from
the *ground-truth* mask, and (4) end-to-end learned RGB generation.

Ground-truth masks are used ONLY for diagnostics; no target masks enter
the end-to-end sampling path. This NEVER trains, promotes or rewrites
existing checkpoints. Output is a new JSON report in central artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.shape_first import load_pipeline
from src.pixel_dataset_v2 import COLORS, SHAPES, caption_v2, synthetic_example_v2
from src.pixel_diffusion import tokenize
from src.pixel_metrics import analyze_image
from src.pixel_shape_quality import (
    MINIMUM_IOU, MINIMUM_SEPARATION, _shape_templates, _iou
)
from src.shape_first import mask_from_rgb
from src.storage_paths import sora_run


def score_binary_mask(mask, expected_shape):
    """Strict reference-only score on the actual learned *binary mask*."""
    if expected_shape not in SHAPES:
        raise ValueError("Expected one of the three v2 shapes")
    if not isinstance(mask,torch.Tensor) or mask.shape not in (
            (32,32),(1,32,32)):
        raise ValueError("Expected [32,32] or [1,32,32] mask")
    if not torch.isfinite(mask).all():
        raise ValueError("Nonfinite mask")
    arr=mask.detach().cpu().squeeze().numpy()>0.5
    ys,xs=np.where(arr)
    area=int(arr.sum())
    result={
        "foreground_pixels":area,
        "fraction_of_image":round(area/1024,5),
        "expected_mask_iou":0.0,
        "shape_separation":0.0,
        "strict_geometry_pass":False,
        "touches_border":True,
    }
    if len(xs)<20:
        return result
    left,right=int(xs.min()),int(xs.max())
    top,bottom=int(ys.min()),int(ys.max())
    if min(right-left+1,bottom-top+1)<7:
        return result
    border=(left==0 or right==31 or top==0 or bottom==31)
    templates=_shape_templates((left,top,right,bottom))
    scores={
        shape: max(_iou(arr,reference) for reference in candidate_masks)
        for shape,candidate_masks in templates.items()
    }
    iou=float(scores[expected_shape])
    separation=iou-max(value for shape,value in scores.items()
                       if shape!=expected_shape)
    result.update({
        "expected_mask_iou":round(iou,5),
        "shape_separation":round(separation,5),
        "touches_border":bool(border),
        "strict_geometry_pass":bool(
            iou>=MINIMUM_IOU[expected_shape]
            and separation>=MINIMUM_SEPARATION
            and not border
        ),
    })
    return result


@torch.no_grad()
def diagnose(run_dir,seed=42,device="cpu"):
    mask_model,rgb_model,manifest=load_pipeline(run_dir,device)
    mask_model.eval()
    rgb_model.eval()
    rows=[]
    for shape_index,shape in enumerate(SHAPES):
        for color_index,color in enumerate(COLORS):
            for language_index,language in enumerate(("en","ar")):
                prompt=caption_v2(color,shape,language,0)
                tokens=tokenize(prompt,dataset_version="v2").unsqueeze(0).to(device)
                # Held-out target is for evaluating independently-trained stages
                # ONLY, not to help the actual learned sampling path.
                target,_,_,_=synthetic_example_v2(
                    seed+shape_index*1709+color_index*311+language_index*83,
                    color=color,shape=shape,language=language,
                    split="validation",variant=0
                )
                rgb_true=target.unsqueeze(0).to(device)
                truth=mask_from_rgb(rgb_true)
                # 1. Deterministic posterior mask reconstruction: no z sampling
                mu,logvar=mask_model.encode(truth)
                posterior_soft=mask_model.decode(mu,tokens).sigmoid()
                posterior_hard=(posterior_soft>=0.5).float()
                intersection=(truth*posterior_hard).sum()
                union=((truth+posterior_hard)>0).float().sum()
                posterior_iou=float(intersection/union) if union else 1.0
                # 2. Pure Gaussian-prior generation: no target whatsoever
                prior_hard,prior_soft=mask_model.generate(
                    tokens,seed=seed+color_index
                )
                # 3. RGB appearance independently on a perfect ground-truth
                # mask. Diagnostic only, NOT model inference.
                appearance_true=rgb_model(truth,tokens)
                detected=analyze_image(
                    appearance_true[0].cpu(),dataset_version="v2"
                )
                mae=float((appearance_true-rgb_true).abs().mean())
                # 4. End-to-end RGB with exactly the model-generated mask.
                appearance_prior=rgb_model(prior_hard.to(device),tokens)
                final=analyze_image(
                    appearance_prior[0].cpu(),dataset_version="v2"
                )
                geom=score_binary_mask(prior_hard[0],shape)
                rows.append({
                    "shape":shape,"color":color,"language":language,
                    "prompt":prompt,
                    "validation_gt_foreground_pixels":int(truth.sum()),
                    "mask_posterior_iou_vs_heldout_gt":round(posterior_iou,5),
                    "mask_posterior_foreground_pixels":int(posterior_hard.sum()),
                    "mask_prior_foreground_pixels":int(prior_hard.sum()),
                    "mask_prior_soft_mean":round(float(prior_soft.mean()),5),
                    "mask_prior_soft_min":round(float(prior_soft.min()),5),
                    "mask_prior_soft_max":round(float(prior_soft.max()),5),
                    "mask_prior_strict_geometry":geom,
                    "appearance_true_mask_rgb_mae":round(mae,5),
                    "appearance_true_mask_predicted_color":detected["color"],
                    "appearance_true_mask_correct_color":bool(
                        detected["color"]==color),
                    "appearance_generated_mask_predicted_color":final["color"],
                    "appearance_generated_mask_correct_color":bool(
                        final["color"]==color),
                })
    if len(rows)!=24:
        raise RuntimeError("Expected 24 diagnostics")
    shape_summaries={}
    for shape in SHAPES:
        subset=[row for row in rows if row["shape"]==shape]
        mean=lambda key:round(sum(row[key] for row in subset)/len(subset),5)
        shape_summaries[shape]={
            "count":8,
            "mean_posterior_iou_vs_heldout_gt":
                mean("mask_posterior_iou_vs_heldout_gt"),
            "mean_gt_foreground_pixels":mean("validation_gt_foreground_pixels"),
            "mean_posterior_foreground_pixels":
                mean("mask_posterior_foreground_pixels"),
            "mean_prior_foreground_pixels":mean("mask_prior_foreground_pixels"),
            "prior_strict_shape_passes":sum(
                row["mask_prior_strict_geometry"]["strict_geometry_pass"]
                for row in subset),
            "clean_mask_rgb_color_passes":sum(
                row["appearance_true_mask_correct_color"] for row in subset),
            "generated_mask_rgb_color_passes":sum(
                row["appearance_generated_mask_correct_color"] for row in subset),
            "mean_clean_mask_rgb_mae":mean("appearance_true_mask_rgb_mae"),
        }
    return {
        "schema_version":1,
        "task":"sora-shape-first-isolated-stage-diagnosis",
        "source_run_dir":str(run_dir),
        "mask_sha256":manifest["sha256"]["mask.pt"],
        "appearance_sha256":manifest["sha256"]["appearance.pt"],
        "validation":"held-out synthetic v2 RGB targets; targets never provided to prior sampling",
        "sample_seed":seed,
        "no_retraining":True,
        "no_release":True,
        "summary_by_shape":shape_summaries,
        "items":rows,
        "interpretation_guide":{
            "low_posterior_iou":"Mask reconstruction itself is not learned; more steps alone are not proof of a fix.",
            "strong_posterior_weak_prior":"Mask CVAE latent prior and encoded posterior mismatch; random-z generation fails.",
            "weak_rgb_from_true_mask":"Appearance network is undertrained or ignores prompt colors even with perfect masks.",
            "strong_true_mask_rgb_weak_generated_mask_rgb":"Generated mask distribution may differ from the appearance network training masks.",
            "all_metrics":"Visual inspection of both mask and RGB galleries remains necessary."
        }
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    report=diagnose(args.run_dir,args.seed,args.device)
    destination=Path(args.output) if args.output else sora_run(
        "shape-first-diagnose")/"report.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8") as out:
        json.dump(report,out,indent=2,ensure_ascii=False)
        out.write("\n")
    print(json.dumps({
        "report":str(destination),
        "summary_by_shape":report["summary_by_shape"],
        "no_retraining":True,
        "no_release":True,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

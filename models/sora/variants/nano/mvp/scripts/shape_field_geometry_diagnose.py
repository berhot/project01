"""Read-only, held-out Sora Shape Field geometry audit and matched comparison.

Separately test:
  1. actual learned mask on the EXACT bbox/layout of held-out real image;
  2. learned mask on Gaussian-sampled prior layouts, with NO target mask;
  3. sensitivity to binary occupancy threshold (0.4, 0.5, 0.6).

Ground-truth masks/layouts may be used ONLY for diagnostic #1, never
generated inference. No source checkpoint writes, shape templates at
inference, training, releases, optimizer, or acceptance based on loss.

Unlike a generic "more training" suggestion, this identifies whether
shapes are already correctly modeled at observed layouts but fail to
generalize to sampled layouts, or are underfit even when positioned.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.shape_field_v2 import load_run
from scripts.shape_first_diagnose import score_binary_mask
from src.pixel_dataset_v2 import COLORS, SHAPES, caption_v2, synthetic_example_v2
from src.pixel_diffusion import tokenize
from src.shape_field_v2 import draw_layout_from_noise, layout_from_mask
from src.shape_first import mask_from_rgb
from src.storage_paths import sora_run

THRESHOLDS=(0.4,0.5,0.6)


def _mask_iou(prediction,truth):
    predicted=prediction>0.5
    expected=truth>0.5
    intersection=int((predicted & expected).sum())
    union=int((predicted | expected).sum())
    return round(intersection/union if union else 1.0,5)


@torch.no_grad()
def diagnose(run_dir,seed=42,device="cpu"):
    mask_model,_,manifest=load_run(run_dir,device=device)
    mask_model.eval()
    rows=[]
    for shape_index,shape in enumerate(SHAPES):
        for color_index,color in enumerate(COLORS):
            for lang_index,language in enumerate(("en","ar")):
                prompt=caption_v2(color,shape,language,0)
                tokens=tokenize(prompt,dataset_version="v2")[None].to(device)
                # Independently seeded validation pixels, NEVER the training set.
                ground_truth,_,_,_=synthetic_example_v2(
                    seed+shape_index*1709+color_index*311+lang_index*83,
                    color=color,shape=shape,language=language,
                    split="validation",variant=0,
                )
                mask=mask_from_rgb(ground_truth[None].to(device))
                seen_layout=layout_from_mask(mask)
                seen_probability=mask_model(seen_layout,tokens).sigmoid()
                prior_layout=draw_layout_from_noise(
                    1,seed+color_index,device=device)
                prior_probability=mask_model(prior_layout,tokens).sigmoid()
                records={}
                for label,probability in (
                    ("heldout_layout",seen_probability),
                    ("sampled_layout",prior_probability),
                ):
                    evaluations={}
                    for threshold in THRESHOLDS:
                        candidate=(probability>=threshold).float()
                        metrics=score_binary_mask(candidate[0],shape)
                        metrics["threshold"]=threshold
                        if label=="heldout_layout":
                            metrics["mask_iou_vs_ground_truth"]=_mask_iou(
                                candidate,mask)
                        evaluations[str(threshold)]=metrics
                    records[label]=evaluations
                bce=F.binary_cross_entropy(seen_probability,mask)
                rows.append({
                    "shape":shape,"color":color,"language":language,
                    "prompt":prompt,"validation_seed":seed+shape_index*1709+
                        color_index*311+lang_index*83,
                    "heldout_mask_bce":round(float(bce),6),
                    "truth_foreground_pixels":int(mask.sum()),
                    "heldout_foreground_probability_mean":
                        round(float(seen_probability.mean()),6),
                    "sampled_foreground_probability_mean":
                        round(float(prior_probability.mean()),6),
                    "threshold_diagnostics":records,
                })
    if len(rows)!=24:
        raise RuntimeError("Expected all 24 validation concept/language combinations")
    by_shape={}
    for shape in SHAPES:
        subset=[r for r in rows if r["shape"]==shape]
        threshold_report={}
        for threshold in THRESHOLDS:
            threshold_key=str(threshold)
            held=[r["threshold_diagnostics"]["heldout_layout"][threshold_key]
                  for r in subset]
            generated=[r["threshold_diagnostics"]["sampled_layout"][threshold_key]
                       for r in subset]
            threshold_report[threshold_key]={
                "heldout_iou_vs_truth_mean":round(sum(
                    item["mask_iou_vs_ground_truth"] for item in held)/8,5),
                "heldout_strict_shape_passes":sum(
                    item["strict_geometry_pass"] for item in held),
                "sampled_strict_shape_passes":sum(
                    item["strict_geometry_pass"] for item in generated),
                "heldout_mean_foreground_pixels":round(sum(
                    item["foreground_pixels"] for item in held)/8,4),
                "sampled_mean_foreground_pixels":round(sum(
                    item["foreground_pixels"] for item in generated)/8,4),
            }
        by_shape[shape]={
            "samples":8,
            "mean_heldout_mask_bce":round(sum(
                r["heldout_mask_bce"] for r in subset)/8,6),
            "mean_true_foreground_pixels":round(sum(
                r["truth_foreground_pixels"] for r in subset)/8,3),
            "thresholds":threshold_report,
        }
    return {
        "schema_version":1,"family":"sora",
        "diagnostic":"shape-field-heldout-vs-prior-layout",
        "source_run_dir":str(run_dir),
        "mask_checksum_sha256":manifest["sha256"]["mask_field.pt"],
        "training_recipe":manifest.get("training_recipe","original-v2"),
        "sample_seed":seed,
        "validation_split":"held-out synthetic v2, never fed to prior inference",
        "no_training":True,"release_approved":False,
        "by_shape":by_shape,"items":rows,
        "interpretation":{
            "weak_both":"Poor IoU on held-out true bbox AND poor sampled shape implies underfit/capacity or conditioning failure.",
            "strong_heldout_weak_prior":"Good IoU when given true bbox but poor sampled layouts implies layout prior/generalization mismatch.",
            "threshold_only":"Threshold effect means mask probability calibration matters; do not tune threshold based only on test examples.",
            "shape_quality":"A high foreground IoU is not equivalent to visually recognizable sharp shape.",
        }
    }


def compare(baseline,candidate):
    if (baseline["diagnostic"]!=candidate["diagnostic"] or
            baseline["sample_seed"]!=candidate["sample_seed"] or
            len(baseline["items"])!=24 or len(candidate["items"])!=24):
        raise ValueError("Comparisons require matched diagnostic and seeds")
    key=lambda x:(x["shape"],x["color"],x["language"],x["validation_seed"])
    if set(map(key,baseline["items"]))!=set(map(key,candidate["items"])):
        raise ValueError("Unmatched held-out samples and captions")
    results={}
    for shape in SHAPES:
        old=baseline["by_shape"][shape]
        new=candidate["by_shape"][shape]
        a=old["thresholds"]["0.5"]
        b=new["thresholds"]["0.5"]
        results[shape]={
            "heldout_ground_truth_iou_change":round(
                b["heldout_iou_vs_truth_mean"]-a["heldout_iou_vs_truth_mean"],5),
            "heldout_strict_pass_change":
                b["heldout_strict_shape_passes"]-a["heldout_strict_shape_passes"],
            "sampled_strict_pass_change":
                b["sampled_strict_shape_passes"]-a["sampled_strict_shape_passes"],
            "baseline":a,"candidate":b,
        }
    return {
        "release_approved":False,
        "comparison_by_shape":results,
        "note":"Read-only diagnostic comparison, not proof of improvement or automatic model promotion.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--baseline-run-dir",default=None)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    report=diagnose(args.run_dir,args.seed,args.device)
    if args.baseline_run_dir is not None:
        baseline=diagnose(args.baseline_run_dir,args.seed,args.device)
        report["comparison"]=compare(baseline,report)
    output=Path(args.output) if args.output else sora_run(
        "shape-field-geometry-diagnose")/"report.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    print(json.dumps({
        "report":str(output),
        "source_run_dir":report["source_run_dir"],
        "by_shape":report["by_shape"],
        "comparison":report.get("comparison"),
        "no_training":True,"release_approved":False,
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

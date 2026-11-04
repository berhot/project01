"""Read-only geometry audit on real forward-diffused held-out toy images.

Unlike final-generation metrics, this isolates whether an existing RGB
denoiser can recover a triangle from x_t under the correct vs swapped
shape caption, using the SAME Gaussian noise. The ground-truth mask
comes from the held-out RGB data, not a procedural inference shortcut.

The clipping, noise schedule, and classifier-free guidance are exactly
the production fixed-x0 posterior-mean model operations. No training
or checkpoint edits. This is a toy 32x32 segmentation diagnostic.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.pixel_conditioning_audit import validation_items
from scripts.pixel_diffusion import load_model, require_guidance_training
from src.storage_paths import sora_run
from src.pixel_diffusion import guided_noise, noise_schedule
from src.pixel_metrics import analyze_image


def foreground(image):
    """Boolean foreground from normalized RGB [B,3,32,32]."""
    if image.ndim != 4 or image.shape[1:] != (3,32,32):
        raise ValueError("Expected [B,3,32,32] RGB")
    return (image.amax(dim=1)-image.amin(dim=1)) > 0.45


def mask_iou(reference, predicted):
    if reference.shape != predicted.shape or reference.ndim != 3:
        raise ValueError("Masks must both have shape [B,32,32]")
    overlap=(reference & predicted).sum(dim=(1,2)).float()
    union=(reference | predicted).sum(dim=(1,2)).float()
    return torch.where(union>0,overlap/union,torch.ones_like(union))


def _summarize(items):
    if not items:
        raise ValueError("Empty metric group")
    return {
        "count":len(items),
        "correct_prompt_mean_mask_iou":round(sum(
            row["correct_prompt_mask_iou"] for row in items)/len(items),6),
        "wrong_prompt_mean_mask_iou":round(sum(
            row["wrong_prompt_mask_iou"] for row in items)/len(items),6),
        "mean_correct_prompt_iou_advantage":round(sum(
            row["correct_prompt_iou_advantage"] for row in items)/len(items),6),
        "correct_prompt_iou_win_rate":round(sum(
            row["correct_prompt_iou_win"] for row in items)/len(items),6),
        "correct_prompt_predicted_shape_rate":round(sum(
            row["correct_prompt_shape_match"] for row in items)/len(items),6),
        "wrong_prompt_predicted_shape_rate":round(sum(
            row["wrong_prompt_shape_match"] for row in items)/len(items),6),
        "correct_prompt_mean_foreground_area_ratio":round(sum(
            row["correct_prompt_area_ratio"] for row in items)/len(items),6),
    }


@torch.inference_mode()
def geometry_audit(model,seeds=(42,123),timesteps=(5,20,50,80),
                   guidance_scale=5.0):
    if model.config.dataset_version!="v2":
        raise ValueError("Requires RGB dataset v2 checkpoint")
    if (not seeds or not timesteps or len(set(seeds))!=len(seeds)
            or len(set(timesteps))!=len(timesteps)):
        raise ValueError("Use unique, nonempty seeds and timesteps")
    if any(t<0 or t>=model.config.diffusion_steps for t in timesteps):
        raise ValueError("Diffusion timestep out of range")
    if not math.isfinite(guidance_scale) or not 0<=guidance_scale<=20:
        raise ValueError("Invalid guidance scale")
    if guidance_scale!=1.0:
        require_guidance_training(model,guidance_scale)
    model.eval()
    device=next(model.parameters()).device
    examples=validation_items(seeds)
    images=torch.stack([e["image"] for e in examples]).to(device)
    good=torch.stack([e["correct_tokens"] for e in examples]).to(device)
    wrong=torch.stack([e["wrong_tokens"] for e in examples]).to(device)
    target_masks=foreground(images)
    target_areas=target_masks.sum(dim=(1,2)).float().clamp(min=1)
    _,_,a_bar=noise_schedule(model.config,device)
    generator=torch.Generator(device=device).manual_seed(87019+sum(seeds))
    rows=[]
    count=len(examples)
    for step in timesteps:
        eps=torch.randn(images.shape,device=device,dtype=images.dtype,
                        generator=generator)
        alpha=a_bar[step]
        xt=alpha.sqrt()*images+(1-alpha).sqrt()*eps
        time=torch.full((2*count,),step,dtype=torch.long,device=device)
        response=guided_noise(
            model,torch.cat((xt,xt)),time,torch.cat((good,wrong)),
            guidance_scale
        )
        x0=((torch.cat((xt,xt))-(1-alpha).sqrt()*response)
            /alpha.sqrt()).clamp(-1,1)
        recovered_good,recovered_wrong=x0.split(count)
        correct_mask=foreground(recovered_good)
        wrong_mask=foreground(recovered_wrong)
        good_iou=mask_iou(target_masks,correct_mask).cpu().tolist()
        bad_iou=mask_iou(target_masks,wrong_mask).cpu().tolist()
        good_areas=correct_mask.sum(dim=(1,2)).cpu().tolist()
        for i,example in enumerate(examples):
            correct_shape=analyze_image(
                recovered_good[i].cpu(),dataset_version="v2")["shape"]
            wrong_shape=analyze_image(
                recovered_wrong[i].cpu(),dataset_version="v2")["shape"]
            rows.append({
                "timestep":step,"seed":example["seed"],
                "shape":example["shape"],"color":example["color"],
                "language":example["language"],
                "target_foreground_pixels":int(target_areas[i]),
                "correct_prompt_foreground_pixels":int(good_areas[i]),
                "correct_prompt_area_ratio":round(
                    float(good_areas[i]/target_areas[i]),6),
                "correct_prompt_mask_iou":round(good_iou[i],6),
                "wrong_prompt_mask_iou":round(bad_iou[i],6),
                "correct_prompt_iou_advantage":round(
                    good_iou[i]-bad_iou[i],6),
                "correct_prompt_iou_win":bool(good_iou[i]>bad_iou[i]),
                "correct_prompt_predicted_shape":correct_shape,
                "wrong_prompt_predicted_shape":wrong_shape,
                "correct_prompt_shape_match":correct_shape==example["shape"],
                "wrong_prompt_shape_match":wrong_shape==example["shape"],
            })
    groups=defaultdict(list)
    for item in rows:
        groups[f'{item["shape"]}_{item["language"]}'].append(item)
    return {
        "schema_version":1,
        "task":"sora-v2-heldout-forward-denoising-geometry",
        "scope":"held-out synthetic image x_t recovery, not pure-noise image generation",
        "dataset_version":"v2",
        "architecture":model.config.architecture,
        "guidance_scale":guidance_scale,
        "clipping_policy":"fixed",
        "seeds":list(seeds),"timesteps":list(timesteps),
        "evaluation_count":len(rows),
        "by_shape_language":{
            key:_summarize(part) for key,part in sorted(groups.items())
        },
        "by_timestep":{
            str(step):_summarize([
                row for row in rows if row["timestep"]==step
            ]) for step in timesteps
        },
        "items":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",default=None)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,123])
    parser.add_argument("--timesteps",nargs="+",type=int,
                        default=[5,20,50,80])
    parser.add_argument("--guidance-scale",type=float,default=5.0)
    parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    parser.add_argument("--cpu-threads",type=int,default=2)
    args=parser.parse_args()
    if args.cpu_threads<1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    model=load_model(args.checkpoint,args.device)
    result=geometry_audit(
        model,seeds=tuple(args.seeds),timesteps=tuple(args.timesteps),
        guidance_scale=args.guidance_scale
    )
    result["checkpoint"]=args.checkpoint
    path=Path(args.output) if args.output else sora_run("geometry-audit")/"report.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    print(json.dumps({
        "report":str(path),
        "evaluation_count":result["evaluation_count"],
        "by_shape_language":result["by_shape_language"],
        "by_timestep":result["by_timestep"],
        "scope":result["scope"],
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

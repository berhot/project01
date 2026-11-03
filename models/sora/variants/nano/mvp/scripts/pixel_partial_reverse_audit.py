"""Read-only partial reverse-DDPM reconstruction from real held-out v2 RGB data.

Unlike text-to-image sampling from random Gaussian x_T, this audit starts
with x_t sampled from q(x_t | x0) for an actual held-out square/triangle.
It then runs EVERY deterministic posterior-mean reverse step t..0,
with either the correct or swapped shape word on IDENTICAL x_t.
No weights are trained; no procedural geometry is injected into outputs.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
import torch

from scripts.pixel_conditioning_audit import validation_items
from scripts.pixel_diffusion import load_model, require_guidance_training
from scripts.pixel_geometry_audit import foreground, mask_iou
from src.pixel_dataset_v2 import COLORS
from src.pixel_diffusion import guided_noise, noise_schedule
from src.pixel_metrics import analyze_image


def _group_seed(seed,color,language):
    """Share epsilon for square/triangle targets of same color/language."""
    return (34567+int(seed)*100003+list(COLORS).index(color)*1531
            +(0 if language=="en" else 257))


@torch.inference_mode()
def partial_reverse_audit(model,seeds=(42,),colors=("red",),
                          start_timesteps=(20,50,80,99),
                          guidance_scale=5.0):
    """Return metrics and learned RGB outputs; no checkpoint or file writes."""
    if model.config.dataset_version!="v2":
        raise ValueError("Partial reconstruction requires bilingual v2 model")
    if (not seeds or not colors or not start_timesteps
            or len(set(seeds))!=len(seeds)
            or len(set(colors))!=len(colors)
            or len(set(start_timesteps))!=len(start_timesteps)):
        raise ValueError("Inputs must be nonempty with unique entries")
    if any(color not in COLORS for color in colors):
        raise ValueError("Invalid color")
    total=model.config.diffusion_steps
    if any(t<0 or t>=total for t in start_timesteps):
        raise ValueError("Starting timestep not in trained schedule")
    if not math.isfinite(guidance_scale) or not 0<=guidance_scale<=20:
        raise ValueError("Invalid guidance scale")
    if guidance_scale!=1:
        require_guidance_training(model,guidance_scale)
    model.eval()
    device=next(model.parameters()).device
    data=[e for e in validation_items(seeds) if e["color"] in colors]
    n=len(data)
    if n==0:
        raise ValueError("No validation images selected")
    original=torch.stack([e["image"] for e in data]).to(device)
    correct=torch.stack([e["correct_tokens"] for e in data]).to(device)
    swapped=torch.stack([e["wrong_tokens"] for e in data]).to(device)
    reference_masks=foreground(original)
    beta,alpha,cum=noise_schedule(model.config,device)
    rows=[]
    correct_outputs={}
    for start in start_timesteps:
        # Keep the SAME random epsilon when only the target shape changes,
        # at each color/language/seed, and reuse that epsilon across starts.
        epsilon=torch.cat([
            torch.randn((1,3,32,32),device=device,dtype=original.dtype,
                        generator=torch.Generator(device=device).manual_seed(
                            _group_seed(e["seed"],e["color"],e["language"])
                        ))
            for e in data
        ],dim=0)
        a0=cum[start]
        noisy=a0.sqrt()*original+(1-a0).sqrt()*epsilon
        # Correct vs wrong prompt start with EXACTLY identical x_t.
        x=torch.cat((noisy.clone(),noisy.clone()),dim=0)
        all_tokens=torch.cat((correct,swapped),dim=0)
        for t in range(start,-1,-1):
            times=torch.full((2*n,),t,dtype=torch.long,device=device)
            eps=guided_noise(model,x,times,all_tokens,guidance_scale)
            at=cum[t]
            x0=((x-(1-at).sqrt()*eps)/at.sqrt()).clamp(-1,1)
            if t==0:
                x=x0
            else:
                prev=cum[t-1]
                mean_x0=beta[t]*prev.sqrt()/(1-at)
                mean_xt=alpha[t].sqrt()*(1-prev)/(1-at)
                x=mean_x0*x0+mean_xt*x
        recovered_good=x[:n]
        recovered_wrong=x[n:]
        masks_good=foreground(recovered_good)
        masks_wrong=foreground(recovered_wrong)
        iou_good=mask_iou(reference_masks,masks_good).cpu().tolist()
        iou_wrong=mask_iou(reference_masks,masks_wrong).cpu().tolist()
        for i,entry in enumerate(data):
            good_class=analyze_image(
                recovered_good[i].cpu(),dataset_version="v2")
            wrong_class=analyze_image(
                recovered_wrong[i].cpu(),dataset_version="v2")
            target_pixels=int(reference_masks[i].sum())
            pixel_count=int(masks_good[i].sum())
            key=f'{entry["seed"]}_{entry["color"]}_{entry["shape"]}_{entry["language"]}'
            correct_outputs[(start,key)]=recovered_good[i].cpu()
            rows.append({
                "start_timestep":start,
                "reverse_steps_executed":start+1,
                "seed":entry["seed"],"color":entry["color"],
                "shape":entry["shape"],"language":entry["language"],
                "input_type":"q(x_t|real_validation_x0)",
                "correct_caption_mask_iou":round(iou_good[i],6),
                "swapped_caption_mask_iou":round(iou_wrong[i],6),
                "correct_caption_iou_advantage":round(
                    iou_good[i]-iou_wrong[i],6),
                "correct_caption_shape_match":good_class["shape"]==entry["shape"],
                "swapped_caption_shape_match":wrong_class["shape"]==entry["shape"],
                "correct_caption_predicted_shape":good_class["shape"],
                "swapped_caption_predicted_shape":wrong_class["shape"],
                "correct_caption_foreground_area_ratio":round(
                    pixel_count/max(1,target_pixels),6),
                "final_rgb_prompt_difference_mae":round(float(
                    (recovered_good[i]-recovered_wrong[i]).abs().mean()),6),
                "output_key":key,
            })
    by_start={}
    for start in start_timesteps:
        sub=[r for r in rows if r["start_timestep"]==start]
        by_start[str(start)]={
            "cases":len(sub),
            "mean_target_mask_iou":round(sum(
                r["correct_caption_mask_iou"] for r in sub)/len(sub),6),
            "correct_shape_rate":round(sum(
                r["correct_caption_shape_match"] for r in sub)/len(sub),6),
            "mean_area_ratio":round(sum(
                r["correct_caption_foreground_area_ratio"] for r in sub)/len(sub),6),
            "triangle_shape_rate":round(sum(
                r["correct_caption_shape_match"] for r in sub
                if r["shape"]=="triangle"
            )/sum(r["shape"]=="triangle" for r in sub),6),
            "square_shape_rate":round(sum(
                r["correct_caption_shape_match"] for r in sub
                if r["shape"]=="square"
            )/sum(r["shape"]=="square" for r in sub),6),
            "correct_shape_word_iou_advantage":round(sum(
                r["correct_caption_iou_advantage"] for r in sub)/len(sub),6),
        }
    return {
        "schema_version":1,
        "task":"learned-rgb-partial-reverse-from-real-v2-validation-x0",
        "scope":"conditional reconstruction given a noisy real target, NOT text-to-image generation",
        "no_model_training":True,
        "architecture":model.config.architecture,
        "sampler":"posterior_mean",
        "x0_clipping":"fixed",
        "guidance_scale":guidance_scale,
        "start_timesteps":list(start_timesteps),
        "seeds":list(seeds),"colors":list(colors),
        "evaluation_count":len(rows),
        "by_start_timestep":by_start,
        "items":rows,
    }, correct_outputs


def save_audit(report,images,output_dir,scale=5):
    if not 1<=scale<=16:
        raise ValueError("scale must be in [1,16]")
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    # Every tile in the preview is a MODEL output from noisy real x_t.
    # No PIL ground-truth shapes are drawn or used in image outputs.
    keys=sorted({k for _,k in images})
    timesteps=report["start_timesteps"]
    cell=32*scale
    tile_height=cell+24
    grid=Image.new("RGB",(len(timesteps)*cell,len(keys)*tile_height),"white")
    draw=ImageDraw.Draw(grid)
    for row,key in enumerate(keys):
        for col,start in enumerate(timesteps):
            image=images[(start,key)]
            rgb=((image.permute(1,2,0).clamp(-1,1).numpy()+1)*127.5).round()
            tile=Image.fromarray(rgb.astype(np.uint8)).resize(
                (cell,cell),Image.Resampling.NEAREST
            )
            x,y=col*cell,row*tile_height
            grid.paste(tile,(x,y))
            draw.text((x+2,y+cell+3),f"{key}, from t={start}",fill="black")
    grid.save(out/"preview_grid.png")
    with (out/"report.json").open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    return out/"preview_grid.png"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42])
    parser.add_argument("--colors",nargs="+",default=["red"])
    parser.add_argument("--start-timesteps",type=int,nargs="+",
                        default=[20,50,80,99])
    parser.add_argument("--guidance-scale",type=float,default=5.0)
    parser.add_argument("--cpu-threads",type=int,default=2)
    args=parser.parse_args()
    if args.cpu_threads<1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    model=load_model(args.checkpoint,"cpu")
    report,images=partial_reverse_audit(
        model,seeds=tuple(args.seeds),colors=tuple(args.colors),
        start_timesteps=tuple(args.start_timesteps),
        guidance_scale=args.guidance_scale
    )
    report["checkpoint"]=args.checkpoint
    preview=save_audit(report,images,args.output_dir)
    print(json.dumps({
        "report":str(Path(args.output_dir)/"report.json"),
        "preview_grid":str(preview),
        "evaluation_count":report["evaluation_count"],
        "by_start_timestep":report["by_start_timestep"],
        "scope":report["scope"],
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

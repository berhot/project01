"""Read-only matched-noise trace of Sora Nano's real reverse RGB diffusion.

Square and triangle prompts begin with IDENTICAL Gaussian RGB noise.
At each reverse timestep, report a prompt-only denoiser response on
the same x_t, and how the two independently evolved samples diverge.
Captures actual learned x0 estimates at selected timesteps.

This is NOT a training script or a procedural image renderer. It never
alters source checkpoints. The measurement cannot certify realistic
image generation or general prompt comprehension.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.pixel_diffusion import load_model, require_guidance_training
from src.storage_paths import sora_run
from src.pixel_dataset_v2 import COLORS, caption_v2
from src.pixel_diffusion import guided_noise, noise_schedule, save_image, tokenize
from src.pixel_metrics import analyze_image


def _snapshots(total):
    indices={total-1,0}
    for fraction in (0.8,0.5,0.2,0.05):
        indices.add(max(0,min(total-1,int((total-1)*fraction))))
    return tuple(sorted(indices,reverse=True))


def _mask(image):
    saturation=image.amax(dim=1)-image.amin(dim=1)
    return saturation > 0.45


def _mask_iou(left,right):
    a,b=_mask(left),_mask(right)
    union=torch.logical_or(a,b).sum().item()
    return float(torch.logical_and(a,b).sum())/union if union else 1.0


@torch.inference_mode()
def trace_pair(model,color="red",language="en",seed=42,guidance_scale=5.0):
    """Return (JSON-safe metrics, learned x0 snapshots, final RGB outputs).

    The reverse rule mirrors 'posterior_mean' in src.pixel_diffusion.
    A third prompt evaluation uses the SAME square-path noisy x_t with
    the triangle tokens, to isolate shape-word sensitivity from diverged
    sample trajectories.
    """
    if model.config.dataset_version!="v2":
        raise ValueError("Reverse trajectory trace requires dataset v2")
    if color not in COLORS or language not in ("en","ar"):
        raise ValueError("Unsupported color or language")
    if not isinstance(seed,int):
        raise ValueError("seed must be an integer")
    if not math.isfinite(guidance_scale) or not 0<=guidance_scale<=20:
        raise ValueError("Invalid guidance_scale")
    if guidance_scale>1:
        require_guidance_training(model,guidance_scale)
    model.eval()
    device=next(model.parameters()).device
    total=model.config.diffusion_steps
    prompts={shape:caption_v2(color,shape,language,0)
             for shape in ("square","triangle")}
    word_ids={
        shape:tokenize(prompt,dataset_version="v2").to(device)
        for shape,prompt in prompts.items()
    }
    generator=torch.Generator(device=device).manual_seed(seed)
    initial=torch.randn((1,3,32,32),device=device,generator=generator)
    square=initial.clone()
    triangle=initial.clone()
    beta,alpha,cum=noise_schedule(model.config,device)
    take=set(_snapshots(total))
    samples={}
    rows=[]
    for t in range(total-1,-1,-1):
        # Three conditions in ONE forward pass. Model sees:
        # (1) square on square-path x_t, (2) triangle on triangle-path x_t,
        # (3) triangle on square-path x_t (counterfactual, same input).
        noisy=torch.cat((square,triangle,square),dim=0)
        tokens=torch.stack((word_ids["square"],word_ids["triangle"],
                            word_ids["triangle"]),dim=0)
        times=torch.full((3,),t,device=device,dtype=torch.long)
        eps=guided_noise(model,noisy,times,tokens,guidance_scale)
        e_square,e_triangle,e_triangle_same=eps.chunk(3,dim=0)
        a=cum[t]
        x0_square_raw=(square-(1-a).sqrt()*e_square)/a.sqrt()
        x0_triangle_raw=(triangle-(1-a).sqrt()*e_triangle)/a.sqrt()
        x0_square=x0_square_raw.clamp(-1,1)
        x0_triangle=x0_triangle_raw.clamp(-1,1)
        if t in take:
            square_cpu=x0_square[0].cpu()
            triangle_cpu=x0_triangle[0].cpu()
            samples[t]=(square_cpu,triangle_cpu)
            rows.append({
                "timestep":t,
                "same_noisy_pixels_prompt_eps_rms":round(float(
                    (e_square-e_triangle_same).square().mean().sqrt()),6),
                "evolved_noisy_pixel_mae":round(float(
                    (square-triangle).abs().mean()),6),
                "predicted_clean_pixel_mae":round(float(
                    (x0_square-x0_triangle).abs().mean()),6),
                "square_triangle_predicted_mask_iou":round(
                    _mask_iou(x0_square,x0_triangle),6),
                "square_predicted_foreground_pixels":int(_mask(x0_square).sum()),
                "triangle_predicted_foreground_pixels":int(_mask(x0_triangle).sum()),
                "square_predicted_class":analyze_image(
                    square_cpu,dataset_version="v2")["shape"],
                "triangle_predicted_class":analyze_image(
                    triangle_cpu,dataset_version="v2")["shape"],
                "square_x0_clipped_fraction":round(float(
                    (x0_square_raw.abs()>1).float().mean()),6),
                "triangle_x0_clipped_fraction":round(float(
                    (x0_triangle_raw.abs()>1).float().mean()),6),
            })
        if t==0:
            square=x0_square
            triangle=x0_triangle
        else:
            prev=cum[t-1]
            mean_x0=beta[t]*prev.sqrt()/(1-a)
            mean_xt=alpha[t].sqrt()*(1-prev)/(1-a)
            square=mean_x0*x0_square+mean_xt*square
            triangle=mean_x0*x0_triangle+mean_xt*triangle
    final={"square":square[0].clamp(-1,1).cpu(),
           "triangle":triangle[0].clamp(-1,1).cpu()}
    report={
        "schema_version":1,
        "scope":"read-only learned RGB posterior-mean reverse-process trajectory; not visual quality validation",
        "architecture":model.config.architecture,
        "dataset_version":"v2",
        "sampler":"posterior_mean",
        "guidance_scale":guidance_scale,
        "color":color,"language":language,"seed":seed,
        "initial_noise_shared_between_prompts":True,
        "all_reverse_timesteps_executed":total,
        "prompts":prompts,
        "snapshots":rows,
        "final_predictions":{
            shape:analyze_image(image,dataset_version="v2")
            for shape,image in final.items()
        },
    }
    return report,samples,final


def export_trace(report,snapshots,final,output_dir,scale=5):
    if not 1<=scale<=16:
        raise ValueError("scale must be between 1 and 16")
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=False)
    for shape in ("square","triangle"):
        save_image(final[shape],root/f"{shape}_final.png")
    ticks=sorted(snapshots,reverse=True)
    cell=32*scale
    grid=Image.new("RGB",(len(ticks)*cell,2*(cell+26)), "white")
    pen=ImageDraw.Draw(grid)
    for i,t in enumerate(ticks):
        for j,shape in enumerate(("square","triangle")):
            image=snapshots[t][j]
            rgb=((image.permute(1,2,0).clamp(-1,1).numpy()+1)*127.5).round().astype("uint8")
            tile=Image.fromarray(rgb).resize((cell,cell),Image.Resampling.NEAREST)
            x,y=i*cell,j*(cell+26)
            grid.paste(tile,(x,y))
            pen.text((x+4,y+cell+4),f"{shape}, t={t}",fill="black")
    grid.save(root/"reverse_trace.png")
    out=dict(report)
    out["output_dir"]=str(root)
    out["preview_grid"]=str(root/"reverse_trace.png")
    out["final_images"]={shape:str(root/f"{shape}_final.png")
                         for shape in final}
    with (root/"report.json").open("x",encoding="utf-8") as fd:
        json.dump(out,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output-dir",default=None)
    p.add_argument("--color",choices=tuple(COLORS),default="red")
    p.add_argument("--language",choices=("ar","en"),default="en")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--guidance-scale",type=float,default=5.0)
    p.add_argument("--cpu-threads",type=int,default=2)
    args=p.parse_args()
    if args.cpu_threads<1:
        p.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    model=load_model(args.checkpoint,"cpu")
    report,snapshots,final=trace_pair(
        model,args.color,args.language,args.seed,args.guidance_scale
    )
    output_dir=args.output_dir or str(sora_run("reverse-trace"))
    result=export_trace(report,snapshots,final,output_dir)
    result["checkpoint"]=args.checkpoint
    print(json.dumps({
        "report":str(Path(output_dir)/"report.json"),
        "preview_grid":result["preview_grid"],
        "all_reverse_timesteps_executed":result["all_reverse_timesteps_executed"],
        "final_predictions":result["final_predictions"],
        "snapshots":result["snapshots"],
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

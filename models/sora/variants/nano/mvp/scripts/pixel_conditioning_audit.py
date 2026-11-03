"""Read-only ordinary-diffusion test of triangle vs square text conditioning.

For each held-out RGB training target, sample ordinary forward q(x_t | x_0)
once, then ask the SAME checkpoint to predict the SAME noise with the correct
and the incorrect shape caption. The only difference is the shape word.
No gradients, no model training, no new checkpoints, no procedural inference.

This is a narrow toy-domain diagnostic; lower noise MSE under the correct
caption is useful evidence but cannot certify visually accurate generations.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.pixel_diffusion import load_model
from src.pixel_dataset_v2 import COLORS, caption_v2, synthetic_example_v2, tokenize_v2
from src.pixel_diffusion import noise_schedule


def validation_items(seeds=(42,123)):
    """Deterministic 4 colors × 2 shapes × 2 languages × N seeds."""
    items=[]
    for seed in seeds:
        for color_i,color in enumerate(COLORS):
            for shape in ("square","triangle"):
                for language in ("en","ar"):
                    # Same seed for both shapes; separate color layout seeds.
                    image_seed=seed+color_i*47
                    image,tokens,prompt,concept=synthetic_example_v2(
                        image_seed,color=color,shape=shape,language=language,
                        split="validation",variant=0,
                    )
                    other="triangle" if shape=="square" else "square"
                    wrong_tokens=tokenize_v2(caption_v2(color,other,language,0))
                    mask=(image.amax(dim=0)-image.amin(dim=0)>0.45)
                    ys,xs=torch.where(mask)
                    if len(xs)==0:
                        raise ValueError("Validation sample unexpectedly lacks foreground")
                    box_area=(int(xs.max()-xs.min())+1)*(int(ys.max()-ys.min())+1)
                    fill=float(mask.sum())/box_area
                    items.append({
                        "seed":seed,"color":color,"shape":shape,"language":language,
                        "image":image,"correct_tokens":tokens,
                        "wrong_tokens":wrong_tokens,
                        "foreground_fill":round(fill,5),
                    })
    return items


@torch.no_grad()
def conditioning_audit(model,seeds=(42,123),timesteps=(5,20,50,80)):
    if model.config.dataset_version!="v2":
        raise ValueError("Only the v2 bilingual model is supported")
    if (not seeds or not timesteps or len(set(seeds))!=len(seeds)
            or len(set(timesteps))!=len(timesteps)):
        raise ValueError("Use nonempty unique seeds and timesteps")
    if any(t<0 or t>=model.config.diffusion_steps for t in timesteps):
        raise ValueError("Timesteps must belong to the model's schedule")
    model.eval()
    device=next(model.parameters()).device
    items=validation_items(seeds)
    images=torch.stack([x["image"] for x in items]).to(device)
    correct=torch.stack([x["correct_tokens"] for x in items]).to(device)
    wrong=torch.stack([x["wrong_tokens"] for x in items]).to(device)
    _,_,alpha_bar=noise_schedule(model.config,device)
    g=torch.Generator(device=device).manual_seed(8675309+sum(seeds))
    results=[]
    n=len(items)
    for t in timesteps:
        noise=torch.randn(images.shape,device=device,dtype=images.dtype,generator=g)
        a=alpha_bar[t]
        xt=a.sqrt()*images+(1-a).sqrt()*noise
        times=torch.full((2*n,),t,device=device,dtype=torch.long)
        prediction=model(
            torch.cat([xt,xt]),times,torch.cat([correct,wrong])
        )
        expected=torch.cat([noise,noise])
        loss=(prediction-expected).square().mean(dim=(1,2,3)).cpu()
        pred_correct,pred_wrong=prediction[:n],prediction[n:]
        sensitivity=(pred_correct-pred_wrong).square().mean(dim=(1,2,3)).sqrt().cpu()
        for i,item in enumerate(items):
            mse_true=float(loss[i])
            mse_false=float(loss[n+i])
            results.append({
                "timestep":t,"seed":item["seed"],
                "color":item["color"],"shape":item["shape"],
                "language":item["language"],
                "foreground_fill":item["foreground_fill"],
                "correct_prompt_noise_mse":round(mse_true,6),
                "swapped_prompt_noise_mse":round(mse_false,6),
                "correct_prompt_advantage":round(mse_false-mse_true,6),
                "correct_prompt_wins":bool(mse_true<mse_false),
                "prompt_response_rms":round(float(sensitivity[i]),6),
            })
    groups=defaultdict(list)
    for row in results:
        groups[f'{row["shape"]}_{row["language"]}'].append(row)
    def summarize(rows):
        return {
            "count":len(rows),
            "mean_correct_prompt_advantage":round(sum(
                r["correct_prompt_advantage"] for r in rows
            )/len(rows),6),
            "correct_prompt_win_rate":round(sum(
                r["correct_prompt_wins"] for r in rows
            )/len(rows),4),
            "mean_prompt_response_rms":round(sum(
                r["prompt_response_rms"] for r in rows
            )/len(rows),6),
        }
    return {
        "schema_version":1,
        "kind":"ordinary-forward-noise-shape-word-sensitivity",
        "architecture":model.config.architecture,
        "dataset_version":"v2",
        "sample_count":n,
        "evaluation_count":len(results),
        "seeds":list(seeds),
        "timesteps":list(timesteps),
        "by_shape_language":{k:summarize(v) for k,v in sorted(groups.items())},
        "by_timestep":{str(t):summarize([
            r for r in results if r["timestep"]==t
        ]) for t in timesteps},
        "validation_foreground_fill":{
            shape:round(sum(x["foreground_fill"] for x in items
                            if x["shape"]==shape)/sum(x["shape"]==shape
                                                     for x in items),4)
            for shape in ("square","triangle")
        },
        "items":results,
        "scope":"held-out synthetic ordinary DDPM q(x_t|x0); read-only, not generated-image quality",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,123])
    p.add_argument("--timesteps",type=int,nargs="+",default=[5,20,50,80])
    p.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    model=load_model(args.checkpoint,args.device)
    report=conditioning_audit(model,tuple(args.seeds),tuple(args.timesteps))
    report["checkpoint"]=args.checkpoint
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    print(json.dumps({
        "report":str(out),"evaluation_count":report["evaluation_count"],
        "shape_language":report["by_shape_language"],
        "by_timestep":report["by_timestep"],
        "foreground_fill":report["validation_foreground_fill"],
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

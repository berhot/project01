"""Sora Nano shape-first: train, evaluate, and sample a two-stage learned model.

Train from scratch (NOT fine-tuning the old 32x32 RGB DDPM).
The RGB original baseline stays immutable. Both new checkpoints live in a
new unique central BERHOT artifact run; NO release or cloud sync occurs.

Run from models/sora/variants/nano/mvp:
  python3 -m scripts.shape_first train --mask-steps 300 --rgb-steps 200
  python3 -m scripts.shape_first evaluate --run-dir /path/to/new/run
  python3 -m scripts.shape_first sample --run-dir /path/to/new/run \
    --prompt 'مثلث أصفر'
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from src.pixel_dataset_v2 import (
    COLORS,SHAPES,caption_v2,balanced_training_batch,synthetic_example_v2,
    tokenize_v2,
)
from src.pixel_diffusion import vocabulary_for,save_image
from src.pixel_metrics import analyze_image,summarize
from src.pixel_shape_quality import score_model_sample
from src.shape_first import (
    ShapeFirstConfig,MaskVAE,RGBAppearance,mask_from_rgb,
    mask_training_objective,appearance_training_objective,generate_pipeline,
)
from src.storage_paths import sora_run

MASK_ARCHITECTURE="sora-shape-mask-cvae-v1"
RGB_ARCHITECTURE="sora-mask-conditioned-rgb-v1"


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_layout_training_pairs(step,count,seed,device="cpu"):
    """Supervised square->circle/square/triangle paired examples for TRAINING.

    Different target shapes use matching seed, color, background, position,
    scale, and EN/AR caption. No such synthetic targets are used in inference.
    """
    if step<1 or count<1:
        raise ValueError("Invalid same-layout pair request")
    sources,targets,tokens=[],[],[]
    colors=tuple(COLORS)
    for i in range(count):
        color=colors[(step+i)%len(colors)]
        seed_for_pair=seed+step*7919+i*101
        square,_,_,_=synthetic_example_v2(
            seed_for_pair,color=color,shape="square",language="en",
            split="train",variant=0)
        source_mask=mask_from_rgb(square.unsqueeze(0))[0]
        for shape in SHAPES:
            truth,_,_,_=synthetic_example_v2(
                seed_for_pair,color=color,shape=shape,language="en",
                split="train",variant=0)
            target_mask=mask_from_rgb(truth.unsqueeze(0))[0]
            for language in ("en","ar"):
                sources.append(source_mask)
                targets.append(target_mask)
                tokens.append(tokenize_v2(caption_v2(color,shape,language,0)))
    return (torch.stack(sources).to(device),
            torch.stack(targets).to(device),
            torch.stack(tokens).to(device))


def load_pipeline(run_dir,device="cpu"):
    root=Path(run_dir)
    manifest=json.loads((root/"run.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version")!=1
            or manifest.get("family")!="sora"
            or manifest.get("model_family")!="shape-first"
            or manifest.get("dataset_version")!="v2"
            or manifest.get("quality_status")!="experimental"):
        raise ValueError("Not an experimental Sora shape-first run")
    config=ShapeFirstConfig(**manifest["config"])
    restored=[]
    for name,kind,cls in (
        ("mask.pt",MASK_ARCHITECTURE,MaskVAE),
        ("appearance.pt",RGB_ARCHITECTURE,RGBAppearance),
    ):
        path=root/name
        if not path.is_file() or sha256(path)!=manifest["sha256"][name]:
            raise ValueError(f"Corrupt/missing shape-first weight file: {name}")
        record=torch.load(path,map_location="cpu",weights_only=True)
        if (record.get("schema_version")!=1
                or record.get("architecture")!=kind
                or record.get("family")!="sora"
                or record.get("dataset_version")!="v2"
                or record.get("config")!=asdict(config)
                or record.get("vocabulary")!=list(vocabulary_for("v2"))):
            raise ValueError(f"Incorrect shape-first checkpoint architecture: {name}")
        model=cls(config)
        model.load_state_dict(record["model"],strict=True)
        model.to(device).eval()
        restored.append(model)
    return restored[0],restored[1],manifest


def _write_model(path,model,architecture,steps,seed):
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "schema_version":1,"family":"sora",
        "architecture":architecture,"dataset_version":"v2",
        "config":asdict(model.config),
        "vocabulary":list(vocabulary_for("v2")),
        "training_steps":steps,"seed":seed,
        "training_data":"synthetic 32x32 RGB shapes, 4 colors, 3 shapes, EN/AR",
        "quality_status":"experimental; requires generated-image review",
        "model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
    },path)


def train(args):
    """New distinct two-stage training, no source RGB DDPM load or overwrite."""
    if (args.mask_steps<1 or args.rgb_steps<1
            or args.batch_size<12 or args.batch_size%12
            or args.paired_layout_count<1 or args.cpu_threads<1
            or args.log_every<1 or args.learning_rate<=0
            or not math.isfinite(args.learning_rate)
            or not 0<=args.beta<=1
            or not 0<=args.paired_weight<=2):
        raise ValueError("Invalid shape-first training arguments")
    if args.device=="cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config=ShapeFirstConfig(
        mask_width=args.mask_width,rgb_width=args.rgb_width,
        text_dim=args.text_dim,latent_dim=args.latent_dim)
    mask_model=MaskVAE(config).to(args.device)
    rgb_model=RGBAppearance(config).to(args.device)
    directory=sora_run("shape-first",args.output_root)
    directory.mkdir(parents=True,exist_ok=False)
    try:
        mask_optimizer=torch.optim.AdamW(mask_model.parameters(),lr=args.learning_rate)
        for step in range(1,args.mask_steps+1):
            images,tokens=balanced_training_batch(step,args.batch_size,args.seed)
            images,tokens=images.to(args.device),tokens.to(args.device)
            pairs=same_layout_training_pairs(
                step,args.paired_layout_count,args.seed,args.device
            ) if args.paired_weight>0 else None
            mask_model.train()
            mask_optimizer.zero_grad(set_to_none=True)
            measures=mask_training_objective(
                mask_model,images,tokens,beta=args.beta,
                contrast_weight=args.paired_weight,paired=pairs)
            loss=measures["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite learned shape-mask objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mask_model.parameters(),1.0)
            mask_optimizer.step()
            if step==1 or step%args.log_every==0 or step==args.mask_steps:
                print(json.dumps({
                    "stage":"learned-mask","step":step,
                    "mask_loss":round(float(loss.detach()),6),
                    "mask_bce":round(float(measures["mask_bce"]),6),
                    "mask_dice":round(float(measures["mask_dice_loss"]),6),
                    "kl":round(float(measures["latent_kl"]),6),
                    "paired_shape_loss":round(float(
                        measures["counterfactual_mask_loss"]),6)
                }),flush=True)

        _write_model(directory/"mask.pt",mask_model,MASK_ARCHITECTURE,
                     args.mask_steps,args.seed)
        rgb_optimizer=torch.optim.AdamW(rgb_model.parameters(),lr=args.learning_rate)
        for step in range(1,args.rgb_steps+1):
            images,tokens=balanced_training_batch(
                step,args.batch_size,args.seed+50_000)
            images,tokens=images.to(args.device),tokens.to(args.device)
            rgb_model.train()
            rgb_optimizer.zero_grad(set_to_none=True)
            measures=appearance_training_objective(rgb_model,images,tokens)
            loss=measures["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite learned RGB appearance objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rgb_model.parameters(),1.0)
            rgb_optimizer.step()
            if step==1 or step%args.log_every==0 or step==args.rgb_steps:
                print(json.dumps({
                    "stage":"learned-appearance","step":step,
                    "rgb_loss":round(float(loss.detach()),6),
                    "weighted_l1":round(float(measures["rgb_l1"]),6),
                    "weighted_mse":round(float(measures["rgb_mse"]),6)
                }),flush=True)
        _write_model(directory/"appearance.pt",rgb_model,
                     RGB_ARCHITECTURE,args.rgb_steps,args.seed)
        manifest={
            "schema_version":1,"family":"sora","model_family":"shape-first",
            "dataset_version":"v2",
            "mask_architecture":MASK_ARCHITECTURE,
            "appearance_architecture":RGB_ARCHITECTURE,
            "config":asdict(config),
            "mask_steps":args.mask_steps,
            "rgb_steps":args.rgb_steps,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "beta":args.beta,
            "paired_weight":args.paired_weight,
            "paired_layout_count":args.paired_layout_count,
            "seed":args.seed,
            "sha256":{
                "mask.pt":sha256(directory/"mask.pt"),
                "appearance.pt":sha256(directory/"appearance.pt"),
            },
            "quality_status":"experimental",
            "training_distribution":"on-demand synthetic v2 RGB images/masks",
            "generator_inference":"Gaussian latent -> learned mask CVAE decoder -> trained RGB neural appearance",
            "procedural_shapes_at_inference":False,
            "release_status":"not-approved",
            "baseline_rgb_ddpm_unchanged":True,
        }
        with (directory/"run.json").open("x",encoding="utf-8") as file:
            json.dump(manifest,file,indent=2,ensure_ascii=False)
            file.write("\n")
    except Exception:
        # Keep partial run for investigation. Do not silently delete weights.
        print(json.dumps({"incomplete_experimental_run":str(directory)}),flush=True)
        raise
    print(json.dumps({
        "run_dir":str(directory),
        "mask_checkpoint":str(directory/"mask.pt"),
        "appearance_checkpoint":str(directory/"appearance.pt"),
        "status":"experimental, NOT a validated release",
    },ensure_ascii=False,indent=2),flush=True)
    return directory


def sample(run_dir,prompt,seed=42,output=None,device="cpu",latent_scale=1.0):
    mask_model,rgb_model,_=load_pipeline(run_dir,device=device)
    rgb,hard,probability=generate_pipeline(
        mask_model,rgb_model,prompt,seed=seed,
        latent_scale=latent_scale,device=device)
    destination=Path(output) if output is not None else sora_run("shape-first-sample")/"image.png"
    save_image(rgb,destination)
    # This is the LEARNED mask, not a PIL-generated shape.
    mask_dest=destination.parent/(destination.stem+"_learned_mask.png")
    with mask_dest.open("xb") as file:
        Image.fromarray((hard[0].numpy()*255).astype("uint8")).save(file,format="PNG")
    return {"image":str(destination),"learned_mask":str(mask_dest),
            "prompt":prompt,"seed":seed}


def evaluate(run_dir,output_dir=None,seed=42,device="cpu",latent_scale=1.0):
    mask_model,rgb_model,manifest=load_pipeline(run_dir,device=device)
    root=Path(output_dir) if output_dir is not None else sora_run("shape-first-eval")
    root.mkdir(parents=True,exist_ok=False)
    masks_dir=root/"learned_masks"
    masks_dir.mkdir(exist_ok=False)
    images=[]
    items=[]
    rows=[]
    old_predictions=[]
    sheet=Image.new("RGB",(4*32*7,6*(32*7+26)),"white")
    mask_sheet=Image.new("RGB",(4*32*7,6*(32*7+26)),"white")
    pen=ImageDraw.Draw(sheet)
    mask_pen=ImageDraw.Draw(mask_sheet)
    for index,shape in enumerate(SHAPES):
        for color_index,color in enumerate(COLORS):
            col=color_index
            class_index=index*len(COLORS)+color_index
            for language_index,language in enumerate(("en","ar")):
                prompt=caption_v2(color,shape,language,0)
                # Shared layout-noise seed for matched EN/AR and shape prompts.
                shared_seed=seed+color_index
                rgb,hard,probability=generate_pipeline(
                    mask_model,rgb_model,prompt,seed=shared_seed,
                    latent_scale=latent_scale,device=device)
                filename=f"{color}_{shape}_{language}.png"
                save_image(rgb,root/filename)
                # Debug outputs come from the trained mask decoder, never PIL
                # shape templates. These are saved separately from RGB gallery.
                learned_mask=Image.fromarray(
                    (hard[0].numpy()*255).astype("uint8"),mode="L"
                )
                learned_mask.save(masks_dir/filename)
                quality=score_model_sample(rgb,color,shape)
                predictions=analyze_image(rgb,dataset_version="v2")
                old_predictions.append({
                    "expected_color":color,"expected_shape":shape,
                    "predicted_color":predictions["color"],
                    "predicted_shape":predictions["shape"],
                    "seed":shared_seed,"language":language,
                })
                row=index*2+language_index
                x,y=col*32*7,row*(32*7+26)
                with Image.open(root/filename) as img:
                    sheet.paste(img.resize((32*7,32*7),Image.Resampling.NEAREST),(x,y))
                pen.text((x+5,y+32*7+5),f"{color} {shape} {language}",fill="black")
                mask_sheet.paste(learned_mask.convert("RGB").resize(
                    (32*7,32*7),Image.Resampling.NEAREST),(x,y))
                mask_pen.text((x+5,y+32*7+5),
                              f"{color} {shape} {language}",fill="black")
                images.append(rgb)
                items.append({
                    "prompt":prompt,"image":filename,"language":language,
                    "seed":shared_seed,"expected_shape":shape,
                    "expected_color":color,
                    "learned_mask":f"learned_masks/{filename}",
                    "strict_geometry":quality,
                })
                rows.append(quality)
    sheet.save(root/"preview_grid.png")
    mask_sheet.save(root/"learned_masks_preview_grid.png")
    strict={
        shape:{
            "images":sum(item["expected_shape"]==shape for item in items),
            "geometry_pass_count":sum(item["strict_geometry"]["qualified_geometry"]
                for item in items if item["expected_shape"]==shape),
            "strict_prompt_pass_count":sum(item["strict_geometry"]["strict_prompt_pass"]
                for item in items if item["expected_shape"]==shape),
        } for shape in SHAPES
    }
    result={
        "schema_version":1,"family":"sora","architecture":"shape-first-mask-vae-plus-rgb-v1",
        "dataset_version":"v2","source_run_dir":str(run_dir),
        "mask_checkpoint_sha256":manifest["sha256"]["mask.pt"],
        "appearance_checkpoint_sha256":manifest["sha256"]["appearance.pt"],
        "sampler":"mask-cvae-gaussian-prior",
        "guidance_scale":None,
        "latent_scale":latent_scale,
        "x0_clipping":None,
        "shared_noise_across_concepts":False,
        "seed_policy":"same Gaussian latent seed per color, both languages and shapes",
        "images":len(items),
        "metrics":summarize(old_predictions,dataset_version="v2"),
        "strict_by_shape":strict,
        "strict_prompt_accuracy":round(sum(
            item["strict_geometry"]["strict_prompt_pass"] for item in items)/24,4),
        "quality_status":"experimental, human review required",
        "items":items,
    }
    with (root/"report.json").open("x",encoding="utf-8") as fd:
        json.dump(result,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    print(json.dumps({
        "preview_grid":str(root/"preview_grid.png"),
        "learned_masks_preview_grid":str(root/"learned_masks_preview_grid.png"),
        "report":str(root/"report.json"),
        "images":24,
        "strict_by_shape":strict,
        "strict_prompt_accuracy":result["strict_prompt_accuracy"],
        "metrics":result["metrics"],
        "release_status":"not-approved",
    },indent=2,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    train_parser=sub.add_parser("train")
    train_parser.add_argument("--mask-steps",type=int,default=500)
    train_parser.add_argument("--rgb-steps",type=int,default=350)
    train_parser.add_argument("--batch-size",type=int,default=12)
    train_parser.add_argument("--mask-width",type=int,default=32)
    train_parser.add_argument("--rgb-width",type=int,default=32)
    train_parser.add_argument("--text-dim",type=int,default=48)
    train_parser.add_argument("--latent-dim",type=int,default=12)
    train_parser.add_argument("--learning-rate",type=float,default=0.0007)
    train_parser.add_argument("--beta",type=float,default=0.01)
    train_parser.add_argument("--paired-weight",type=float,default=0.5)
    train_parser.add_argument("--paired-layout-count",type=int,default=2)
    train_parser.add_argument("--seed",type=int,default=42)
    train_parser.add_argument("--cpu-threads",type=int,default=2)
    train_parser.add_argument("--log-every",type=int,default=50)
    train_parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    train_parser.add_argument("--output-root",default=None,
                              help="Optional override for central BERHOT artifacts")
    eval_parser=sub.add_parser("evaluate")
    eval_parser.add_argument("--run-dir",required=True)
    eval_parser.add_argument("--output-dir",default=None)
    eval_parser.add_argument("--seed",type=int,default=42)
    eval_parser.add_argument("--latent-scale",type=float,default=1.0)
    eval_parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    sample_parser=sub.add_parser("sample")
    sample_parser.add_argument("--run-dir",required=True)
    sample_parser.add_argument("--prompt",required=True)
    sample_parser.add_argument("--seed",type=int,default=42)
    sample_parser.add_argument("--latent-scale",type=float,default=1.0)
    sample_parser.add_argument("--output",default=None)
    sample_parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    args=parser.parse_args()
    if args.action=="train":
        train(args)
    elif args.action=="evaluate":
        evaluate(args.run_dir,args.output_dir,args.seed,args.device,args.latent_scale)
    else:
        print(json.dumps(sample(args.run_dir,args.prompt,args.seed,args.output,
                                args.device,args.latent_scale),
                         indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

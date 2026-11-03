"""Train/evaluate/sample Sora Shape Field V2, separately from failed V1.

The mask is learned by a PyTorch implicit neural field conditioned on shape
words + Gaussian-sampled layout. RGB is learned by a neural foreground/
background color model conditioned on color words. No procedural shape
template is used at inference. The dataset is only 32x32 toy shapes.

All runs written centrally under BERHOT/storage/artifacts/sora/nano/mvp/.
This tool cannot write approved releases, mutate V1 or RGB-DDPM weights.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image,ImageDraw
import torch

from src.pixel_dataset_v2 import COLORS,SHAPES,balanced_training_batch,caption_v2
from src.pixel_diffusion import save_image,vocabulary_for
from src.pixel_metrics import summarize,analyze_image
from src.pixel_shape_quality import score_model_sample
from scripts.shape_first_diagnose import score_binary_mask
from src.shape_field_v2 import (
    ShapeFieldConfig,NeuralMaskField,NeuralColorizer,
    geometry_objective,color_objective,generate
)
from src.storage_paths import sora_run
from src.shape_field_v21 import (
    matched_layout_shape_batch,shape_word_separation_objective,
    circle_teacher_preservation
)

MASK_KIND="sora-implicit-shape-field-v2"
COLOR_KIND="sora-neural-colorizer-v2"


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(1048576),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_weight(path,network,kind,steps,seed):
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "schema_version":1,
        "family":"sora",
        "architecture":kind,
        "dataset_version":"v2",
        "config":asdict(network.config),
        "vocabulary":list(vocabulary_for("v2")),
        "steps":steps,"seed":seed,
        "model":{key:value.detach().cpu()
                 for key,value in network.state_dict().items()},
        "quality_status":"experimental",
    },path)


def load_run(run_dir,device="cpu"):
    root=Path(run_dir)
    manifest=json.loads((root/"run.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version")!=1
            or manifest.get("family")!="sora"
            or manifest.get("model_family")!="shape-field-v2"
            or manifest.get("dataset_version")!="v2"
            or manifest.get("release_status")!="not-approved"):
        raise ValueError("Not a compatible experimental shape-field v2 run")
    cfg=ShapeFieldConfig(**manifest["config"])
    models=[]
    for filename,kind,model_type in (
        ("mask_field.pt",MASK_KIND,NeuralMaskField),
        ("colorizer.pt",COLOR_KIND,NeuralColorizer),
    ):
        file=root/filename
        if not file.is_file() or checksum(file)!=manifest["sha256"][filename]:
            raise ValueError(f"Missing or corrupt weight file: {filename}")
        state=torch.load(file,map_location="cpu",weights_only=True)
        if (state.get("schema_version")!=1
                or state.get("architecture")!=kind
                or state.get("family")!="sora"
                or state.get("dataset_version")!="v2"
                or state.get("config")!=asdict(cfg)
                or state.get("vocabulary")!=list(vocabulary_for("v2"))):
            raise ValueError(f"Incompatible architecture or vocab in {filename}")
        network=model_type(cfg).to(device)
        network.load_state_dict(state["model"],strict=True)
        network.eval()
        models.append(network)
    return models[0],models[1],manifest


def train(args):
    init_v2=getattr(args,"init_from_v2",None)
    preserve_circle_weight=getattr(args,"circle_preserve_weight",0.2)
    if (args.mask_steps<1 or args.color_steps<0 or
            args.batch_size<12 or args.batch_size%12 or
            args.log_every<1 or args.cpu_threads<1 or
            not math.isfinite(args.learning_rate) or args.learning_rate<=0):
        raise ValueError("Invalid shape-field training arguments")
    v21=getattr(args,"shape_v21",False)
    pair_weight=getattr(args,"pair_weight",1.0)
    rank_weight=getattr(args,"rank_weight",0.1)
    rank_margin=getattr(args,"rank_margin",0.03)
    pair_examples=getattr(args,"paired_layout_count",4)
    if init_v2 and not v21:
        raise ValueError("V2 warm-start requires explicit --shape-v21")
    if args.color_steps==0 and not (v21 and init_v2):
        raise ValueError("Skipping color training requires V2.1 warm-start")
    if not math.isfinite(preserve_circle_weight) or not 0<=preserve_circle_weight<=2:
        raise ValueError("Invalid circle preservation weight")
    if preserve_circle_weight>0 and v21 and not init_v2:
        raise ValueError("V2.1 frozen circle preservation requires --init-from-v2")
    if v21 and (not all(math.isfinite(x) for x in (
            pair_weight,rank_weight,rank_margin))
            or not 0<pair_weight<=2 or not 0<=rank_weight<=1
            or not 0<=rank_margin<=1 or not 1<=pair_examples<=32):
        raise ValueError("Invalid separate V2.1 paired-shape training recipe")
    cfg=ShapeFieldConfig(hidden=args.hidden,word_dim=args.word_dim)
    if args.device=="cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    shape=NeuralMaskField(cfg).to(args.device)
    color=NeuralColorizer(cfg).to(args.device)
    teacher=None
    warm_start=None
    if init_v2:
        teacher,source_color,source_manifest=load_run(init_v2,args.device)
        if (source_manifest.get("training_recipe","original-v2")!="original-v2"
                or source_manifest.get("mask_architecture")!=MASK_KIND
                or teacher.config!=cfg or source_color.config!=cfg):
            raise ValueError("Expected matching original Shape Field V2 teacher checkpoint")
        shape.load_state_dict(teacher.state_dict())
        color.load_state_dict(source_color.state_dict())
        teacher.eval()
        teacher.requires_grad_(False)
        warm_start={
            "source_run_dir":str(init_v2),
            "mask_source_sha256":source_manifest["sha256"]["mask_field.pt"],
            "color_source_sha256":source_manifest["sha256"]["colorizer.pt"],
            "method":"copy both model weights, reset optimizer, optionally keep colorizer frozen",
        }
    # Only a NEW dedicated path; NEVER overwrite V1, V3 or DDPM weights.
    run=sora_run("shape-field-v21" if v21 else "shape-field-v2",args.output_root)
    run.mkdir(parents=True,exist_ok=False)
    try:
        optimizer=torch.optim.AdamW(shape.parameters(),lr=args.learning_rate)
        for step in range(1,args.mask_steps+1):
            images,tokens=balanced_training_batch(step,args.batch_size,args.seed)
            images,tokens=images.to(args.device),tokens.to(args.device)
            shape.train()
            optimizer.zero_grad(set_to_none=True)
            loss=geometry_objective(shape,images,tokens)
            paired=None
            total=loss["loss"]
            preserved=None
            if v21:
                pair_layouts,pair_masks,pair_tokens,_=matched_layout_shape_batch(
                    step,pair_examples,args.seed,args.device
                )
                paired=shape_word_separation_objective(
                    shape,pair_layouts,pair_masks,pair_tokens,
                    ranking_margin=rank_margin,ranking_weight=rank_weight
                )
                total=total+pair_weight*paired["loss"]
                if teacher is not None and preserve_circle_weight>0:
                    preserved=circle_teacher_preservation(
                        shape,teacher,images,tokens,args.seed+step*17
                    )
                    total=total+preserve_circle_weight*preserved["loss"]
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite neural geometry loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(shape.parameters(),1.0)
            optimizer.step()
            if step==1 or step%args.log_every==0 or step==args.mask_steps:
                report={
                    "stage":"neural-shape-field","step":step,
                    "total_loss":round(float(total.detach()),6),
                    "ordinary_bce":round(float(loss["bce"]),6),
                    "ordinary_dice_loss":round(float(loss["dice_loss"]),6),
                    "mask_probability_mean":round(float(loss["mask_mean"]),6)
                }
                if paired is not None:
                    report.update({
                        "paired_reconstruction_loss":round(float(paired["reconstruction_loss"]),6),
                        "shape_word_rank_loss":round(float(paired["shape_rank_loss"]),6),
                        "correct_shape_word_win_rate":round(float(
                            paired["correct_shape_word_win_rate"]),4),
                        "correct_vs_wrong_word_bce_advantage":round(float(
                            paired["correct_vs_wrong_word_bce_advantage"]),6)
                    })
                if preserved is not None:
                    report.update({
                        "circle_teacher_mse":round(float(preserved["loss"].detach()),7),
                        "circle_teacher_samples":preserved["examples"],
                    })
                print(json.dumps(report),flush=True)
        _save_weight(run/"mask_field.pt",shape,MASK_KIND,args.mask_steps,args.seed)
        optimizer=torch.optim.AdamW(color.parameters(),lr=args.learning_rate)
        for step in range(1,args.color_steps+1):
            images,tokens=balanced_training_batch(step,args.batch_size,
                                                   args.seed+30000)
            images,tokens=images.to(args.device),tokens.to(args.device)
            color.train()
            optimizer.zero_grad(set_to_none=True)
            loss=color_objective(color,images,tokens)
            if not torch.isfinite(loss["loss"]):
                raise FloatingPointError("Nonfinite neural RGB loss")
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(color.parameters(),1.0)
            optimizer.step()
            if step==1 or step%args.log_every==0 or step==args.color_steps:
                print(json.dumps({
                    "stage":"neural-colorizer","step":step,
                    "total_loss":round(float(loss["loss"].detach()),6),
                    "foreground_l1":round(float(loss["foreground_l1"]),6),
                    "background_l1":round(float(loss["background_l1"]),6)
                }),flush=True)
        _save_weight(run/"colorizer.pt",color,COLOR_KIND,args.color_steps,args.seed)
        manifest={
            "schema_version":1,"family":"sora","dataset_version":"v2",
            "model_family":"shape-field-v2",
            "config":asdict(cfg),
            "mask_architecture":MASK_KIND,
            "color_architecture":COLOR_KIND,
            "mask_steps":args.mask_steps,"color_steps":args.color_steps,
            "batch_size":args.batch_size,"learning_rate":args.learning_rate,
            "seed":args.seed,
            "training_recipe":("shape-word-matched-layout-v2.1" if v21 else "original-v2"),
            "warm_start":warm_start,
            "colorizer_reused_untrained":bool(args.color_steps==0 and init_v2),
            "circle_teacher_weight":preserve_circle_weight if teacher is not None else None,
            "paired_shape_objective":({
                "pair_weight":pair_weight,
                "rank_weight":rank_weight,
                "rank_margin":rank_margin,
                "paired_layout_count":pair_examples,
                "reconstruction":"same layout with 3 different target shapes x 2 languages",
            } if v21 else None),
            "sha256":{
                "mask_field.pt":checksum(run/"mask_field.pt"),
                "colorizer.pt":checksum(run/"colorizer.pt")
            },
            "training_data":"synthetic 32x32 RGB v2",
            "inference":"Gaussian layout -> learned neural mask -> learned text color RGB",
            "quality_status":"experimental",
            "release_status":"not-approved",
            "original_rgb_ddpm_unchanged":True,
            "shape_first_v1_unchanged":True,
        }
        with (run/"run.json").open("x",encoding="utf-8") as file:
            json.dump(manifest,file,indent=2,ensure_ascii=False)
            file.write("\n")
    except Exception:
        print(json.dumps({"incomplete_experimental_run":str(run)}),flush=True)
        raise
    print(json.dumps({
        "run_dir":str(run),
        "mask_checkpoint":str(run/"mask_field.pt"),
        "color_checkpoint":str(run/"colorizer.pt"),
        "release_status":"not-approved",
    },ensure_ascii=False,indent=2),flush=True)
    return run


def _save_learned_mask(mask,path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    bitmap=(mask[0].numpy()*255).round().astype(np.uint8)
    Image.fromarray(bitmap,mode="L").save(path)


def sample(run_dir,prompt,seed=42,output=None,device="cpu"):
    mask_model,color_model,_=load_run(run_dir,device)
    rgb,mask,_=generate(mask_model,color_model,prompt,seed,device)
    target=Path(output) if output else sora_run("shape-field-v2-sample")/"sample.png"
    save_image(rgb,target)
    mask_path=target.parent/(target.stem+"_mask.png")
    _save_learned_mask(mask,mask_path)
    return {"image":str(target),"learned_mask":str(mask_path),
            "prompt":prompt,"seed":seed}


@torch.no_grad()
def evaluate(run_dir,seed=42,output_dir=None,device="cpu"):
    mask_model,color_model,manifest=load_run(run_dir,device)
    dest=Path(output_dir) if output_dir else sora_run("shape-field-v2-eval")
    dest.mkdir(parents=True,exist_ok=False)
    masks=dest/"learned_masks"
    masks.mkdir(exist_ok=False)
    columns=4
    size=32*7
    spacing=25
    rgb_grid=Image.new("RGB",(columns*size,6*(size+spacing)),"white")
    mask_grid=Image.new("RGB",(columns*size,6*(size+spacing)),"white")
    rgb_pen=ImageDraw.Draw(rgb_grid)
    mask_pen=ImageDraw.Draw(mask_grid)
    predictions=[]
    items=[]
    for shape_index,shape in enumerate(SHAPES):
        for color_index,color in enumerate(COLORS):
            for lang_index,language in enumerate(("en","ar")):
                prompt=caption_v2(color,shape,language,0)
                # For each color, use identical z-layout across shapes/languages.
                shared_seed=seed+color_index
                rgb,mask,soft=generate(mask_model,color_model,prompt,
                                       shared_seed,device)
                filename=f"{color}_{shape}_{language}.png"
                save_image(rgb,dest/filename)
                _save_learned_mask(mask,masks/filename)
                metric=score_model_sample(rgb,color,shape)
                mask_metric=score_binary_mask(mask,shape)
                legacy=analyze_image(rgb,dataset_version="v2")
                predictions.append({
                    "expected_color":color,"expected_shape":shape,
                    "predicted_color":legacy["color"],
                    "predicted_shape":legacy["shape"],
                    "language":language,"seed":shared_seed,
                })
                items.append({
                    "image":filename,"prompt":prompt,
                    "language":language,"seed":shared_seed,
                    "expected_shape":shape,"expected_color":color,
                    "strict_geometry":metric,
                    "learned_mask_geometry":mask_metric,
                    "learned_mask":f"learned_masks/{filename}"
                })
                row=shape_index*2+lang_index
                x,y=color_index*size,row*(size+spacing)
                with Image.open(dest/filename) as png:
                    rgb_grid.paste(png.resize((size,size),Image.Resampling.NEAREST),(x,y))
                with Image.open(masks/filename) as png:
                    mask_grid.paste(png.convert("RGB").resize(
                        (size,size),Image.Resampling.NEAREST),(x,y))
                label=f"{color} {shape} {language}"
                rgb_pen.text((x+3,y+size+4),label,fill="black")
                mask_pen.text((x+3,y+size+4),label,fill="black")
    rgb_grid.save(dest/"preview_grid.png")
    mask_grid.save(dest/"learned_masks_preview_grid.png")
    quality={
        shape:{
            "images":8,
            "learned_mask_pass_count":sum(
                row["learned_mask_geometry"]["strict_geometry_pass"] for row in items
                if row["expected_shape"]==shape
            ),
            "geometry_pass_count":sum(
                row["strict_geometry"]["qualified_geometry"] for row in items
                if row["expected_shape"]==shape
            ),
            "strict_prompt_pass_count":sum(
                row["strict_geometry"]["strict_prompt_pass"] for row in items
                if row["expected_shape"]==shape
            ),
        } for shape in SHAPES
    }
    result={
        "schema_version":1,"family":"sora","model_family":"shape-field-v2",
        "dataset_version":"v2","source_run_dir":str(run_dir),
        "seed":seed,"sampler":"neural-mask-gaussian-layout",
        "same_latent_per_color":True,
        "metrics":summarize(predictions,dataset_version="v2"),
        "strict_by_shape":quality,
        "strict_prompt_accuracy":round(
            sum(row["strict_geometry"]["strict_prompt_pass"] for row in items)/24,4
        ),
        "items":items,
        "release_status":"not-approved",
        "human_visual_review_required":True,
    }
    with (dest/"report.json").open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2,ensure_ascii=False)
        file.write("\n")
    print(json.dumps({
        "preview_grid":str(dest/"preview_grid.png"),
        "learned_masks_preview_grid":str(dest/"learned_masks_preview_grid.png"),
        "report":str(dest/"report.json"),
        "images":24,"strict_by_shape":quality,
        "strict_prompt_accuracy":result["strict_prompt_accuracy"],
        "metrics":result["metrics"],
        "release_status":"not-approved",
    },ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    tr=commands.add_parser("train")
    tr.add_argument("--mask-steps",type=int,default=160)
    tr.add_argument("--color-steps",type=int,default=100)
    tr.add_argument("--batch-size",type=int,default=12)
    tr.add_argument("--hidden",type=int,default=64)
    tr.add_argument("--word-dim",type=int,default=32)
    tr.add_argument("--learning-rate",type=float,default=0.001)
    tr.add_argument("--seed",type=int,default=42)
    tr.add_argument("--device",default="cpu",choices=("cpu","cuda","mps"))
    tr.add_argument("--cpu-threads",type=int,default=2)
    tr.add_argument("--log-every",type=int,default=20)
    tr.add_argument("--output-root",default=None)
    tr.add_argument("--shape-v21",action="store_true",
                    help="Opt-in V2.1 same-layout 3-shape bilingual supervision; stores a SEPARATE run")
    tr.add_argument("--pair-weight",type=float,default=1.0)
    tr.add_argument("--rank-weight",type=float,default=0.1)
    tr.add_argument("--rank-margin",type=float,default=0.03)
    tr.add_argument("--paired-layout-count",type=int,default=4)
    tr.add_argument("--init-from-v2",default=None,
                    help="Optional exact original V2 run dir; clone weights into NEW V2.1 run")
    tr.add_argument("--circle-preserve-weight",type=float,default=0.2)
    ev=commands.add_parser("evaluate")
    ev.add_argument("--run-dir",required=True)
    ev.add_argument("--seed",type=int,default=42)
    ev.add_argument("--output-dir",default=None)
    ev.add_argument("--device",default="cpu",choices=("cpu","cuda","mps"))
    sm=commands.add_parser("sample")
    sm.add_argument("--run-dir",required=True)
    sm.add_argument("--prompt",required=True)
    sm.add_argument("--seed",type=int,default=42)
    sm.add_argument("--output",default=None)
    sm.add_argument("--device",default="cpu",choices=("cpu","cuda","mps"))
    args=parser.parse_args()
    if args.action=="train":
        train(args)
    elif args.action=="evaluate":
        evaluate(args.run_dir,args.seed,args.output_dir,args.device)
    else:
        print(json.dumps(sample(args.run_dir,args.prompt,args.seed,args.output,
                                args.device),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

"""Train and infer Sora Nano bilingual RGB pixel diffusion.

Training uses labeled synthetic images, while inference starts from RGB noise
and never calls the procedural image renderer.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import uuid

from PIL import Image, ImageDraw
import numpy as np
import torch

from src.pixel_metrics import analyze_image, summarize
from src.pixel_diffusion import (
    COLORS, SHAPES, WORDS, PixelConfig, PixelDenoiser, caption,
    generate, save_image, synthetic_example, training_loss, vocabulary_for,
    shape_aware_training_objective,
)
from src.pixel_dataset_v2 import (
    CONCEPTS as CONCEPTS_V2, caption_v2, balanced_training_batch,
)
from src.pixel_shape_contrast import matched_shape_batch, matched_shape_objective
from src.pixel_preservation import preservation_objective

PROMPTS = [
    ("red circle", "دائرة حمراء"), ("green circle", "دائرة خضراء"),
    ("blue circle", "دائرة زرقاء"), ("red square", "مربع أحمر"),
    ("green square", "مربع أخضر"), ("blue square", "مربع أزرق"),
]


def load_model(path,device="cpu"):
    record=torch.load(path,map_location="cpu",weights_only=True)
    if (record.get("schema_version")!=1
            or record.get("architecture") not in (
                "rgb-unet-text-v1","rgb-unet-spatial-text-v2"
            )):
        raise ValueError("Incompatible Sora pixel diffusion checkpoint")
    cfg=PixelConfig(**record["config"])
    if cfg.architecture != record["architecture"]:
        raise ValueError("Checkpoint architecture does not match config")
    if record.get("vocabulary") != list(vocabulary_for(cfg.dataset_version)):
        raise ValueError("Checkpoint vocabulary does not match its dataset version")
    model=PixelDenoiser(cfg)
    model.load_state_dict(record["model"],strict=True)
    model.cfg_drop_prob = float(record.get("cfg_drop_prob", 0.0))
    return model.to(device)


def initialize_v2_from_v1(model, checkpoint):
    """Warm-start v2 RGB pixels from a trained v1 RGB checkpoint.

    All U-Net/time/text-projection weights are transferred. Existing vocabulary
    rows keep their indices. Only new v2 token rows need initialization.
    This is weight transfer, not a procedural inference shortcut or v2 success.
    """
    if model.config.dataset_version != "v2":
        raise ValueError("Warm start is only supported for a new v2 model")
    source = load_model(checkpoint, "cpu")
    if source.config.dataset_version != "v1":
        raise ValueError("Warm start requires a v1 RGB pixel checkpoint")
    for field in ("channels", "text_dim", "diffusion_steps", "image_size",
                  "max_tokens", "architecture"):
        if getattr(source.config, field) != getattr(model.config, field):
            raise ValueError(f"Incompatible warm-start architecture: {field}")
    old_state = source.state_dict()
    new_state = model.state_dict()
    if set(old_state) != set(new_state):
        raise ValueError("Warm-start state keys differ")
    for name, tensor in old_state.items():
        if name == "text.token.weight":
            if (tensor.ndim != 2 or new_state[name].shape[0] <= tensor.shape[0]
                    or new_state[name].shape[1] != tensor.shape[1]):
                raise ValueError("Unexpected vocabulary expansion")
            new_state[name][:tensor.shape[0]] = tensor.to(
                device=new_state[name].device, dtype=new_state[name].dtype
            )
            # Semantic interpolation is only an INITIALIZATION; training must
            # learn these new word vectors and all new image concepts.
            old_words = vocabulary_for("v1")
            new_words = vocabulary_for("v2")
            synonyms = {
                "yellow": ("red", "green"),
                "triangle": ("circle", "square"),
                "مثلث": ("مربع", "دائرة"),
                "أصفر": ("أحمر", "أخضر"),
                "صفراء": ("حمراء", "خضراء"),
                "light": ("white", "background"),
                "فاتحة": ("بيضاء", "خلفية"),
            }
            for word, components in synonyms.items():
                if word not in new_words:
                    continue
                if any(w not in old_words for w in components):
                    raise ValueError(f"Unknown warm-start embedding components for {word}")
                new_state[name][new_words.index(word)] = torch.stack([
                    tensor[old_words.index(w)] for w in components
                ]).mean(0).to(new_state[name].device)
        else:
            if new_state[name].shape != tensor.shape:
                raise ValueError(f"Incompatible warm-start tensor: {name}")
            new_state[name] = tensor.to(
                device=new_state[name].device, dtype=new_state[name].dtype
            ).clone()
    model.load_state_dict(new_state, strict=True)
    return {
        "source_checkpoint": str(checkpoint),
        "source_dataset_version": "v1",
        "source_training_steps": torch.load(
            checkpoint, map_location="cpu", weights_only=True
        ).get("training_steps"),
        "method": "v1 to v2 weight transfer, new token embeddings initialized by semantic averaging",
    }


def initialize_v2_from_v2(model, checkpoint):
    """Start a NEW v2 fine-tuning run from existing v2 model weights.

    AdamW state is intentionally reset; this is a fine-tune, not an exact
    optimizer-state resume. The source checkpoint is never overwritten.
    """
    if model.config.dataset_version != "v2":
        raise ValueError("v2 fine-tuning requires a v2 destination")
    source = load_model(checkpoint, "cpu")
    if source.config.dataset_version != "v2":
        raise ValueError("v2 fine-tuning requires a v2 RGB checkpoint")
    source_state=source.state_dict()
    if source.config == model.config:
        model.load_state_dict(source_state,strict=True)
        transfer_method="v2 RGB weights reloaded; optimizer reset"
    elif (source.config.architecture=="rgb-unet-text-v1"
            and model.config.architecture=="rgb-unet-spatial-text-v2"
            and all(getattr(source.config,key)==getattr(model.config,key)
                    for key in ("channels","text_dim","diffusion_steps",
                                "image_size","max_tokens","dataset_version"))):
        # New spatial-attention residuals start at EXACTLY zero output.
        # Every older U-Net / text layer retains its original trained weights.
        target_state=model.state_dict()
        for name,tensor in source_state.items():
            if name not in target_state or target_state[name].shape!=tensor.shape:
                raise ValueError(f"Incompatible transferable RGB weight: {name}")
            target_state[name]=tensor.to(target_state[name].device).clone()
        added=set(target_state)-set(source_state)
        if not added or any(not k.startswith(("spatial16.","spatial32."))
                            for k in added):
            raise ValueError("Unexpected new-architecture parameter keys")
        model.load_state_dict(target_state,strict=True)
        transfer_method="v2 pooled RGB weights -> spatial-text RGB architecture; zero-initialized new attention; optimizer reset"
    else:
        raise ValueError("Fine-tuning checkpoint architecture/config does not match")
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        "source_checkpoint": str(checkpoint),
        "source_dataset_version": "v2",
        "source_architecture": source.config.architecture,
        "source_training_steps": metadata.get("training_steps"),
        "method": transfer_method,
    }


def require_guidance_training(model, guidance_scale):
    if guidance_scale != 1.0 and getattr(model, "cfg_drop_prob", 0.0) <= 0:
        raise ValueError(
            "Classifier-free guidance needs a checkpoint trained with "
            "--cfg-drop-prob > 0. Retrain; do not reuse the old pixel checkpoint."
        )


def train(args):
    if (args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.cpu_threads < 1 or args.log_every < 1
            or not 0 <= args.cfg_drop_prob < 1):
        raise ValueError("Invalid training arguments")
    init_v1 = getattr(args, "init_from_v1", None)
    init_v2 = getattr(args, "init_from_v2", None)
    triangle_fraction = getattr(args, "triangle_fraction", None)
    conditioning = getattr(args,"conditioning","pooled")
    silhouette_weight = getattr(args,"silhouette_weight",0.15)
    paired_shape_weight = getattr(args,"paired_shape_weight",0.0)
    paired_shape_pairs = getattr(args,"paired_shape_pairs",4)
    paired_silhouette_weight = getattr(args,"paired_silhouette_weight",0.5)
    preserve_weight = getattr(args,"preserve_weight",0.0)
    preserve_null_weight = getattr(args,"preserve_null_weight",0.5)
    if conditioning not in ("pooled","spatial"):
        raise ValueError("conditioning must be pooled or spatial")
    if conditioning=="spatial" and args.dataset_version!="v2":
        raise ValueError("Spatial text architecture is only supported for dataset v2")
    if not 0 <= silhouette_weight <= 2:
        raise ValueError("silhouette_weight must be between 0 and 2")
    if not 0 <= paired_shape_weight <= 2:
        raise ValueError("paired_shape_weight must be between 0 and 2")
    if not 0 <= paired_silhouette_weight <= 2:
        raise ValueError("paired_silhouette_weight must be between 0 and 2")
    if paired_shape_pairs < 1:
        raise ValueError("paired_shape_pairs must be positive")
    if paired_shape_weight > 0 and (conditioning!="spatial" or args.dataset_version!="v2"):
        raise ValueError("Matched counterfactual shape loss requires spatial v2 architecture")
    if not 0 <= preserve_weight <= 10 or not np.isfinite(preserve_weight):
        raise ValueError("preserve_weight must be finite in [0,10]")
    if not 0 <= preserve_null_weight <= 2 or not np.isfinite(preserve_null_weight):
        raise ValueError("preserve_null_weight must be finite in [0,2]")
    if preserve_weight > 0 and (not init_v2 or conditioning!="spatial"
                                or args.dataset_version!="v2"):
        raise ValueError("Teacher preservation requires spatial v2 fine-tuning from --init-from-v2")
    if preserve_weight > 0 and triangle_fraction is not None and triangle_fraction >= 1:
        raise ValueError("Teacher preservation requires circle/square replay examples")
    if init_v1 and init_v2:
        raise ValueError("Cannot initialize from both v1 and v2")
    if triangle_fraction is not None and args.dataset_version != "v2":
        raise ValueError("Triangle curriculum is only supported for dataset v2")
    if triangle_fraction is not None and not 0 <= triangle_fraction <= 1:
        raise ValueError("triangle_fraction must be in [0,1]")
    if args.device=="cpu": torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    architecture=("rgb-unet-spatial-text-v2" if conditioning=="spatial"
                  else "rgb-unet-text-v1")
    cfg=PixelConfig(channels=args.channels,diffusion_steps=args.diffusion_steps,
                    dataset_version=args.dataset_version,architecture=architecture)
    model=PixelDenoiser(cfg).to(args.device)
    warm_start = None
    if init_v1 is not None:
        warm_start = initialize_v2_from_v1(model, init_v1)
    elif init_v2 is not None:
        warm_start = initialize_v2_from_v2(model, init_v2)
    if warm_start is not None:
        print(json.dumps({"warm_start":warm_start},ensure_ascii=False),flush=True)
    teacher=None
    if preserve_weight > 0:
        teacher=load_model(init_v2,args.device)
        if teacher.config != cfg:
            raise ValueError("Teacher checkpoint must have identical model configuration")
        teacher.eval()
        teacher.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    run=Path(args.output_root)/(
        "pixel-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    for step in range(1,args.steps+1):
        if cfg.dataset_version == "v2":
            images, tokens = balanced_training_batch(
                step,args.batch_size,args.seed,triangle_fraction=triangle_fraction
            )
            images, tokens = images.to(args.device), tokens.to(args.device)
        else:
            batch=[]
            for j in range(args.batch_size):
                class_id=(step*args.batch_size+j)%6
                color=list(COLORS)[class_id%3]
                shape=SHAPES[class_id//3]
                language="ar" if (j+step)%2 else "en"
                batch.append(synthetic_example(
                    args.seed+step*4099+j*13,color,shape,language))
            images=torch.stack([item[0] for item in batch]).to(args.device)
            tokens=torch.stack([item[1] for item in batch]).to(args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if cfg.architecture=="rgb-unet-spatial-text-v2":
            objectives=shape_aware_training_objective(
                model,images,tokens,cfg_drop_prob=args.cfg_drop_prob,
                silhouette_weight=silhouette_weight
            )
            loss=objectives["total_loss"]
            noise_value=float(objectives["noise_mse"])
            silhouette_value=float(objectives["silhouette_loss"])
        else:
            loss=training_loss(model,images,tokens,cfg_drop_prob=args.cfg_drop_prob)
            noise_value=float(loss.detach())
            silhouette_value=None
        paired_log=None
        if paired_shape_weight > 0:
            pair_images,pair_tokens,_=matched_shape_batch(
                step,paired_shape_pairs,args.seed,split="train"
            )
            pair_objectives=matched_shape_objective(
                model,pair_images.to(args.device),pair_tokens.to(args.device),
                silhouette_weight=paired_silhouette_weight
            )
            loss=loss+paired_shape_weight*pair_objectives["total_loss"]
            paired_log={
                "paired_shape_noise_mse":round(float(pair_objectives["pair_noise_mse"]),6),
                "paired_shape_difference_loss":round(float(pair_objectives["pair_difference_loss"]),6),
                "paired_shape_loss":round(float(pair_objectives["pair_shape_loss"]),6),
            }
        preserve_log=None
        if preserve_weight > 0:
            preserved=preservation_objective(
                model,teacher,images,tokens,null_weight=preserve_null_weight
            )
            loss=loss+preserve_weight*preserved["total_loss"]
            preserve_log={
                "preserve_conditional_mse":round(float(preserved["conditional_loss"]),6),
                "preserve_null_mse":round(float(preserved["null_loss"]),6),
                "protected_old_shape_examples":preserved["protected_examples"],
            }
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite image diffusion loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            report={"step":step,"noise_mse":round(noise_value,6)}
            if silhouette_value is not None:
                report["silhouette_dice_loss"]=round(silhouette_value,6)
                report["total_loss"]=round(float(loss.detach()),6)
            if paired_log is not None:
                report.update(paired_log)
                report["total_loss"]=round(float(loss.detach()),6)
            if preserve_log is not None:
                report.update(preserve_log)
                report["total_loss"]=round(float(loss.detach()),6)
            print(json.dumps(report),flush=True)
    path=run/"checkpoint.pt"
    torch.save({
        "schema_version":1,"architecture":cfg.architecture,"family":"sora",
        "config":asdict(cfg),"vocabulary":list(vocabulary_for(cfg.dataset_version)),
        "dataset_version":cfg.dataset_version,
        "cfg_drop_prob":args.cfg_drop_prob,
        "warm_start":warm_start,
        "triangle_fraction":triangle_fraction,
        "silhouette_weight":silhouette_weight if conditioning=="spatial" else None,
        "paired_shape_weight":paired_shape_weight,
        "paired_shape_pairs":paired_shape_pairs if paired_shape_weight>0 else None,
        "paired_silhouette_weight":(
            paired_silhouette_weight if paired_shape_weight>0 else None
        ),
        "preserve_weight":preserve_weight,
        "preserve_null_weight":preserve_null_weight if preserve_weight>0 else None,
        "preserve_teacher_checkpoint":str(init_v2) if preserve_weight>0 else None,
        "model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "seed":args.seed,"training_steps":args.steps,
        "training_distribution":(
            "synthetic v2 4-color/3-shape bilingual RGB images, variable position/size/light background"
            if cfg.dataset_version=="v2" else
            "synthetic v1 six-class bilingual one-shape RGB images"
        ),
        "quality_status":"experimental; visual quality must be verified",
    },path)
    print(json.dumps({"checkpoint":str(path)}))
    return path


def evaluate(model,path,seed=42,sample_steps=None,scale=8,sampler="posterior_mean",guidance_scale=1.0,shared_noise=False,x0_clipping="fixed"):
    require_guidance_training(model, guidance_scale)
    version = model.config.dataset_version
    prompts = PROMPTS if version=="v1" else [
        (f"{color} {shape}",caption_v2(color,shape,"ar"))
        for color,shape in CONCEPTS_V2
    ]
    columns = 3 if version=="v1" else 4
    row_count = 2 * (len(prompts)//columns)
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    grid=Image.new("RGB",(columns*32*scale,row_count*(32*scale+25)),"white")
    draw=ImageDraw.Draw(grid)
    entries=[]
    predictions=[]
    device=next(model.parameters()).device
    for j,(english,arabic) in enumerate(prompts):
        for lang_index,prompt in enumerate((english,arabic)):
            sample_seed=seed if shared_noise else seed+j
            image=generate(model,prompt,seed=sample_seed,device=device,steps=sample_steps,sampler=sampler,guidance_scale=guidance_scale,x0_clipping=x0_clipping)
            name=english.replace(" ","_")+("_en.png" if lang_index==0 else "_ar.png")
            save_image(image,path/name)
            thumbnail=Image.open(path/name).resize((32*scale,32*scale),Image.Resampling.NEAREST)
            row=2*(j//columns)+lang_index
            col=j%columns
            x,y=col*32*scale,row*(32*scale+25)
            grid.paste(thumbnail,(x,y))
            draw.text((x+6,y+32*scale+4),english+(" EN" if lang_index==0 else " AR"),fill="black")
            expected_color, expected_shape = english.split()
            finding = analyze_image(image,dataset_version=version)
            language = "en" if lang_index == 0 else "ar"
            predictions.append({
                "expected_color":expected_color,"expected_shape":expected_shape,
                "predicted_color":finding["color"],"predicted_shape":finding["shape"],
                "foreground_pixels":finding["foreground_pixels"],
                "fill_ratio":finding["fill_ratio"],
                "language":language,"seed":sample_seed,
            })
            entries.append({"prompt":prompt,"image":name,"seed":sample_seed,
                            "language":language,"heuristic":finding})
    grid.save(path/"preview_grid.png")
    metrics = summarize(predictions,dataset_version=version)
    (path/"report.json").write_text(json.dumps({
        "schema_version":1,"architecture":model.config.architecture,
        "dataset_version":version,
        "training_data":"synthetic; no photorealistic or open-domain data",
        "image_pixels_generated_by_diffusion":True,
        "sampler":sampler,"sample_steps":sample_steps,
        "x0_clipping":x0_clipping,
        "shared_noise_across_concepts":bool(shared_noise),
        "guidance_scale":guidance_scale,
        "checkpoint_cfg_drop_prob":getattr(model, "cfg_drop_prob", 0.0),
        "metrics":metrics,
        "visual_quality":"not automatically certified",
        "items":entries,
    },indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"preview_grid":str(path/"preview_grid.png"),"images":len(entries),"metrics":metrics}))
    return metrics


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    t=sub.add_parser("train")
    t.add_argument("--steps",type=int,default=3000)
    t.add_argument("--batch-size",type=int,default=12)
    t.add_argument("--learning-rate",type=float,default=0.0003)
    t.add_argument("--seed",type=int,default=42)
    t.add_argument("--base-channels","--channels",dest="channels",type=int,default=32)
    t.add_argument("--cfg-drop-prob",type=float,default=0.15)
    t.add_argument("--dataset-version",choices=("v1","v2"),default="v1")
    init=t.add_mutually_exclusive_group()
    init.add_argument("--init-from-v1",default=None,help="Optional original v1 RGB checkpoint to warm-start new v2 model")
    init.add_argument("--init-from-v2",default=None,help="Optional trained v2 checkpoint to fine-tune; new optimizer, new checkpoint")
    t.add_argument("--triangle-fraction",type=float,default=None,
                   help="Opt-in v2 curriculum; e.g. 0.50 trains 50 percent triangle examples")
    t.add_argument("--conditioning",choices=("pooled","spatial"),default="pooled",
                   help="Opt-in new per-pixel text attention architecture")
    t.add_argument("--silhouette-weight",type=float,default=0.15,
                   help="Spatial model training-only RGB silhouette Dice loss weight")
    t.add_argument("--paired-shape-weight",type=float,default=0.0,
                   help="Opt-in counterfactual square/triangle loss weight (spatial v2 only)")
    t.add_argument("--paired-shape-pairs",type=int,default=4,
                   help="Number of matched square/triangle pairs per step")
    t.add_argument("--paired-silhouette-weight",type=float,default=0.5,
                   help="Shape difference mask weight within paired objective")
    t.add_argument("--preserve-weight",type=float,default=0.0,
                   help="Optional frozen-source circle/square denoising preservation loss")
    t.add_argument("--preserve-null-weight",type=float,default=0.5,
                   help="Weight on teacher CFG null-prompt preservation")
    t.add_argument("--diffusion-steps",type=int,default=100)
    t.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    t.add_argument("--cpu-threads",type=int,default=2)
    t.add_argument("--log-every",type=int,default=100)
    t.add_argument("--output-root",default="storage/artifacts/sora/nano/mvp")
    e=sub.add_parser("evaluate")
    e.add_argument("--checkpoint",required=True)
    e.add_argument("--output-dir",required=True)
    e.add_argument("--seed",type=int,default=42)
    e.add_argument("--sample-steps",type=int,default=None)
    e.add_argument("--sampler",choices=("posterior_mean","ancestral","ddim"),default="posterior_mean")
    e.add_argument("--guidance-scale",type=float,default=1.0)
    e.add_argument("--x0-clipping",choices=("fixed","dynamic"),default="fixed",
                   help="Inference-only x0 clipping policy; fixed preserves existing behavior")
    e.add_argument("--shared-noise",action="store_true",help="Use identical initial noise for every concept and language")
    e.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    s=sub.add_parser("sample")
    s.add_argument("--checkpoint",required=True)
    s.add_argument("--prompt",required=True)
    s.add_argument("--output",required=True)
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--sample-steps",type=int,default=None)
    s.add_argument("--sampler",choices=("posterior_mean","ancestral","ddim"),default="posterior_mean")
    s.add_argument("--guidance-scale",type=float,default=1.0)
    s.add_argument("--x0-clipping",choices=("fixed","dynamic"),default="fixed")
    s.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    args=p.parse_args()
    if args.action=="train": train(args)
    elif args.action=="evaluate":
        evaluate(load_model(args.checkpoint,args.device),args.output_dir,args.seed,args.sample_steps,sampler=args.sampler,guidance_scale=args.guidance_scale,shared_noise=args.shared_noise,x0_clipping=args.x0_clipping)
    else:
        model=load_model(args.checkpoint,args.device)
        require_guidance_training(model, args.guidance_scale)
        save_image(generate(model,args.prompt,args.seed,args.device,args.sample_steps,sampler=args.sampler,guidance_scale=args.guidance_scale,x0_clipping=args.x0_clipping),args.output)
        print(json.dumps({"output":args.output,"prompt":args.prompt},ensure_ascii=False))

if __name__=="__main__": main()

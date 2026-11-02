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
    generate, save_image, synthetic_example, training_loss,
)

PROMPTS = [
    ("red circle", "دائرة حمراء"), ("green circle", "دائرة خضراء"),
    ("blue circle", "دائرة زرقاء"), ("red square", "مربع أحمر"),
    ("green square", "مربع أخضر"), ("blue square", "مربع أزرق"),
]


def load_model(path,device="cpu"):
    record=torch.load(path,map_location="cpu",weights_only=True)
    if (record.get("schema_version")!=1
            or record.get("architecture")!="rgb-unet-text-v1"
            or record.get("vocabulary")!=list(WORDS)):
        raise ValueError("Incompatible Sora pixel diffusion checkpoint")
    cfg=PixelConfig(**record["config"])
    model=PixelDenoiser(cfg)
    model.load_state_dict(record["model"],strict=True)
    model.cfg_drop_prob = float(record.get("cfg_drop_prob", 0.0))
    return model.to(device)


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
    if args.device=="cpu": torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg=PixelConfig(channels=args.channels,diffusion_steps=args.diffusion_steps)
    model=PixelDenoiser(cfg).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    run=Path(args.output_root)/(
        "pixel-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    for step in range(1,args.steps+1):
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
        loss=training_loss(model,images,tokens,cfg_drop_prob=args.cfg_drop_prob)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite image diffusion loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            print(json.dumps({"step":step,"noise_mse":round(float(loss.detach()),6)}),flush=True)
    path=run/"checkpoint.pt"
    torch.save({
        "schema_version":1,"architecture":cfg.architecture,"family":"sora",
        "config":asdict(cfg),"vocabulary":list(WORDS),
        "cfg_drop_prob":args.cfg_drop_prob,
        "model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "seed":args.seed,"training_steps":args.steps,
        "training_distribution":"synthetic one-shape RGB images with English and Arabic prompts",
        "quality_status":"experimental; visual quality must be verified",
    },path)
    print(json.dumps({"checkpoint":str(path)}))
    return path


def evaluate(model,path,seed=42,sample_steps=None,scale=8,sampler="posterior_mean",guidance_scale=1.0):
    require_guidance_training(model, guidance_scale)
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    grid=Image.new("RGB",(3*32*scale,4*(32*scale+25)),"white")
    draw=ImageDraw.Draw(grid)
    entries=[]
    predictions=[]
    device=next(model.parameters()).device
    for j,(english,arabic) in enumerate(PROMPTS):
        for lang_index,prompt in enumerate((english,arabic)):
            image=generate(model,prompt,seed=seed+j,device=device,steps=sample_steps,sampler=sampler,guidance_scale=guidance_scale)
            name=english.replace(" ","_")+("_en.png" if lang_index==0 else "_ar.png")
            save_image(image,path/name)
            thumbnail=Image.open(path/name).resize((32*scale,32*scale),Image.Resampling.NEAREST)
            row=2*(j//3)+lang_index
            col=j%3
            x,y=col*32*scale,row*(32*scale+25)
            grid.paste(thumbnail,(x,y))
            draw.text((x+6,y+32*scale+4),english+(" EN" if lang_index==0 else " AR"),fill="black")
            expected_color, expected_shape = english.split()
            finding = analyze_image(image)
            language = "en" if lang_index == 0 else "ar"
            predictions.append({
                "expected_color":expected_color,"expected_shape":expected_shape,
                "predicted_color":finding["color"],"predicted_shape":finding["shape"],
                "foreground_pixels":finding["foreground_pixels"],
                "fill_ratio":finding["fill_ratio"],
                "language":language,"seed":seed+j,
            })
            entries.append({"prompt":prompt,"image":name,"seed":seed+j,
                            "language":language,"heuristic":finding})
    grid.save(path/"preview_grid.png")
    metrics = summarize(predictions)
    (path/"report.json").write_text(json.dumps({
        "schema_version":1,"architecture":"rgb-unet-text-v1",
        "training_data":"synthetic; no photorealistic or open-domain data",
        "image_pixels_generated_by_diffusion":True,
        "sampler":sampler,"sample_steps":sample_steps,
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
    e.add_argument("--sampler",choices=("posterior_mean","ddim"),default="posterior_mean")
    e.add_argument("--guidance-scale",type=float,default=1.0)
    e.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    s=sub.add_parser("sample")
    s.add_argument("--checkpoint",required=True)
    s.add_argument("--prompt",required=True)
    s.add_argument("--output",required=True)
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--sample-steps",type=int,default=None)
    s.add_argument("--sampler",choices=("posterior_mean","ddim"),default="posterior_mean")
    s.add_argument("--guidance-scale",type=float,default=1.0)
    s.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    args=p.parse_args()
    if args.action=="train": train(args)
    elif args.action=="evaluate":
        evaluate(load_model(args.checkpoint,args.device),args.output_dir,args.seed,args.sample_steps,sampler=args.sampler,guidance_scale=args.guidance_scale)
    else:
        model=load_model(args.checkpoint,args.device)
        require_guidance_training(model, args.guidance_scale)
        save_image(generate(model,args.prompt,args.seed,args.device,args.sample_steps,sampler=args.sampler,guidance_scale=args.guidance_scale),args.output)
        print(json.dumps({"output":args.output,"prompt":args.prompt},ensure_ascii=False))

if __name__=="__main__": main()

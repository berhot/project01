"""Train and evaluate a SEPARATE Sora Nano V3.1 radius-conditioned pilot.

This is a narrow experiment: squares only, 32x32, continuously varied corner
radius. It does not implement general conversational image generation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.shape_field_v31 import (
    RadiusConfig, RadiusMaskField, training_batch, training_mask,
    radius_objective, config_dict
)
from src.storage_paths import sora_run


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(args):
    if (args.steps < 1 or args.batch_size < 2 or args.log_every < 1
            or args.cpu_threads < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0):
        raise ValueError("Invalid training arguments")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    model = RadiusMaskField(RadiusConfig(hidden=args.hidden)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    root = sora_run("shape-field-v31-radius", args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps+1):
        layout, radii, masks = training_batch(step, args.batch_size, args.seed)
        layout, radii, masks = (x.to(args.device) for x in (layout,radii,masks))
        optimizer.zero_grad(set_to_none=True)
        result = radius_objective(model, layout, radii, masks)
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError("Nonfinite V3.1 radius loss")
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step":step,
                "total_loss":round(float(result["loss"].detach()),6),
                "bce":round(float(result["bce"]),6),
                "dice":round(float(result["dice"]),6),
                "edge":round(float(result["edge"]),6)}), flush=True)
    checkpoint = root/"radius_field.pt"
    torch.save({"schema_version":1,"architecture":"radius-mask-v3.1",
                "config":config_dict(model),"state_dict":{
                    k:v.detach().cpu() for k,v in model.state_dict().items()},
                "status":"experimental","steps":args.steps},checkpoint)
    manifest = {
        "schema_version":1, "family":"sora", "training_recipe":"radius-field-v3.1",
        "checkpoint_sha256":sha256(checkpoint), "steps":args.steps,
        "seed":args.seed, "quality_status":"experimental",
        "release_status":"not-approved",
        "training_data":"synthetic radius-varied square masks",
        "inference":"neural occupancy from layout and continuous radius",
        "scope":"single square, 32x32; no general prompt generation"}
    (root/"run.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"run_dir":str(root),"release_status":"not-approved"}))
    return root


def load(run_dir,device="cpu"):
    root=Path(run_dir)
    record=json.loads((root/"run.json").read_text())
    file=root/"radius_field.pt"
    if (record.get("training_recipe")!="radius-field-v3.1"
            or record.get("release_status")!="not-approved"
            or not file.is_file()
            or record.get("checkpoint_sha256")!=sha256(file)):
        raise ValueError("Invalid V3.1 experimental checkpoint")
    state=torch.load(file,map_location="cpu",weights_only=True)
    if state.get("architecture")!="radius-mask-v3.1":
        raise ValueError("Unexpected model architecture")
    model=RadiusMaskField(RadiusConfig(**state["config"])).to(device)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval()
    return model


def _iou(a,b):
    x=a>0.5
    y=b>0.5
    union=int((x|y).sum())
    return float((x&y).sum())/union if union else 1.0


@torch.no_grad()
def evaluate(run_dir,seed=12345,device="cpu",output_dir=None):
    model=load(run_dir,device)
    root=Path(output_dir) if output_dir else sora_run("shape-field-v31-radius-eval")
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for radius in (0.0,0.25,0.5,0.75,1.0):
        for sample_index in range(8):
            target, layout=training_mask(
                seed+sample_index*71+int(radius*1000),radius)
            binary, probs=model.predict(
                layout[None].to(device),
                torch.tensor([[radius]],dtype=torch.float32,device=device))
            iou=_iou(binary[0,0].cpu(),target[0])
            rows.append({"radius":radius,"index":sample_index,
                         "heldout_iou":round(iou,6)})
            if sample_index==0:
                truth=(target[0].numpy()*255).astype(np.uint8)
                learned=(binary[0,0].cpu().numpy()*255).astype(np.uint8)
                image=np.concatenate((truth,learned),axis=1)
                Image.fromarray(image).resize((512,256),Image.Resampling.NEAREST).save(
                    root/f"radius_{int(radius*100):03d}.png")
    grouped={str(radius):round(sum(r["heldout_iou"] for r in rows
                if r["radius"]==radius)/8,6)
             for radius in (0.0,0.25,0.5,0.75,1.0)}
    report={"source_run_dir":str(run_dir),"seed":seed,"items":rows,
            "heldout_iou_by_radius":grouped,"no_training":True,
            "release_approved":False}
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"report":str(root/"report.json"),
                      "heldout_iou_by_radius":grouped,
                      "release_approved":False},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    tr=commands.add_parser("train")
    tr.add_argument("--steps",type=int,default=400)
    tr.add_argument("--batch-size",type=int,default=20)
    tr.add_argument("--hidden",type=int,default=64)
    tr.add_argument("--learning-rate",type=float,default=0.001)
    tr.add_argument("--seed",type=int,default=42)
    tr.add_argument("--log-every",type=int,default=40)
    tr.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    tr.add_argument("--cpu-threads",type=int,default=2)
    tr.add_argument("--output-root")
    ev=commands.add_parser("evaluate")
    ev.add_argument("--run-dir",required=True)
    ev.add_argument("--seed",type=int,default=12345)
    ev.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    ev.add_argument("--output-dir")
    args=parser.parse_args()
    if args.action=="train":
        train(args)
    else:
        evaluate(args.run_dir,args.seed,args.device,args.output_dir)


if __name__=="__main__":
    main()

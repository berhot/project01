"""Inspect local Sora RGB checkpoints safely, selecting actual v1/v2 paths.

Use: python3 -m scripts.pixel_checkpoints list
     python3 -m scripts.pixel_checkpoints latest --dataset-version v1 --min-steps 10000

Prints plain path in latest mode to support shell capture. Does not modify
model weights or search unrelated Nabra/FANN artifacts.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from src.pixel_diffusion import vocabulary_for


def read_metadata(path):
    record=torch.load(path,map_location="cpu",weights_only=True)
    if record.get("architecture") not in ("rgb-unet-text-v1","rgb-unet-spatial-text-v2"):
        raise ValueError(f"Not an RGB pixel checkpoint: {path}")
    cfg=record["config"]
    version=cfg.get("dataset_version","v1")
    if version not in ("v1","v2"):
        raise ValueError(f"Invalid dataset_version: {path}")
    if cfg.get("architecture","rgb-unet-text-v1") != record.get("architecture"):
        raise ValueError(f"Architecture metadata mismatch: {path}")
    if record.get("vocabulary")!=list(vocabulary_for(version)):
        raise ValueError(f"Checkpoint vocabulary mismatch: {path}")
    return {
        "checkpoint":str(path),
        "dataset_version":version,
        "architecture":record["architecture"],
        "training_steps":int(record.get("training_steps",0)),
        "base_channels":int(cfg["channels"]),
        "diffusion_steps":int(cfg["diffusion_steps"]),
        "cfg_drop_prob":float(record.get("cfg_drop_prob",0.0)),
        "warm_start":record.get("warm_start"),
    }


def list_checkpoints(root):
    root=Path(root)
    records=[]
    for checkpoint in sorted(root.glob("pixel-*/checkpoint.pt")):
        try:
            records.append(read_metadata(checkpoint))
        except (ValueError,KeyError,RuntimeError,EOFError) as exc:
            print(f"SKIP invalid pixel checkpoint {checkpoint}: {exc}",file=sys.stderr)
    return records


def select_latest(records,version,minimum_steps=0):
    filtered=[x for x in records if
              x["dataset_version"]==version and
              x["training_steps"]>=minimum_steps]
    if not filtered:
        raise FileNotFoundError(f"No {version} checkpoint with >= {minimum_steps} steps")
    # Only paths are sorted; training steps do not automatically mean best image quality.
    return max(filtered,key=lambda x:x["checkpoint"])


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="action",required=True)
    for action in ("list","latest"):
        p=sub.add_parser(action)
        p.add_argument("--root",default="storage/artifacts/sora/nano/mvp")
        p.add_argument("--dataset-version",choices=("v1","v2"),
                       default="v1" if action=="latest" else None)
        p.add_argument("--min-steps",type=int,default=0)
    args=parser.parse_args()
    items=list_checkpoints(args.root)
    if args.action=="list":
        if args.dataset_version:
            items=[i for i in items if i["dataset_version"]==args.dataset_version]
        if args.min_steps:
            items=[i for i in items if i["training_steps"]>=args.min_steps]
        print(json.dumps(items,indent=2,ensure_ascii=False))
    else:
        print(select_latest(items,args.dataset_version,args.min_steps)["checkpoint"])


if __name__=="__main__":
    main()

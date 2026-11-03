"""Train / infer / evaluate an explicitly layout-conditioned learned neural field.

Run from models/sora/variants/nano/mvp:
    python3 -m scripts.spatial_field train --steps 1100 --device cpu
    python3 -m scripts.spatial_field evaluate --checkpoint PATH --output-dir DIR
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.model import CAPTIONS, caption_id, synthetic_image
from src.storage_paths import sora_run
from src.spatial_field import (
    SpatialConfig, SpatialField, synthetic_batch, spatial_loss, render, save_field_png
)


def restore(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 1
            or checkpoint.get("architecture") != "spatial-field-v1"
            or checkpoint.get("captions") != list(CAPTIONS)):
        raise ValueError("Not a supported spatial-field checkpoint")
    model = SpatialField(SpatialConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device)


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training options")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    cfg = SpatialConfig(hidden=args.hidden, embedding_dim=args.embedding_dim)
    model = SpatialField(cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run = sora_run("spatial",getattr(args,"output_root",None))
    run.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps + 1):
        # Balanced class IDs, with varied positions/sizes on each step.
        labels = torch.tensor([i % len(CAPTIONS) for i in range(args.batch_size)])
        images, masks, layouts = synthetic_batch(labels, step=step)
        labels, images, masks, layouts = (
            labels.to(args.device), images.to(args.device),
            masks.to(args.device), layouts.to(args.device)
        )
        logits, predicted = model(labels, layouts)
        loss, segmentation, color = spatial_loss(logits, predicted, images, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite spatial-field loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            print(json.dumps({
                "step": step, "loss": round(float(loss.detach()), 6),
                "mask_bce": round(float(segmentation.detach()), 6),
                "rgb_mae": round(float(color.detach()), 6)
            }), flush=True)
    checkpoint = run / "checkpoint.pt"
    torch.save({
        "schema_version": 1, "architecture": "spatial-field-v1",
        "family": "sora", "config": asdict(cfg), "captions": list(CAPTIONS),
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "training_steps": args.steps, "seed": args.seed,
        "training_distribution": "procedural shapes + explicit geometry conditioning",
        "quality_status": "constrained neural renderer; not text-to-image",
    }, checkpoint)
    print(json.dumps({"checkpoint": str(checkpoint)}, indent=2))
    return checkpoint


def evaluate(model, output_dir, seed=42, scale=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    grid = Image.new("RGB", (3 * 32 * scale, 2 * (32 * scale + 28)), "white")
    draw = ImageDraw.Draw(grid)
    result = []
    background = torch.tensor([246, 246, 244], dtype=torch.float32) / 255
    device = next(model.parameters()).device
    for i, caption in enumerate(CAPTIONS):
        rgb, layout, logits = render(model, i, seed=seed, device=device)
        name = caption.replace(" ", "_") + ".png"
        save_field_png(rgb, output_dir / name)
        tile = Image.open(output_dir / name).resize(
            (32 * scale, 32 * scale), Image.Resampling.NEAREST
        )
        column, row = i % 3, i // 3
        x, y = column * 32 * scale, row * (32 * scale + 28)
        grid.paste(tile, (x, y))
        draw.text((x + 8, y + 32 * scale + 5), caption, fill="black")

        # Only evaluation, not inference: independently render ground truth
        # for this held-out layout to measure whether the learned field fits.
        target = ((synthetic_image(i, seed + 1000*i) + 1) / 2).permute(1, 2, 0)
        truth_mask = ((target - background).abs().amax(dim=-1) > 0.03)
        predicted_mask = logits > 0
        overlap = (truth_mask & predicted_mask).sum().item()
        union = (truth_mask | predicted_mask).sum().item()
        iou = float(overlap / union) if union else 1.0
        mae = float((rgb - target).abs().mean())
        result.append({
            "caption": caption, "filename": name,
            "mask_iou": round(iou, 5),
            "rgb_mae": round(mae, 5),
            "layout": [round(float(v), 3) for v in layout],
        })
    grid.save(output_dir / "preview_grid.png")
    mean_iou = sum(row["mask_iou"] for row in result) / len(result)
    report = {
        "schema_version": 1,
        "architecture": "spatial-field-v1",
        "seed": seed, "mean_mask_iou": round(mean_iou, 5),
        "quality_context": "six synthetic classes with given position and size; not an unconditioned text-to-image model",
        "items": result,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({
        "preview_grid": str(output_dir / "preview_grid.png"),
        "report": str(output_dir / "report.json"),
        "mean_mask_iou": round(mean_iou, 5)
    }))
    return output_dir / "preview_grid.png"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("train")
    p.add_argument("--steps", type=int, default=1100)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--learning-rate", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--output-root", default=None,
                   help="Default: BERHOT/storage/artifacts/sora/nano/mvp")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p = sub.add_parser("sample")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.mode == "train":
        if args.cpu_threads < 1:
            parser.error("cpu-threads must be >= 1")
        train(args)
    elif args.mode == "evaluate":
        evaluate(restore(args.checkpoint, args.device), args.output_dir, seed=args.seed)
    else:
        model = restore(args.checkpoint, args.device)
        image, _, _ = render(model, caption_id(args.prompt), args.seed, args.device)
        save_field_png(image, args.output)
        print(json.dumps({"output": args.output, "prompt": args.prompt}))


if __name__ == "__main__":
    main()

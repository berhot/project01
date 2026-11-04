"""Train, sample and evaluate an experimental 32x32 shape conditional VAE.

Run from models/sora/variants/nano/mvp with python3 -m scripts.cvae ...
"""
import argparse
from dataclasses import asdict
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.model import CAPTIONS, caption_id
from src.storage_paths import sora_run
from src.cvae import CVAEConfig, ShapeCVAE, batch_from_synthetic, cvae_loss, generate, save_rgb_png


def restore(checkpoint_path, device="cpu"):
    record = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (record.get("schema_version") != 1
            or record.get("architecture") != "shape-cvae-v1"
            or record.get("captions") != list(CAPTIONS)):
        raise ValueError("Not a supported Sora Nano shape-CVAE checkpoint")
    config = CVAEConfig(**record["config"])
    model = ShapeCVAE(config)
    model.load_state_dict(record["model"], strict=True)
    return model.to(device)


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training settings")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = CVAEConfig(channels=args.channels, latent_dim=args.latent_dim, beta=args.beta)
    model = ShapeCVAE(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run = sora_run("cvae",getattr(args,"output_root",None))
    run.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps + 1):
        labels = torch.randint(0, len(CAPTIONS), (args.batch_size,), device="cpu")
        images = batch_from_synthetic(labels, step).to(args.device)
        labels = labels.to(args.device)
        predicted, mean, logvar = model(images, labels)
        loss, reconstruction, divergence = cvae_loss(predicted, images, mean, logvar, config.beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CVAE loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            print(json.dumps({
                "step": step, "loss": round(float(loss.detach()), 6),
                "reconstruction": round(float(reconstruction.detach()), 6),
                "kl": round(float(divergence.detach()), 6)
            }), flush=True)
    checkpoint = run / "checkpoint.pt"
    torch.save({
        "schema_version": 1, "family": "sora", "architecture": "shape-cvae-v1",
        "config": asdict(config), "captions": list(CAPTIONS),
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "seed": args.seed, "training_steps": args.steps,
        "training_data": "procedural shapes only",
        "quality_status": "toy-shapes-only; not production",
    }, checkpoint)
    print(json.dumps({"checkpoint": str(checkpoint)}))
    return checkpoint


def evaluate(model, output_dir, seed=42, scale=8):
    """Save six genuine model outputs; never draw shapes during inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (32*scale*3, (32*scale+28)*2), "white")
    draw = ImageDraw.Draw(sheet)
    report = []
    for i, caption in enumerate(CAPTIONS):
        rgb = generate(model, i, seed=seed)
        outfile = output_dir / (caption.replace(" ", "_") + ".png")
        save_rgb_png(rgb, outfile)
        thumbnail = Image.open(outfile).resize((32*scale, 32*scale), Image.Resampling.NEAREST)
        col, row = i % 3, i // 3
        ox, oy = col*32*scale, row*(32*scale+28)
        sheet.paste(thumbnail, (ox, oy))
        draw.text((ox+8, oy+32*scale+6), caption, fill="black")
        report.append({"caption": caption, "image": outfile.name, "seed": seed})
    sheet.save(output_dir / "preview_grid.png")
    (output_dir / "report.json").write_text(json.dumps({
        "schema_version": 1, "architecture": "shape-cvae-v1",
        "training_distribution": "synthetic six-class shapes",
        "visual_quality": "requires human inspection; not automatically certified",
        "samples": report,
    }, indent=2) + "\n", encoding="utf8")
    return output_dir / "preview_grid.png"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("train")
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--channels", type=int, default=32)
    training.add_argument("--latent-dim", type=int, default=8)
    training.add_argument("--beta", type=float, default=0.05)
    training.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    training.add_argument("--output-root", default=None,
                          help="Default: BERHOT/storage/artifacts/sora/nano/mvp")
    sampling = sub.add_parser("sample")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--prompt", required=True)
    sampling.add_argument("--output", default=None)
    sampling.add_argument("--seed", type=int, default=42)
    sampling.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output-dir", default=None)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.action == "train":
        train(args)
    elif args.action == "sample":
        model = restore(args.checkpoint, args.device)
        output=Path(args.output) if args.output else sora_run("cvae-sample")/"sample.png"
        output.parent.mkdir(parents=True,exist_ok=True)
        save_rgb_png(generate(model, caption_id(args.prompt), args.seed, args.device), output)
        print(json.dumps({"output": str(output), "prompt": args.prompt}))
    else:
        model = restore(args.checkpoint, args.device)
        print(json.dumps({"preview_grid": str(evaluate(model, args.output_dir or sora_run("cvae-eval"), args.seed))}))


if __name__ == "__main__":
    main()

"""Train a learned 3D layout DDPM and pair it with a trained spatial renderer.

No external geometry is supplied during inference. Not a pixel diffusion model,
a free-form text generator, or a substitute for real image-design research.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import uuid

from PIL import Image, ImageDraw
import torch

from src.layout_diffusion import LayoutConfig, LayoutDenoiser, sample_layout, training_loss
from src.spatial_field import SpatialConfig, SpatialField
from src.model import CAPTIONS
from src.storage_paths import sora_run


def restore_prior(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record.get("schema_version") != 1
            or record.get("architecture") != "layout-ddpm-v1"
            or record.get("captions") != list(CAPTIONS)):
        raise ValueError("Invalid Sora learned-layout checkpoint")
    model = LayoutDenoiser(LayoutConfig(**record["config"]))
    model.load_state_dict(record["model"], strict=True)
    return model.to(device)


def restore_renderer(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record.get("schema_version") != 1
            or record.get("architecture") != "spatial-field-v1"
            or record.get("captions") != list(CAPTIONS)):
        raise ValueError("Requires a trained spatial-field-v1 checkpoint")
    model = SpatialField(SpatialConfig(**record["config"]))
    model.load_state_dict(record["model"], strict=True)
    return model.to(device)


def training_layouts(batch_size, rng):
    """Synthetic dataset generation only; never used in model inference."""
    coords = []
    for _ in range(batch_size):
        width = rng.randint(12, 21)
        left = rng.randint(3, 32 - width - 3)
        top = rng.randint(3, 32 - width - 3)
        coords.append((left + width / 2, top + width / 2, width / 2))
    return torch.tensor(coords, dtype=torch.float32)


def train(args):
    if (args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.cpu_threads < 1):
        raise ValueError("Invalid training settings")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = LayoutConfig(steps=args.diffusion_steps, hidden=args.hidden)
    model = LayoutDenoiser(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output = sora_run("layout",getattr(args,"output_root",None))
    output.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps + 1):
        labels = torch.randint(0, len(CAPTIONS), (args.batch_size,), device=args.device)
        layouts = training_layouts(args.batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, labels, layouts)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite layout diffusion loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            print(json.dumps({
                "step": step, "noise_mse": round(float(loss.detach()), 6)
            }), flush=True)
    checkpoint = output / "checkpoint.pt"
    torch.save({
        "schema_version": 1, "architecture": "layout-ddpm-v1",
        "config": asdict(cfg), "captions": list(CAPTIONS),
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "training_steps": args.steps, "seed": args.seed,
        "training_source": "synthetic 32x32 shape layout distribution",
        "quality_status": "constrained learned layout, not pixel diffusion",
    }, checkpoint)
    print(json.dumps({"checkpoint": str(checkpoint)}))
    return checkpoint


@torch.no_grad()
def evaluate(prior, renderer, output_dir, seed=42, scale=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = torch.arange(len(CAPTIONS), dtype=torch.long)
    layouts = sample_layout(prior, labels, seed=seed)
    device = next(renderer.parameters()).device
    _, rgb = renderer(labels.to(device), layouts.to(device))
    rgb = rgb.cpu()
    sheet = Image.new("RGB", (32*scale*3, (32*scale+28)*2), "white")
    draw = ImageDraw.Draw(sheet)
    entries = []
    for i, caption in enumerate(CAPTIONS):
        filename = caption.replace(" ", "_") + ".png"
        image = Image.fromarray(
            (rgb[i].clamp(0, 1).numpy() * 255).round().astype("uint8")
        )
        image.save(output_dir / filename)
        col, row = i % 3, i // 3
        sheet.paste(image.resize(
            (32*scale, 32*scale), Image.Resampling.NEAREST
        ), (col*32*scale, row*(32*scale+28)))
        draw.text((col*32*scale + 6, row*(32*scale+28) + 32*scale + 5),
                  caption, fill="black")
        entries.append({
            "caption": caption, "image": filename,
            "generated_center_and_halfwidth": [
                round(float(v), 4) for v in layouts[i]
            ]
        })
    sheet.save(output_dir / "preview_grid.png")
    (output_dir / "report.json").write_text(json.dumps({
        "schema_version": 1, "task": "learned-layout-plus-spatial-renderer",
        "seed": seed, "layout_source": "trained DDPM",
        "pixel_source": "trained spatial field",
        "visual_quality": "requires human inspection",
        "limitations": "six labels, one shape, bounded geometry",
        "items": entries,
    }, indent=2) + "\n", encoding="utf8")
    return output_dir / "preview_grid.png"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--learning-rate", type=float, default=0.002)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--diffusion-steps", type=int, default=64)
    training.add_argument("--hidden", type=int, default=64)
    training.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    training.add_argument("--cpu-threads", type=int, default=2)
    training.add_argument("--output-root", default=None,
                          help="Default: BERHOT/storage/artifacts/sora/nano/mvp")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--layout-checkpoint", required=True)
    evaluation.add_argument("--spatial-checkpoint", required=True)
    evaluation.add_argument("--output-dir", default=None)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        prior = restore_prior(args.layout_checkpoint, args.device)
        renderer = restore_renderer(args.spatial_checkpoint, args.device)
        print(json.dumps({
            "preview_grid": str(evaluate(prior, renderer, args.output_dir or sora_run("layout-eval"), args.seed))
        }))


if __name__ == "__main__":
    main()

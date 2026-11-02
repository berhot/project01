"""Train from scratch on procedurally generated image/caption pairs."""
import argparse
import json
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.model import Config, Denoiser, CAPTIONS, synthetic_image, training_loss, sample, save_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-root", default="storage/artifacts/image/nano/mvp")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        p.error("Invalid training settings")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = Config(channels=args.channels, diffusion_steps=args.diffusion_steps)
    device = torch.device(args.device)
    model = Denoiser(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    run = Path(args.output_root) / ("image-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps + 1):
        ids = [(step * args.batch_size + j) % len(CAPTIONS) for j in range(args.batch_size)]
        images = torch.stack([synthetic_image(label, args.seed + step * 10000 + j) for j, label in enumerate(ids)]).to(device)
        labels = torch.tensor(ids, device=device)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = training_loss(model, images, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite diffusion loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
    checkpoint = {
        "schema_version": 1, "family": "image", "task": "caption-conditioned-ddpm-demo",
        "config": asdict(config), "captions": list(CAPTIONS),
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "seed": args.seed, "steps": args.steps,
        "training_data": "deterministic synthetic shapes only",
        "quality_status": "engineering-mvp-not-production",
    }
    torch.save(checkpoint, run / "checkpoint.pt")
    save_png(sample(model, 0, seed=args.seed, device=args.device), run / "sample_red_circle.png")
    print(json.dumps({"checkpoint": str(run / "checkpoint.pt"), "sample": str(run / "sample_red_circle.png")}, indent=2))


if __name__ == "__main__":
    main()

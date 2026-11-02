"""Use locally trained weights; no remote text-to-image service."""
import argparse
import json
import torch

from src.model import CAPTIONS, Config, Denoiser, caption_id, sample, save_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("task") != "caption-conditioned-ddpm-demo" or checkpoint.get("captions") != list(CAPTIONS):
        raise ValueError("Unsupported image checkpoint")
    if checkpoint["config"].get("schedule_version") != "cosine_v1":
        raise ValueError("Legacy checkpoint uses an incompatible diffusion schedule. Retrain with updated code.")
    model = Denoiser(Config(**checkpoint["config"])).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    save_png(sample(model, caption_id(args.prompt), seed=args.seed, device=args.device), args.output)
    print(json.dumps({"output": args.output, "prompt": args.prompt, "seed": args.seed}))


if __name__ == "__main__":
    main()

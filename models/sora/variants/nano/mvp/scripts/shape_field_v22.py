"""Opt-in, immutable Sora Nano Shape Field V2.2 geometry experiment.

Uses an unchanged pretrained V2.1 model and colorizer. Existing V2/V2.1
trainer, checkpoints, release criteria, and inference remain untouched.
A new run is always written to a fresh shape-field-v22-* directory.
"""
import argparse
from dataclasses import asdict
import json
import math
import random

import numpy as np
import torch

from scripts.shape_field_v2 import (
    MASK_KIND, COLOR_KIND, checksum, load_run, _save_weight, evaluate
)
from src.pixel_dataset_v2 import balanced_training_batch
from src.shape_first import mask_from_rgb
from src.shape_field_v2 import (
    NeuralMaskField, NeuralColorizer, layout_from_mask, geometry_objective
)
from src.shape_field_v21 import (
    matched_layout_shape_batch, shape_word_separation_objective
)
from src.shape_field_v22 import boundary_centroid_objective
from src.storage_paths import sora_run


def train(args):
    if (args.steps < 1 or args.batch_size < 12 or args.batch_size % 12
            or not 1 <= args.paired_layout_count <= 32
            or args.log_every < 1 or args.cpu_threads < 1
            or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0
            or not math.isfinite(args.circle_preserve_weight)
            or not 0 <= args.circle_preserve_weight <= 5):
        raise ValueError("Invalid V2.2 training configuration")
    # Validate auxiliary-loss weights before creating any artifacts.
    if any(not math.isfinite(x) or not 0 <= x <= 5 for x in
           (args.boundary_weight, args.centroid_weight)):
        raise ValueError("Invalid auxiliary geometry loss weights")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    source, source_color, manifest = load_run(args.init_from_v21, args.device)
    if manifest.get("training_recipe") != "shape-word-matched-layout-v2.1":
        raise ValueError("V2.2 requires an immutable V2.1 source checkpoint")
    if args.boundary_weight == 0 and args.centroid_weight == 0:
        raise ValueError("At least one V2.2 geometry objective must be enabled")
    config = source.config
    model = NeuralMaskField(config).to(args.device)
    model.load_state_dict(source.state_dict(), strict=True)
    source.requires_grad_(False)
    source.eval()
    color = NeuralColorizer(config).to(args.device)
    color.load_state_dict(source_color.state_dict(), strict=True)
    color.eval()
    color.requires_grad_(False)
    root = sora_run("shape-field-v22", args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    try:
        for step in range(1, args.steps + 1):
            images, tokens = balanced_training_batch(step, args.batch_size,
                                                      args.seed)
            images = images.to(args.device)
            tokens = tokens.to(args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            base = geometry_objective(model, images, tokens)
            truth = mask_from_rgb(images)
            layout = layout_from_mask(truth)
            logits = model(layout, tokens)
            auxiliary = boundary_centroid_objective(
                logits, truth, tokens, args.boundary_weight,
                args.centroid_weight)
            pair_layouts, pair_targets, pair_tokens, _ = matched_layout_shape_batch(
                step, args.paired_layout_count, args.seed, args.device)
            paired = shape_word_separation_objective(
                model, pair_layouts, pair_targets, pair_tokens)
            paired_aux = boundary_centroid_objective(
                paired["correct_logits"], pair_targets, pair_tokens,
                args.boundary_weight, args.centroid_weight)
            # Frozen V2.1 teacher: maintain circle predictions on the
            # same layouts rather than modifying inference with templates.
            vocab_tokens = tokens
            from src.shape_field_v2 import keyword_ids, SHAPE_WORDS, draw_layout_from_noise
            from src.pixel_diffusion import vocabulary_for
            vocabulary = vocabulary_for("v2")
            selected = keyword_ids(vocab_tokens, SHAPE_WORDS)
            circles = ((selected == vocabulary.index("circle")) |
                       (selected == vocabulary.index("دائرة")))
            circle_layout = torch.cat((
                layout[circles],
                draw_layout_from_noise(int(circles.sum()),
                                       args.seed + step*17, args.device)
            ), dim=0)
            circle_tokens = torch.cat((tokens[circles], tokens[circles]), 0)
            with torch.no_grad():
                circle_teacher = source(circle_layout, circle_tokens).sigmoid()
            circle_prediction = model(circle_layout, circle_tokens).sigmoid()
            circle_penalty = torch.nn.functional.mse_loss(
                circle_prediction, circle_teacher)
            total = (base["loss"] + paired["loss"] + auxiliary["loss"] +
                     paired_aux["loss"] +
                     args.circle_preserve_weight*circle_penalty)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite V2.2 total geometry loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps({
                    "stage": "shape-field-v22", "step": step,
                    "total_loss": round(float(total.detach()), 6),
                    "edge_loss": round(float(auxiliary["edge_loss"]), 6),
                    "paired_edge_loss": round(float(paired_aux["edge_loss"]), 6),
                    "centroid_loss": round(float(auxiliary["centroid_loss"]), 6),
                    "circle_teacher_mse": round(float(circle_penalty.detach()), 7),
                }), flush=True)
        _save_weight(root/"mask_field.pt", model, MASK_KIND, args.steps, args.seed)
        # Save a distinct, byte-equivalent colorizer checkpoint without
        # training it, so existing load_run checksum checks remain valid.
        import shutil
        shutil.copyfile(
            __import__("pathlib").Path(args.init_from_v21)/"colorizer.pt",
            root/"colorizer.pt")
        record = {
            "schema_version": 1, "family": "sora", "dataset_version": "v2",
            "model_family": "shape-field-v2",
            "config": asdict(config),
            "mask_architecture": MASK_KIND,
            "color_architecture": COLOR_KIND,
            "mask_steps": args.steps, "color_steps": 0,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "seed": args.seed, "training_recipe": "shape-boundary-centroid-v2.2",
            "warm_start": {
                "source_run_dir": str(args.init_from_v21),
                "mask_source_sha256": manifest["sha256"]["mask_field.pt"],
                "color_source_sha256": manifest["sha256"]["colorizer.pt"],
            },
            "v22_objective": {
                "boundary_weight": args.boundary_weight,
                "centroid_weight": args.centroid_weight,
                "circle_preserve_weight": args.circle_preserve_weight,
                "paired_layout_count": args.paired_layout_count,
            },
            "sha256": {
                "mask_field.pt": checksum(root/"mask_field.pt"),
                "colorizer.pt": checksum(root/"colorizer.pt"),
            },
            "quality_status": "experimental",
            "release_status": "not-approved",
            "colorizer_reused_untrained": True,
            "original_rgb_ddpm_unchanged": True,
            "shape_first_v1_unchanged": True,
        }
        with (root/"run.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except Exception:
        print(json.dumps({"incomplete_experimental_run": str(root)}), flush=True)
        raise
    print(json.dumps({
        "run_dir": str(root), "release_status": "not-approved",
        "mask_checkpoint": str(root/"mask_field.pt"),
        "color_checkpoint": str(root/"colorizer.pt"),
    }, indent=2), flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from-v21", required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--paired-layout-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--centroid-weight", type=float, default=0.05)
    parser.add_argument("--circle-preserve-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

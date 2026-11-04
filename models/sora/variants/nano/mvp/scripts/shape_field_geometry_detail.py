"""Read-only, per-sample geometry evidence for Shape Field V2/V2.1.

Does not alter model weights, threshold, training recipe, or release state.
The supplied truth mask is used only to assess heldout predictions, never to
construct or postprocess a sampled inference mask.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.shape_field_v2 import load_run
from src.pixel_dataset_v2 import COLORS, SHAPES, caption_v2, synthetic_example_v2
from src.pixel_diffusion import tokenize
from src.shape_field_v2 import draw_layout_from_noise, layout_from_mask
from src.shape_first import mask_from_rgb
from src.storage_paths import sora_run


def _array(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.shape[-2:] != (32, 32) or mask.size != 1024:
        raise ValueError("Expected exactly one 32x32 mask")
    mask = mask.reshape(32, 32)
    if not np.isfinite(mask).all():
        raise ValueError("Nonfinite mask")
    return mask


def _boundary(mask):
    # Four-neighbor boundary; excludes interior without relying on SciPy.
    inside = mask.astype(bool)
    interior = inside.copy()
    interior[1:, :] &= inside[:-1, :]
    interior[:-1, :] &= inside[1:, :]
    interior[:, 1:] &= inside[:, :-1]
    interior[:, :-1] &= inside[:, 1:]
    return inside & ~interior


def _overlap(a, b):
    union = int(np.logical_or(a, b).sum())
    return round(float(np.logical_and(a, b).sum()) / union, 6) if union else 1.0


def geometry_stats(mask):
    foreground = _array(mask) > 0.5
    yy, xx = np.where(foreground)
    if not len(xx):
        return {"foreground_pixels": 0, "bbox": None, "centroid": None,
                "boundary_pixels": 0}
    return {
        "foreground_pixels": int(len(xx)),
        "bbox": {"left": int(xx.min()), "top": int(yy.min()),
                 "right": int(xx.max()), "bottom": int(yy.max()),
                 "width": int(xx.max() - xx.min() + 1),
                 "height": int(yy.max() - yy.min() + 1)},
        "centroid": {"x": round(float(xx.mean()), 6),
                     "y": round(float(yy.mean()), 6)},
        "boundary_pixels": int(_boundary(foreground).sum()),
    }


def compare_masks(predicted, reference):
    pred = _array(predicted) > 0.5
    truth = _array(reference) > 0.5
    p, t = geometry_stats(pred), geometry_stats(truth)
    dx = None if p["centroid"] is None or t["centroid"] is None else round(
        p["centroid"]["x"] - t["centroid"]["x"], 6)
    dy = None if p["centroid"] is None or t["centroid"] is None else round(
        p["centroid"]["y"] - t["centroid"]["y"], 6)
    return {
        "mask_iou": _overlap(pred, truth),
        "boundary_iou": _overlap(_boundary(pred), _boundary(truth)),
        "foreground_pixel_error": p["foreground_pixels"] - t["foreground_pixels"],
        "area_error_fraction": round(
            (p["foreground_pixels"] - t["foreground_pixels"]) /
            t["foreground_pixels"], 6) if t["foreground_pixels"] else None,
        "centroid_dx": dx, "centroid_dy": dy,
        "prediction": p, "reference": t,
    }


def _tile(mask):
    return np.uint8((_array(mask) > 0.5) * 255)


def save_comparison_grid(path, truth, heldout, sampled):
    """Truth/heldout/sampled/heldout errors; sampled is NOT truth-aligned."""
    truth, heldout, sampled = map(_tile, (truth, heldout, sampled))
    # Red false positives, blue false negatives, black correct background,
    # white true positives.
    overlap = np.zeros((32, 32, 3), dtype=np.uint8)
    overlap[(truth > 0) & (heldout > 0)] = (255, 255, 255)
    overlap[(truth == 0) & (heldout > 0)] = (255, 0, 0)
    overlap[(truth > 0) & (heldout == 0)] = (0, 90, 255)
    panels = [np.repeat(x[:, :, None], 3, axis=2)
              for x in (truth, heldout, sampled)] + [overlap]
    image = Image.fromarray(np.concatenate(panels, axis=1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((1024, 256), Image.Resampling.NEAREST).save(path)


@torch.no_grad()
def diagnose_details(run_dir, seed=42, device="cpu", output_dir=None):
    model, _, manifest = load_run(run_dir, device=device)
    model.eval()
    root = Path(output_dir) if output_dir else sora_run("shape-field-geometry-detail")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape_index, shape in enumerate(SHAPES):
        for color_index, color in enumerate(COLORS):
            for language_index, language in enumerate(("en", "ar")):
                validation_seed = (seed + shape_index*1709 +
                                   color_index*311 + language_index*83)
                prompt = caption_v2(color, shape, language, 0)
                tokens = tokenize(prompt, dataset_version="v2")[None].to(device)
                example, _, _, _ = synthetic_example_v2(
                    validation_seed, color=color, shape=shape,
                    language=language, split="validation", variant=0)
                truth = mask_from_rgb(example[None].to(device))
                heldout = model(layout_from_mask(truth), tokens).sigmoid()
                sampled_layout = draw_layout_from_noise(
                    1, seed + color_index, device=device)
                sampled = model(sampled_layout, tokens).sigmoid()
                # Sampled layout has no ground-truth pair. Comparing it with
                # validation truth directly would be scientifically invalid.
                sampled_statistics = geometry_stats(sampled >= 0.5)
                file = root / "previews" / f"{shape}-{color}-{language}.png"
                save_comparison_grid(file, truth, heldout, sampled)
                rows.append({
                    "shape": shape, "color": color, "language": language,
                    "validation_seed": validation_seed, "prompt": prompt,
                    "heldout_vs_truth": compare_masks(heldout >= 0.5, truth),
                    "sampled": sampled_statistics,
                    "sampled_minus_heldout_pixels": (
                        sampled_statistics["foreground_pixels"] -
                        geometry_stats(heldout >= 0.5)["foreground_pixels"]),
                    "preview": str(file),
                })
    report = {
        "schema_version": 1,
        "diagnostic": "shape-field-per-sample-geometry-detail",
        "source_run_dir": str(run_dir),
        "mask_checksum_sha256": manifest["sha256"]["mask_field.pt"],
        "seed": seed,
        "threshold": 0.5,
        "sampled_has_matching_truth": False,
        "no_training": True,
        "release_approved": False,
        "items": rows,
    }
    report_path = root / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report_path, report = diagnose_details(
        args.run_dir, args.seed, args.device, args.output_dir)
    print(json.dumps({"report": str(report_path), "items": len(report["items"]),
                      "no_training": True, "release_approved": False}, indent=2))


if __name__ == "__main__":
    main()

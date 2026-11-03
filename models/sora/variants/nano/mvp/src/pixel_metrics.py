"""Transparent toy-domain image scoring for colored 32x32 single-shape outputs.

Rules are deliberately narrow: near-white background, one colored object.
These heuristics are not a general image understanding or aesthetic metric.
Failures/undetected shapes count as errors, never as successful predictions.
"""
from collections import defaultdict
import math

import torch

COLORS = ("red", "green", "blue")
SHAPES = ("circle", "square")


def analyze_image(image, dataset_version="v1"):
    """Return an interpretable color/shape estimate, or 'unknown'."""
    if dataset_version not in ("v1", "v2"):
        raise ValueError("Unsupported metric dataset")
    if not isinstance(image, torch.Tensor) or image.shape != (3, 32, 32):
        raise ValueError("Expected [3,32,32] tensor")
    if not torch.isfinite(image).all():
        raise ValueError("Nonfinite image")
    # Model images use [-1,1]. Ground truth has white (246,246,244).
    rgb = ((image.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0)
    background = torch.tensor((246, 246, 244), dtype=rgb.dtype) / 255
    mask = (rgb - background).abs().amax(dim=-1) > 0.18
    ys, xs = torch.where(mask)
    count = int(len(xs))
    result = {"color": "unknown", "shape": "unknown",
              "foreground_pixels": count, "fill_ratio": 0.0}
    if count < 20:
        return result
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    width, height = right-left+1, bottom-top+1
    if min(width, height) < 7:
        return result
    fill = count / (width * height)
    result["fill_ratio"] = round(fill, 5)
    mean = rgb[mask].mean(dim=0)
    ordering = torch.argsort(mean, descending=True)
    dominant, second = int(ordering[0]), int(ordering[1])
    if dataset_version == "v2":
        r, g, b = [float(v) for v in mean]
        if r > b + 0.2 and g > b + 0.2 and abs(r-g) < 0.35:
            result["color"] = "yellow"
        elif float(mean[dominant] - mean[second]) > 0.12:
            result["color"] = ("red", "green", "blue")[dominant]
        if 0.60 < width/height < 1.5 and fill >= 0.38:
            if fill >= 0.88:
                result["shape"] = "square"
            elif fill >= 0.68:
                result["shape"] = "circle"
            else:
                result["shape"] = "triangle"
    else:
        if float(mean[dominant] - mean[second]) > 0.12:
            result["color"] = ("red", "green", "blue")[dominant]
        if 0.7 < width/height < 1.3 and fill >= 0.66:
            result["shape"] = "square" if fill >= 0.88 else "circle"
    return result


def summarize(predictions, dataset_version="v1"):
    """Summarize per-image correctness and paired Arabic/English consistency."""
    if dataset_version not in ("v1", "v2"):
        raise ValueError("Unsupported metric dataset")
    if not predictions:
        raise ValueError("Need at least one prediction")
    pairs = defaultdict(dict)
    color_correct = shape_correct = both_correct = 0
    for row in predictions:
        expected_colors = COLORS if dataset_version == "v1" else COLORS + ("yellow",)
        expected_shapes = SHAPES if dataset_version == "v1" else SHAPES + ("triangle",)
        if row["expected_color"] not in expected_colors or row["expected_shape"] not in expected_shapes:
            raise ValueError("Invalid expected toy concept")
        if row["language"] not in ("ar", "en"):
            raise ValueError("Language must be ar or en")
        color_ok = row["predicted_color"] == row["expected_color"]
        shape_ok = row["predicted_shape"] == row["expected_shape"]
        color_correct += int(color_ok)
        shape_correct += int(shape_ok)
        both_correct += int(color_ok and shape_ok)
        key = (row["expected_color"], row["expected_shape"], row["seed"])
        if row["language"] in pairs[key]:
            raise ValueError("Duplicate language for same concept/seed")
        pairs[key][row["language"]] = row
    n = len(predictions)
    paired = [p for p in pairs.values() if "en" in p and "ar" in p]
    consistent = sum(
        p["en"]["predicted_color"] == p["ar"]["predicted_color"]
        and p["en"]["predicted_shape"] == p["ar"]["predicted_shape"]
        and p["en"]["predicted_color"] != "unknown"
        and p["en"]["predicted_shape"] != "unknown"
        for p in paired
    )
    both_correct_pairs = sum(
        all(p[language]["predicted_color"] == p[language]["expected_color"]
            and p[language]["predicted_shape"] == p[language]["expected_shape"]
            for language in ("en", "ar"))
        for p in paired
    )
    return {
        "total_images": n,
        "color_accuracy": round(color_correct/n, 4),
        "shape_accuracy": round(shape_correct/n, 4),
        "prompt_accuracy": round(both_correct/n, 4),
        "bilingual_pairs": len(paired),
        "bilingual_prediction_consistency": round(consistent/len(paired), 4) if paired else None,
        "bilingual_both_correct": round(both_correct_pairs/len(paired), 4) if paired else None,
        "metric_scope": "toy single-shape/color heuristics; not human visual validation",
        "dataset_version": dataset_version,
    }

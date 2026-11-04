"""Read-only summary of Shape Field per-sample geometry evidence.

Pass the report produced by scripts.shape_field_geometry_detail. The prior
sampled layouts have no paired ground truth: differences are descriptive
distribution shifts only, not sampled IoU or causal proof of training issues.
"""
import argparse
import json
from pathlib import Path

SHAPES = ("circle", "square", "triangle")


def _mean(values):
    return round(sum(values) / len(values), 6)


def review(report):
    if (report.get("diagnostic") != "shape-field-per-sample-geometry-detail"
            or report.get("no_training") is not True
            or report.get("release_approved") is not False
            or report.get("sampled_has_matching_truth") is not False):
        raise ValueError("Not a valid read-only shape-field geometry detail report")
    items = report.get("items")
    if not isinstance(items, list) or len(items) != 24:
        raise ValueError("Expected exactly 24 geometry records")
    summary = {}
    for shape in SHAPES:
        rows = [item for item in items if item.get("shape") == shape]
        if len(rows) != 8 or len({(r.get("color"), r.get("language")) for r in rows}) != 8:
            raise ValueError(f"Expected eight unique bilingual color combinations for {shape}")
        held = [r["heldout_vs_truth"] for r in rows]
        truth_pixels = [r["reference"]["foreground_pixels"] for r in held]
        predicted_pixels = [r["prediction"]["foreground_pixels"] for r in held]
        sample_pixels = [r["sampled"]["foreground_pixels"] for r in rows]
        if any(x <= 0 for x in truth_pixels):
            raise ValueError("Unexpected empty validation masks")
        summary[shape] = {
            "samples": 8,
            "heldout_iou_mean": _mean([x["mask_iou"] for x in held]),
            "heldout_boundary_iou_mean": _mean([x["boundary_iou"] for x in held]),
            "heldout_truth_pixels_mean": _mean(truth_pixels),
            "heldout_pred_pixels_mean": _mean(predicted_pixels),
            "sampled_pixels_mean": _mean(sample_pixels),
            "heldout_area_error_fraction_mean": _mean(
                [x["area_error_fraction"] for x in held]),
            "heldout_abs_centroid_dx_mean": _mean(
                [abs(x["centroid_dx"]) for x in held if x["centroid_dx"] is not None])
                if all(x["centroid_dx"] is not None for x in held) else None,
            "heldout_abs_centroid_dy_mean": _mean(
                [abs(x["centroid_dy"]) for x in held if x["centroid_dy"] is not None])
                if all(x["centroid_dy"] is not None for x in held) else None,
            "sampled_to_heldout_area_ratio": round(
                sum(sample_pixels) / sum(predicted_pixels), 6)
                if sum(predicted_pixels) else None,
            "heldout_bbox_width_error_mean": _mean([
                x["prediction"]["bbox"]["width"] - x["reference"]["bbox"]["width"]
                for x in held if x["prediction"]["bbox"] and x["reference"]["bbox"]
            ]) if all(x["prediction"]["bbox"] and x["reference"]["bbox"] for x in held)
                else None,
            "heldout_bbox_height_error_mean": _mean([
                x["prediction"]["bbox"]["height"] - x["reference"]["bbox"]["height"]
                for x in held if x["prediction"]["bbox"] and x["reference"]["bbox"]
            ]) if all(x["prediction"]["bbox"] and x["reference"]["bbox"] for x in held)
                else None,
        }
    return {
        "source_run_dir": report["source_run_dir"],
        "seed": report["seed"],
        "by_shape": summary,
        "interpretation": (
            "Low heldout IoU/boundary IoU indicates inadequate geometry even with "
            "observed target layout. A sampled-to-heldout area ratio far from 1 "
            "suggests distribution sensitivity but cannot establish causal "
            "mechanism without matched target-layout experiments. The sampled "
            "layout has no paired ground truth."
        ),
        "no_training": True,
        "release_approved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    print(json.dumps(review(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

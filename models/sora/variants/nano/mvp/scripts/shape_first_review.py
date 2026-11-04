"""Compare classic RGB-DDPM with learned shape-first Sora using actual PNGs.

Cross-architecture seed values cannot be interpreted as identical stochastic
noise. Comparison is by matched prompt concept and language, fixed sample
protocol, and the same independent strict-geometry scorer. This is a
research gate only and cannot issue an approved model release.
"""
import argparse
import json
from pathlib import Path

from scripts.pixel_shape_quality import audit_gallery
from src.storage_paths import sora_run


def compare_architectures(baseline,shape_first,min_triangle_passes=2):
    if (baseline["sample_count"]!=24 or shape_first["sample_count"]!=24
            or not 1<=min_triangle_passes<=8):
        raise ValueError("Need 24 prompts and a valid minimum-triangle requirement")
    expected={(item["expected_color"],item["expected_shape"],item["language"])
              for item in baseline["items"]}
    actual={(item["expected_color"],item["expected_shape"],item["language"])
            for item in shape_first["items"]}
    if len(expected)!=24 or expected!=actual:
        raise ValueError("Baseline and new model must cover the exact same 24 prompts")
    before,after=baseline["by_shape"],shape_first["by_shape"]
    checks={
        "circle_geometry_not_regressed":
            after["circle"]["geometry_pass_count"]>=before["circle"]["geometry_pass_count"],
        "square_geometry_not_regressed":
            after["square"]["geometry_pass_count"]>=before["square"]["geometry_pass_count"],
        "triangle_geometry_visible":
            after["triangle"]["geometry_pass_count"]>=min_triangle_passes
            and after["triangle"]["geometry_pass_count"]>before["triangle"]["geometry_pass_count"],
        "color_not_regressed":all(
            after[shape]["color_accuracy"]>=before[shape]["color_accuracy"]
            for shape in ("circle","square","triangle")
        ),
        "overall_strict_prompt_improved":
            shape_first["strict_prompt_accuracy"]>baseline["strict_prompt_accuracy"],
    }
    return {
        "research_gate_pass":all(checks.values()),
        "release_approved":False,
        "human_visual_review_required":True,
        "checks":checks,
        "baseline_shape_quality":before,
        "new_shape_first_quality":after,
        "baseline_strict_prompt_accuracy":baseline["strict_prompt_accuracy"],
        "candidate_strict_prompt_accuracy":shape_first["strict_prompt_accuracy"],
        "warning":"Architecture differs. Same concepts/languages, but Gaussian latent and DDPM seeds are NOT equivalent. This is comparative screening, not paired-noise inference or release approval.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir",required=True,
                        help="Original RGB DDPM guidance_5_seed_42 folder")
    parser.add_argument("--candidate-dir",required=True,
                        help="New shape-first 24-PNG evaluation folder")
    parser.add_argument("--min-triangle-passes",type=int,default=2)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    baseline=audit_gallery(args.baseline_dir)
    candidate=audit_gallery(args.candidate_dir)
    report=compare_architectures(
        baseline,candidate,min_triangle_passes=args.min_triangle_passes
    )
    report.update({
        "schema_version":1,
        "baseline_dir":str(args.baseline_dir),
        "candidate_dir":str(args.candidate_dir),
    })
    output=Path(args.output) if args.output else sora_run("shape-first-review")/"report.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as file:
        json.dump(report,file,indent=2,ensure_ascii=False)
        file.write("\n")
    print(json.dumps({
        "review_report":str(output),
        "research_gate_pass":report["research_gate_pass"],
        "release_approved":False,
        "checks":report["checks"],
        "baseline_strict_prompt_accuracy":report["baseline_strict_prompt_accuracy"],
        "candidate_strict_prompt_accuracy":report["candidate_strict_prompt_accuracy"],
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

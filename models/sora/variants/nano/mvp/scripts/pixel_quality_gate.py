"""Conservative no-training model candidate gate from saved Sora RGB PNG reports.

This is an automated TOY-domain heuristic gate, not human image review.
Requires identical v2 prompts, seed policy, sampler and CFG in the old/new
runs; it will not silently compare incomparable evaluations.
"""
import argparse
import json
from pathlib import Path

from scripts.pixel_diagnose import audit_evaluation


def compare_reports(baseline_report,candidate_report,
                    triangle_min=0.125,old_shape_drop=0.0,
                    color_drop=0.0417):
    original=Path(baseline_report)
    candidate=Path(candidate_report)
    before=json.loads(original.read_text(encoding="utf-8"))
    after=json.loads(candidate.read_text(encoding="utf-8"))
    for label,doc in (("baseline",before),("candidate",after)):
        if doc.get("dataset_version")!="v2":
            raise ValueError(f"{label} is not a v2 report")
    for key in ("dataset_version","sampler","guidance_scale",
                "shared_noise_across_concepts"):
        if before.get(key)!=after.get(key):
            raise ValueError(f"Cannot compare different {key}")
    def signatures(doc):
        return sorted((item["image"],item["language"],item["seed"])
                      for item in doc["items"])
    if signatures(before)!=signatures(after):
        raise ValueError("Prompt/seeds differ; compare exactly the same evaluation")
    b=audit_evaluation(original)
    a=audit_evaluation(candidate)
    bsh=b["shape_accuracy_by_expected_shape"]
    ash=a["shape_accuracy_by_expected_shape"]
    bprompt=before["metrics"]["prompt_accuracy"]
    aprompt=after["metrics"]["prompt_accuracy"]
    bcolor=before["metrics"]["color_accuracy"]
    acolor=after["metrics"]["color_accuracy"]
    checks={
        "triangle_recognizable_heuristic":ash["triangle"]>=triangle_min,
        "circles_preserved":ash["circle"]>=bsh["circle"]-old_shape_drop,
        "squares_preserved":ash["square"]>=bsh["square"]-old_shape_drop,
        "colors_preserved":acolor>=bcolor-color_drop,
        "overall_prompt_accuracy_not_worse":aprompt>=bprompt,
    }
    return {
        "accepted_by_toy_gate":all(checks.values()),
        "human_visual_review_required":True,
        "checks":checks,
        "before":{
            "shape_accuracy":bsh,"color_accuracy":bcolor,
            "prompt_accuracy":bprompt,
        },
        "after":{
            "shape_accuracy":ash,"color_accuracy":acolor,
            "prompt_accuracy":aprompt,
        },
        "note":"Gate checks toy segmentation heuristics only; inspect actual 24 generated images before choosing a checkpoint.",
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline-report",required=True)
    parser.add_argument("--candidate-report",required=True)
    args=parser.parse_args()
    result=compare_reports(args.baseline_report,args.candidate_report)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if not result["accepted_by_toy_gate"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()

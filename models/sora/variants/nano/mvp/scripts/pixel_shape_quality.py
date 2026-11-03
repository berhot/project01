"""Evaluate actual Sora 24-image RGB sample galleries with strict geometry.

The ideal masks are evaluation references ONLY. No model calls, training,
synthetic output generation or checkpoint mutations occur in this script.
All per-image PNGs must be present; missing or wrong-seed samples fail
instead of being classified as successes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.pixel_shape_quality import SHAPES, score_model_sample
from src.storage_paths import sora_run


def audit_gallery(evaluation_dir):
    root=Path(evaluation_dir)
    report=json.loads((root/"report.json").read_text(encoding="utf-8"))
    if report.get("dataset_version")!="v2" or len(report.get("items",[]))!=24:
        raise ValueError("Expect a real v2 gallery with 24 bilingual samples")
    rows=[]
    identities=set()
    for entry in report["items"]:
        filename=entry["image"]
        if Path(filename).name!=filename or Path(filename).suffix!=".png":
            raise ValueError("Unsafe or unsupported gallery image path")
        parts=Path(filename).stem.split("_")
        if (len(parts)!=3 or parts[1] not in SHAPES
                or parts[2] not in ("en","ar")
                or entry["language"]!=parts[2]):
            raise ValueError("Unexpected bilingual concept filename")
        color,shape,language=parts
        identity=(color,shape,language,entry["seed"])
        if identity in identities:
            raise ValueError("Duplicate concept/seed in gallery")
        identities.add(identity)
        image_path=root/filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as original:
            rgb=np.array(original.convert("RGB"),dtype=np.float32)
        if rgb.shape!=(32,32,3):
            raise ValueError(f"Unexpected RGB image size: {image_path}")
        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1)/127.5-1
        row=score_model_sample(tensor,color,shape)
        row.update({"image":filename,"expected_color":color,
                    "seed":entry["seed"],"language":language})
        rows.append(row)
    def by_shape(shape):
        subset=[x for x in rows if x["expected_shape"]==shape]
        return {
            "samples":len(subset),
            "mean_ideal_mask_iou":round(sum(x["expected_mask_iou"]
                                             for x in subset)/len(subset),5),
            "geometry_pass_count":sum(x["qualified_geometry"] for x in subset),
            "geometry_pass_rate":round(sum(x["qualified_geometry"]
                                              for x in subset)/len(subset),4),
            "strict_prompt_pass_rate":round(sum(x["strict_prompt_pass"]
                                                  for x in subset)/len(subset),4),
            "color_accuracy":round(sum(x["color_correct"]
                                         for x in subset)/len(subset),4),
        }
    return {
        "schema_version":1,
        "scope":"32x32 synthetic-shape learned-image geometry screening; human review still required",
        "source_evaluation_dir":str(root),
        "sampler":report.get("sampler"),
        "guidance_scale":report.get("guidance_scale"),
        "x0_clipping":report.get("x0_clipping","fixed"),
        "shared_noise_across_concepts":report.get(
            "shared_noise_across_concepts",False),
        "sample_count":len(rows),
        "by_shape":{shape:by_shape(shape) for shape in SHAPES},
        "strict_prompt_accuracy":round(
            sum(x["strict_prompt_pass"] for x in rows)/len(rows),4),
        "items":rows,
    }


def compare_galleries(before,after):
    """Conservative research promotion screen, NEVER release approval."""
    if before["sample_count"]!=24 or after["sample_count"]!=24:
        raise ValueError("Different gallery sizes")
    for field in ("sampler","guidance_scale","x0_clipping",
                  "shared_noise_across_concepts"):
        if before[field]!=after[field]:
            raise ValueError(f"Cannot compare different {field}")
    key=lambda row:(row["expected_color"],row["expected_shape"],
                    row["language"],row["seed"])
    if {key(x) for x in before["items"]}!={key(x) for x in after["items"]}:
        raise ValueError("Concepts/languages/seeds do not match")
    old,new=before["by_shape"],after["by_shape"]
    checks={
        "circle_geometry_not_regressed":
            new["circle"]["geometry_pass_rate"]>=old["circle"]["geometry_pass_rate"],
        "square_geometry_not_regressed":
            new["square"]["geometry_pass_rate"]>=old["square"]["geometry_pass_rate"],
        "color_accuracy_not_regressed":all(
            new[shape]["color_accuracy"]>=old[shape]["color_accuracy"]
            for shape in SHAPES),
        "at_least_one_recognizable_triangle":
            new["triangle"]["geometry_pass_count"]>0,
        "triangle_silhouette_improved":
            new["triangle"]["mean_ideal_mask_iou"]>
            old["triangle"]["mean_ideal_mask_iou"]+0.04,
        "strict_prompt_score_improved":
            after["strict_prompt_accuracy"]>before["strict_prompt_accuracy"],
    }
    return {
        "eligible_for_human_review":all(checks.values()),
        "human_visual_review_required":True,
        "checks":checks,
        "baseline":{shape:before["by_shape"][shape] for shape in SHAPES},
        "candidate":{shape:after["by_shape"][shape] for shape in SHAPES},
        "note":"Screen only; also run pixel_quality_gate and human inspect all 24 images. Never auto-release.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir",required=True)
    parser.add_argument("--baseline-dir",default=None,
                        help="Optional matching baseline to compare")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    result=audit_gallery(args.evaluation_dir)
    if args.baseline_dir is not None:
        baseline=audit_gallery(args.baseline_dir)
        result["comparison"]=compare_galleries(baseline,result)
    destination=Path(args.output) if args.output else sora_run("shape-quality")/"report.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8") as fd:
        json.dump(result,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    print(json.dumps({
        "report":str(destination),
        "by_shape":result["by_shape"],
        "strict_prompt_accuracy":result["strict_prompt_accuracy"],
        "comparison":result.get("comparison"),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

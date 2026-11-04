"""Audit Sora Nano saved RGB evaluation results without retraining.

Reads existing guidance comparison's summary.json, report.json and PNGs.
Distinguishes prompt accuracy from bilingual agreement, and detects majority
prediction collapse. These are toy-domain heuristic diagnostics, not ground
truth human assessment. Never uses PIL to synthesize image outputs.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from src.storage_paths import sora_run

import numpy as np
from PIL import Image


def image_rgb(path):
    with Image.open(path) as image:
        if image.size != (32, 32):
            raise ValueError(f"Expected real 32x32 generated PNG: {path}")
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def audit_evaluation(report_path):
    report_path=Path(report_path)
    report=json.loads(report_path.read_text(encoding="utf-8"))
    version=report.get("dataset_version","v1")
    expected=24 if version=="v2" else 12
    items=report.get("items",[])
    if len(items)!=expected:
        raise ValueError(f"Expected {expected} generated image entries for {version}")
    predictions=Counter()
    classes=defaultdict(lambda: {"count":0,"correct":0,"en":None,"ar":None})
    color_count=Counter()
    shape_count=Counter()
    shape_totals=Counter()
    shape_correct=Counter()
    shape_prompt_correct=Counter()
    class_totals=Counter()
    class_prompt_correct=Counter()
    for item in items:
        filename=item["image"]
        if not filename.endswith(("_en.png","_ar.png")):
            raise ValueError("Unexpected model PNG name")
        suffix="_en.png" if filename.endswith("_en.png") else "_ar.png"
        expected_color, expected_shape=filename[:-len(suffix)].split("_")
        language=item["language"]
        if language not in ("en","ar") or language!=suffix[1:3]:
            raise ValueError("Image language tag mismatch")
        prediction=item["heuristic"]
        pred_color,pred_shape=prediction["color"],prediction["shape"]
        predictions[(pred_color,pred_shape)]+=1
        color_count[(expected_color,pred_color)]+=1
        shape_count[(expected_shape,pred_shape)]+=1
        key=(expected_color,expected_shape,item["seed"])
        slot=classes[key]
        if slot[language] is not None:
            raise ValueError("Duplicate image for same language/concept/seed")
        rgb=image_rgb(report_path.parent/filename)
        correct=pred_color==expected_color and pred_shape==expected_shape
        shape_totals[expected_shape] += 1
        shape_correct[expected_shape] += int(pred_shape==expected_shape)
        shape_prompt_correct[expected_shape] += int(correct)
        class_key=f"{expected_color} {expected_shape}"
        class_totals[class_key] += 1
        class_prompt_correct[class_key] += int(correct)
        slot[language]={"color":pred_color,"shape":pred_shape,"correct":correct,
                        "image":filename,"pixels":rgb}
        slot["count"]+=1
        slot["correct"]+=int(correct)
    same_prediction_different_prompts=0
    agreement_but_both_wrong=0
    pair_differences=[]
    examples=[]
    for (color,shape,seed),entry in sorted(classes.items()):
        if entry["count"]!=2 or entry["en"] is None or entry["ar"] is None:
            raise ValueError("Incomplete English/Arabic prompt pair")
        en,ar=entry["en"],entry["ar"]
        agree=(en["color"],en["shape"])==(ar["color"],ar["shape"])
        if agree:
            same_prediction_different_prompts+=1
        if agree and not en["correct"] and not ar["correct"]:
            agreement_but_both_wrong+=1
        diff=float(np.abs(en["pixels"]-ar["pixels"]).mean())
        pair_differences.append(diff)
        examples.append({
            "expected":f"{color} {shape}", "seed":seed,
            "english_prediction":f"{en['color']} {en['shape']}",
            "arabic_prediction":f"{ar['color']} {ar['shape']}",
            "both_correct":bool(en["correct"] and ar["correct"]),
            "agree_but_both_wrong":bool(agree and not en["correct"] and not ar["correct"]),
            "english_arabic_mean_absolute_pixel_difference":round(diff,6),
        })
    square_triangle_comparisons=[]
    if report.get("shared_noise_across_concepts") and version=="v2":
        # Contrast images made with EXACTLY the same initial Gaussian noise,
        # same color, same language; only the shape word changes.
        colors=sorted({color for color,_,_ in classes})
        seeds=sorted({seed for _,_,seed in classes})
        for color in colors:
            for seed in seeds:
                square=classes[(color,"square",seed)]
                triangle=classes[(color,"triangle",seed)]
                for language in ("en","ar"):
                    square_rgb=square[language]["pixels"]
                    triangle_rgb=triangle[language]["pixels"]
                    square_mask=(
                        square_rgb.max(axis=-1)-square_rgb.min(axis=-1)>0.2
                    )
                    triangle_mask=(
                        triangle_rgb.max(axis=-1)-triangle_rgb.min(axis=-1)>0.2
                    )
                    intersection=int(np.logical_and(square_mask,triangle_mask).sum())
                    union=int(np.logical_or(square_mask,triangle_mask).sum())
                    square_triangle_comparisons.append({
                        "color":color,"language":language,"seed":seed,
                        "square_triangle_pixel_mae":round(float(
                            np.abs(square_rgb-triangle_rgb).mean()
                        ),6),
                        "square_triangle_mask_iou":round(
                            intersection/union if union else 1.0,6
                        ),
                        "square_mask_area":int(square_mask.sum()),
                        "triangle_mask_area":int(triangle_mask.sum()),
                        "square_predicted_shape":square[language]["shape"],
                        "triangle_predicted_shape":triangle[language]["shape"],
                    })
    major,count=predictions.most_common(1)[0]
    return {
        "dataset_version":version,
        "guidance_scale":report.get("guidance_scale"),
        "total_images":len(items),
        "distinct_heuristic_predictions":len(predictions),
        "dominant_prediction":list(major),
        "dominant_prediction_fraction":round(count/len(items),4),
        "predicted_class_histogram":{
            " ".join(pair):count for pair,count in sorted(predictions.items())
        },
        "expected_to_predicted_color":{
            f"{a} -> {b}":n for (a,b),n in sorted(color_count.items())
        },
        "expected_to_predicted_shape":{
            f"{a} -> {b}":n for (a,b),n in sorted(shape_count.items())
        },
        "shape_accuracy_by_expected_shape":{
            shape:round(shape_correct[shape]/shape_totals[shape],4)
            for shape in sorted(shape_totals)
        },
        "prompt_accuracy_by_expected_shape":{
            shape:round(shape_prompt_correct[shape]/shape_totals[shape],4)
            for shape in sorted(shape_totals)
        },
        "prompt_accuracy_by_concept":{
            concept:round(class_prompt_correct[concept]/class_totals[concept],4)
            for concept in sorted(class_totals)
        },
        "bilingual_pairs":len(classes),
        "bilingual_heuristic_agreement":round(same_prediction_different_prompts/len(classes),4),
        "bilingual_agree_but_both_wrong":agreement_but_both_wrong,
        "mean_bilingual_pixel_difference":round(float(np.mean(pair_differences)),6),
        "pairs":examples,
        "shared_noise_across_concepts":bool(report.get("shared_noise_across_concepts",False)),
        "square_triangle_same_noise_comparison":square_triangle_comparisons,
        "mean_square_triangle_mask_iou":(
            round(float(np.mean([
                row["square_triangle_mask_iou"] for row in square_triangle_comparisons
            ])),6) if square_triangle_comparisons else None
        ),
        "original_metrics":report.get("metrics"),
    }


def analyze_directory(directory,output=None):
    root=Path(directory)
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if summary.get("task")!="sora-pixel-bilingual-guidance-comparison":
        raise ValueError("This is not a Sora RGB guidance comparison folder")
    results=[]
    for item in summary["results"]:
        report_path=Path(item["report"])
        if not report_path.exists():
            # Support moved/copied evaluation directories.
            report_path=root/Path(item["report"]).parent.name/"report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing existing model report: {report_path}")
        analysis=audit_evaluation(report_path)
        analysis["seed"]=item["seed"]
        results.append(analysis)
    result={
        "schema_version":1,
        "type":"sora-pixel-saved-samples-diagnostic",
        "quality_note":"Heuristics can misclassify blurry outputs; verify preview PNGs visually.",
        "no_retraining":True,"no_synthetic_inference":True,"results":results,
    }
    if output is not None:
        destination=Path(output)
        destination.parent.mkdir(parents=True,exist_ok=True)
        # Refuse overwrite of prior audit/report.
        with destination.open("x",encoding="utf-8") as file:
            json.dump(result,file,ensure_ascii=False,indent=2)
            file.write("\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir",required=True)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    output=args.output or str(sora_run("pixel-diagnose")/"report.json")
    audit=analyze_directory(args.evaluation_dir,output)
    print(json.dumps({
        "evaluation_dir":args.evaluation_dir,
        "summary":[{
            "scale":r["guidance_scale"],"seed":r["seed"],
            "prompt_accuracy":r["original_metrics"]["prompt_accuracy"],
            "unique_predicted_classes":r["distinct_heuristic_predictions"],
            "dominant_prediction":r["dominant_prediction"],
            "dominant_fraction":r["dominant_prediction_fraction"],
            "bilingual_agree_but_both_wrong":r["bilingual_agree_but_both_wrong"],
            "bilingual_pixel_mae":r["mean_bilingual_pixel_difference"],
            "shape_accuracy":r["shape_accuracy_by_expected_shape"],
            "prompt_by_shape":r["prompt_accuracy_by_expected_shape"],
            "prompt_by_concept":r["prompt_accuracy_by_concept"],
            "same_noise_square_triangle_mask_iou":r["mean_square_triangle_mask_iou"],
        } for r in audit["results"]],
        "saved_diagnostic":output,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

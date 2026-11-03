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
        "bilingual_pairs":len(classes),
        "bilingual_heuristic_agreement":round(same_prediction_different_prompts/len(classes),4),
        "bilingual_agree_but_both_wrong":agreement_but_both_wrong,
        "mean_bilingual_pixel_difference":round(float(np.mean(pair_differences)),6),
        "pairs":examples,
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
    audit=analyze_directory(args.evaluation_dir,args.output)
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
        } for r in audit["results"]],
        "saved_diagnostic":args.output,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

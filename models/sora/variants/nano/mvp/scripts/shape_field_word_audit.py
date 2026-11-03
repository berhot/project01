"""Read-only shape-word sensitivity audit on any saved Sora Shape Field V2 run.

Three prompts at each SAME layout and color differ ONLY by shape word.
Measures whether text actually changes the learned pixel mask. This is
not an image-generation quality guarantee, never trains or publishes.
Works for both original V2 and opt-in V2.1 training recipes.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.shape_field_v2 import load_run
from scripts.shape_first_diagnose import score_binary_mask
from src.pixel_dataset_v2 import COLORS,SHAPES,caption_v2,tokenize_v2
from src.shape_field_v2 import draw_layout_from_noise
from src.storage_paths import sora_run


def binary_iou(first,second):
    a=first>0.5
    b=second>0.5
    inter=int((a&b).sum())
    union=int((a|b).sum())
    return round(inter/union if union else 1.0,5)


@torch.no_grad()
def audit_words(run_dir,seed=42,device="cpu"):
    mask_model,_,manifest=load_run(run_dir,device)
    mask_model.eval()
    rows=[]
    for color_index,color in enumerate(COLORS):
        for language in ("en","ar"):
            prompts=[caption_v2(color,shape,language,0) for shape in SHAPES]
            tokens=torch.stack([tokenize_v2(prompt) for prompt in prompts]).to(device)
            # The ONE identical spatial layout is deliberately repeated.
            layout=draw_layout_from_noise(1,seed+color_index,device=device)
            layouts=layout.expand(len(SHAPES),-1)
            logits=mask_model(layouts,tokens)
            masks=(logits.sigmoid()>=0.5).float()
            shape_metrics=[score_binary_mask(masks[i],shape)
                           for i,shape in enumerate(SHAPES)]
            comparisons={
                f"{SHAPES[i]}_vs_{SHAPES[j]}":binary_iou(masks[i],masks[j])
                for i in range(3) for j in range(i+1,3)
            }
            rows.append({
                "color":color,"language":language,
                "layout":[round(float(v),5) for v in layout[0].tolist()],
                "prompts":prompts,
                "shape_metrics":{
                    shape:shape_metrics[i] for i,shape in enumerate(SHAPES)
                },
                "pair_mask_iou":comparisons,
            })
    by_shape={}
    for shape in SHAPES:
        selected=[r["shape_metrics"][shape] for r in rows]
        by_shape[shape]={
            "count":len(selected),
            "strict_geometry_pass_count":sum(
                r["strict_geometry_pass"] for r in selected),
            "mean_foreground_pixels":round(
                sum(r["foreground_pixels"] for r in selected)/len(selected),5),
            "mean_expected_shape_iou":round(
                sum(r["expected_mask_iou"] for r in selected)/len(selected),5),
        }
    same_shape_rates={
        key:round(sum(r["pair_mask_iou"][key] for r in rows)/len(rows),5)
        for key in rows[0]["pair_mask_iou"]
    }
    return {
        "schema_version":1,
        "source_run_dir":str(run_dir),
        "model_family":manifest["model_family"],
        "training_recipe":manifest.get("training_recipe","original-v2"),
        "seed":seed,
        "sample_count":len(rows)*3,
        "no_training":True,
        "release_approved":False,
        "shape_word_mask_overlap":same_shape_rates,
        "by_shape":by_shape,
        "items":rows,
        "interpretation":"Pair IoU near 1 indicates masks hardly change when only shape word changes; independent strict geometry remains the primary outcome.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    report=audit_words(args.run_dir,args.seed,args.device)
    output=Path(args.output) if args.output else sora_run("shape-word-audit")/"report.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2,ensure_ascii=False)
        fd.write("\n")
    print(json.dumps({
        "report":str(output),
        "shape_word_mask_overlap":report["shape_word_mask_overlap"],
        "by_shape":report["by_shape"],
        "no_training":True,"release_approved":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

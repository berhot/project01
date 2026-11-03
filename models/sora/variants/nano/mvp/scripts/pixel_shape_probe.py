"""Non-training bilingual shape-conditioning probe for an existing RGB checkpoint.

Uses held-out synthetic square/triangle targets with identical layout,
color, and x_t; measures how the denoiser reacts when only text changes.
Nothing is sampled/drawn by PIL at image inference and no weights change.
"""
import argparse
import json
from pathlib import Path

from scripts.pixel_diffusion import load_model
from src.pixel_shape_contrast import paired_prompt_probe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--pairs",type=int,default=8)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    args=parser.parse_args()
    if args.pairs<1 or args.pairs>1000:
        parser.error("--pairs must be in [1,1000]")
    model=load_model(args.checkpoint,args.device)
    if model.config.dataset_version!="v2":
        parser.error("Shape probe requires RGB dataset-v2 checkpoint")
    result=paired_prompt_probe(model,seed=args.seed,pairs=args.pairs)
    result["checkpoint"]=str(args.checkpoint)
    result["architecture"]=model.config.architecture
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2,ensure_ascii=False)
        file.write("\n")
    print(json.dumps({
        "output":str(out),
        "checkpoint":str(args.checkpoint),
        "pairs":result["pairs"],
        "predicted_epsilon_difference_rms":
            result["predicted_epsilon_difference_rms"],
        "expected_epsilon_difference_rms":
            result["expected_epsilon_difference_rms"],
        "prompt_difference_ratio":result["prompt_difference_ratio"],
        "shape_difference_alignment":result["shape_difference_alignment"],
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

"""Compare real RGB diffusion generations across classifier-free guidance scales.

Each scale/seed gets 12 independent image outputs and an inspectable JSON
report. Metrics are toy-domain heuristics, not claims of aesthetic quality.
"""
import argparse
import json
import math
from pathlib import Path

from scripts.pixel_diffusion import evaluate, load_model, require_guidance_training


def compare(model, output_dir, scales=(1.0,3.0,5.0,7.5), seeds=(42,),
            sampler="posterior_mean", sample_steps=None, shared_noise=False,
            x0_clipping="fixed"):
    if not scales or not seeds:
        raise ValueError("Need at least one scale and seed")
    if len(set(scales)) != len(scales) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate scales/seeds")
    for scale in scales:
        if not math.isfinite(scale) or not 0 <= scale <= 20:
            raise ValueError("Guidance scales must be between 0 and 20")
        require_guidance_training(model, float(scale))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for scale in scales:
        name = ("%.3f" % scale).rstrip("0").rstrip(".").replace(".", "p")
        for seed in seeds:
            directory = root / ("guidance_" + name + "_seed_" + str(seed))
            result = evaluate(
                model, directory, seed=seed, sampler=sampler,
                sample_steps=sample_steps, guidance_scale=float(scale),
                shared_noise=shared_noise,x0_clipping=x0_clipping,
            )
            results.append({
                "guidance_scale":float(scale), "seed":int(seed),
                "preview_grid":str(directory / "preview_grid.png"),
                "report":str(directory / "report.json"), "metrics":result,
            })
    summary = {
        "schema_version":1,
        "task":"sora-pixel-bilingual-guidance-comparison",
        "visual_quality":"requires human evaluation; toy metrics are heuristic",
        "checkpoint_cfg_drop_prob":getattr(model,"cfg_drop_prob",0.0),
        "shared_noise_across_concepts":bool(shared_noise),
        "sampler":sampler,
        "x0_clipping":x0_clipping,
        "results":results,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--guidance-scales", type=float, nargs="+",
                        default=[1.0,3.0,5.0,7.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--sampler", choices=("posterior_mean","ancestral","ddim"),
                        default="posterior_mean")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--x0-clipping",choices=("fixed","dynamic"),default="fixed",
                        help="Inference-only clipping ablation; fixed is historical default")
    parser.add_argument("--shared-noise", action="store_true",
                        help="Use the same starting Gaussian noise for all prompts")
    parser.add_argument("--device", choices=("cpu","mps","cuda"), default="cpu")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    summary = compare(
        model,args.output_dir,args.guidance_scales,args.seeds,
        args.sampler,args.sample_steps,args.shared_noise,args.x0_clipping,
    )
    print(json.dumps({
        "summary":str(Path(args.output_dir) / "summary.json"),
        "runs":len(summary["results"]),
        "scores":[{
            "guidance_scale":r["guidance_scale"],"seed":r["seed"],
            "prompt_accuracy":r["metrics"]["prompt_accuracy"],
            "bilingual_both_correct":r["metrics"]["bilingual_both_correct"],
        } for r in summary["results"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

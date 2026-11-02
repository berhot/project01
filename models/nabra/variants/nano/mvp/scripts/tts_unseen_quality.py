"""Generate a reproducible held-out speech listening pack from a local checkpoint."""
import argparse
import json

from src.tts.unseen_quality import evaluate_unseen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", default="data/eval/arabic_unseen_v1.txt")
    parser.add_argument("--output", required=True, help="New output directory; existing directory refused")
    parser.add_argument("--speaker")
    parser.add_argument("--train-manifest", action="append", default=[],
                        help="Repeat to reject prompts present in training manifests")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_unseen(args.checkpoint, args.prompts, args.output,
                             speaker=args.speaker, max_frames=args.max_frames,
                             device=args.device, train_sources=args.train_manifest)
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

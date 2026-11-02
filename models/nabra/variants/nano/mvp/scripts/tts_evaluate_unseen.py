"""Evaluate unseen prompts using Nabra's diagnostic recovery checkpoints."""
import argparse
import json
import torch

from src.tts.unseen_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-checkpoint", required=True)
    parser.add_argument("--vocoder-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", help="Optional JSONL: each line has id and text")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=800)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    result = evaluate(args.acoustic_checkpoint, args.vocoder_checkpoint, args.output,
                      args.prompts, args.device, args.max_frames)
    print(json.dumps({"output": args.output, "clips": len(result["prompts"]),
                      "status": result["quality_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

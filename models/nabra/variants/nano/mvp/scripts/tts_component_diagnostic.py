"""Run a non-training TTS diagnostic on existing paired recordings."""
import argparse
import json
from src.tts.component_diagnostic import inspect_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    result = inspect_components(args.checkpoint, args.output, count=args.count,
                                split=args.split, max_frames=args.max_frames,
                                device=args.device, save_audio=not args.no_audio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

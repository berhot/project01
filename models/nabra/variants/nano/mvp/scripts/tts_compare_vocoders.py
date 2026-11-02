"""Run Nabra's matched-reference vocoder comparison."""
import argparse
import json

from src.tts.vocoder_comparison import compare_vocoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-checkpoint", required=True)
    parser.add_argument("--recovery-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = compare_vocoders(args.standard_checkpoint, args.recovery_checkpoint,
                              args.output, args.record_index, args.split, args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

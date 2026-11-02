"""Inspect corpus before multi-sentence Nabra training; no weights created."""
import argparse
import json
from src.tts.multisentence import readiness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--val-source", required=True)
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-distinct", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if corpus is insufficient")
    args = parser.parse_args()
    result = readiness(args.train_source, args.val_source, args.min_train, args.min_distinct)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and result["readiness"] != "ready-for-experimental-training":
        parser.exit(2, "Corpus lacks enough recordings or distinct prompts.\n")


if __name__ == "__main__":
    main()

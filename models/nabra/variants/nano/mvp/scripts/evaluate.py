
import argparse

from src.ingestion import inspect_split
from src.inference import get_device, load_checkpoint
from src.evaluator import (
    evaluate_asr,
    write_evaluation,
)
from src.packaging import sha256_file
from src.paths import VAL_ROOT

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", required=True
    )

    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    device = get_device()

    model, tokenizer, feature_config = load_checkpoint(
        args.checkpoint, device
    )

    records = inspect_split(VAL_ROOT)

    expected = checkpoint["source_manifest"]["val"]

    if records != expected:
        raise ValueError(
            "Validation dataset changed since training"
        )

    result = evaluate_asr(
        model, tokenizer, feature_config, records
    )

    report = write_evaluation(
        checkpoint["run_id"],
        checkpoint["epoch"],
        sha256_file(args.checkpoint),
        result,
    )

    print("NABRA — Evaluation")
    print("CER:", f"{result['cer']:.4f}")
    print("WER:", f"{result['wer']:.4f}")
    print("Report:", report)
    print("Status: PASS")


if __name__ == "__main__":
    main()

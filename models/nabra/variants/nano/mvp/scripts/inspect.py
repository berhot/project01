
import argparse

from src.ingestion import (
    inspect_split,
    write_manifest,
)

from src.paths import (
    TRAIN_ROOT,
    VAL_ROOT,
    DATA_ROOT,
)


def main():
    parser = argparse.ArgumentParser(
        description="Nabra speech data inspection"
    )

    parser.add_argument(
        "--train",
        default=str(TRAIN_ROOT),
    )

    parser.add_argument(
        "--val",
        default=str(VAL_ROOT),
    )

    args = parser.parse_args()

    train = inspect_split(args.train)
    val = inspect_split(args.val)

    train_hashes = {
        item["audio_sha256"]
        for item in train
    }

    val_hashes = {
        item["audio_sha256"]
        for item in val
    }

    if train_hashes & val_hashes:
        raise ValueError(
            "Duplicate audio across train and validation"
        )

    write_manifest(
        train,
        DATA_ROOT / "train.jsonl",
    )

    write_manifest(
        val,
        DATA_ROOT / "val.jsonl",
    )

    print("NABRA Nano MVP — Ingestion")
    print("--------------------------")
    print("Training recordings:", len(train))
    print("Validation recordings:", len(val))
    print(
        "Training audio seconds:",
        round(sum(x["duration_seconds"] for x in train), 2),
    )
    print(
        "Validation audio seconds:",
        round(sum(x["duration_seconds"] for x in val), 2),
    )
    print("Status: PASS")


if __name__ == "__main__":
    main()

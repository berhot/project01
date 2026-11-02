
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from .dataset import TextDataset
from .ingestion import corpus_sha256
from .model import JSMModel, ModelConfig
from .paths import DEFAULT_CORPUS, get_run_dir


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


@torch.inference_mode()
def evaluate_model(model, dataset, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in dataset.get_validation_windows(
        batch_size=8
    ):
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        count = y.numel()
        total_loss += loss.item() * count
        total_tokens += count

    if total_tokens == 0:
        raise ValueError("No validation tokens evaluated")

    mean_loss = total_loss / total_tokens

    return {
        "validation_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "evaluated_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data", default=str(DEFAULT_CORPUS)
    )

    args = parser.parse_args()

    run_dir = get_run_dir(args.run_id).resolve()
    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    checkpoints_dir = (
        run_dir / "checkpoints"
    ).resolve()

    if not checkpoint_path.is_relative_to(
        checkpoints_dir
    ):
        raise ValueError(
            "Checkpoint does not belong to this run"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("tokenizer_type") != "char-v1":
        raise ValueError("Unsupported tokenizer")

    actual_hash = corpus_sha256(args.data)

    if checkpoint["corpus_sha256"] != actual_hash:
        raise ValueError("Corpus has changed")

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    dataset = TextDataset(
        file_path=args.data,
        block_size=config.block_size,
    )

    if dataset.tokenizer.chars != checkpoint["tokenizer_chars"]:
        raise ValueError("Tokenizer mismatch")

    device = get_device()
    model = JSMModel(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    metrics = evaluate_model(
        model, dataset, device
    )

    result = {
        "schema_version": "1.0.0",
        "model_family": "jsm",
        "variant": "nano",
        "run_id": args.run_id,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_step": int(checkpoint["step"]),
        "corpus_sha256": actual_hash,
        "tokenizer_type": "char-v1",
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        **metrics,
    }

    output_dir = run_dir / "evaluations"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        output_dir
        / f"step_{checkpoint['step']:06d}.json"
    )

    if output.exists():
        raise FileExistsError(
            f"Evaluation already exists: {output}"
        )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("JSM Nano MVP — Evaluation")
    print("-------------------------")
    print("Run:", args.run_id)
    print("Step:", checkpoint["step"])
    print(
        "Validation loss:",
        f"{metrics['validation_loss']:.4f}",
    )
    print(
        "Perplexity:",
        f"{metrics['perplexity']:.4f}",
    )
    print("Tokens:", metrics["evaluated_tokens"])
    print("Report:", output)
    print("Status: PASS")


if __name__ == "__main__":
    main()

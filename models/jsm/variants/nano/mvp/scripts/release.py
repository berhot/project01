
import argparse
import hashlib
import json
import math
import shutil
import uuid

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from .model import JSMModel, ModelConfig
from .paths import get_run_dir, get_release_dir
from .tokenizer import CharTokenizer


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--max-val-loss",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--approve",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.approve:
        parser.error("Explicit --approve required")

    if not math.isfinite(args.max_val_loss):
        parser.error("Invalid validation threshold")

    run_dir = get_run_dir(args.run_id).resolve()
    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    if not checkpoint_path.is_relative_to(
        (run_dir / "checkpoints").resolve()
    ):
        raise ValueError(
            "Checkpoint is outside the selected run"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("tokenizer_type") != "char-v1":
        raise ValueError("Unsupported tokenizer")

    step = int(checkpoint["step"])

    evaluation_path = (
        run_dir
        / "evaluations"
        / f"step_{step:06d}.json"
    )

    if not evaluation_path.is_file():
        raise FileNotFoundError(
            "Evaluate this checkpoint before releasing it"
        )

    evaluation = json.loads(
        evaluation_path.read_text(
            encoding="utf-8"
        )
    )

    if evaluation["run_id"] != args.run_id:
        raise ValueError("Evaluation run mismatch")

    if evaluation["training_step"] != step:
        raise ValueError("Evaluation step mismatch")

    if evaluation["checkpoint"] != checkpoint_path.name:
        raise ValueError("Evaluation checkpoint mismatch")

    if evaluation["checkpoint_sha256"] != sha256_file(
        checkpoint_path
    ):
        raise ValueError(
            "Checkpoint changed after evaluation"
        )

    if evaluation["corpus_sha256"] != checkpoint["corpus_sha256"]:
        raise ValueError("Corpus fingerprint mismatch")

    val_loss = float(evaluation["validation_loss"])

    if not math.isfinite(val_loss):
        raise ValueError("Invalid validation loss")

    if val_loss > args.max_val_loss:
        raise ValueError(
            f"Validation loss {val_loss:.4f} exceeds "
            f"threshold {args.max_val_loss:.4f}"
        )

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    tokenizer = CharTokenizer(
        checkpoint["tokenizer_chars"]
    )

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Vocabulary size mismatch")

    # Validate original checkpoint weights.
    model = JSMModel(config)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    release_dir = get_release_dir(args.version)

    if release_dir.exists():
        raise FileExistsError(
            f"Release exists: {release_dir}"
        )

    release_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        release_dir.parent
        / f".{args.version}-{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()

    try:
        weights = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor
            in model.state_dict().items()
        }

        weights_path = (
            temporary / "model.safetensors"
        )

        save_file(
            weights,
            str(weights_path),
        )

        # Verify the exported weights reload.
        exported = load_file(
            str(weights_path),
            device="cpu",
        )

        verification_model = JSMModel(config)
        verification_model.load_state_dict(
            exported,
            strict=True,
        )

        for name, tensor in weights.items():
            if not torch.equal(
                tensor,
                exported[name],
            ):
                raise ValueError(
                    f"Weight export mismatch: {name}"
                )

        write_json(
            temporary / "config.json",
            asdict(config),
        )

        write_json(
            temporary / "tokenizer.json",
            tokenizer.to_dict(),
        )

        file_hashes = {
            name: {
                "sha256": sha256_file(
                    temporary / name
                )
            }
            for name in (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
            )
        }

        manifest = {
            "schema_version": "1.0.0",
            "model_family": "jsm",
            "variant": "nano",
            "version": args.version,
            "source_run_id": args.run_id,
            "training_step": step,
            "corpus_sha256": checkpoint["corpus_sha256"],
            "tokenizer_type": "char-v1",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "approval": {
                "approved": True,
                "method": "explicit-cli-approval",
                "max_validation_loss": args.max_val_loss,
            },
            "evaluation": {
                "validation_loss": val_loss,
                "perplexity": evaluation["perplexity"],
                "evaluated_tokens": evaluation["evaluated_tokens"],
                "checkpoint_sha256": (
                    evaluation["checkpoint_sha256"]
                ),
            },
            "files": file_hashes,
        }

        write_json(
            temporary / "manifest.json",
            manifest,
        )

        # Final publication only after verification.
        if release_dir.exists():
            raise FileExistsError(release_dir)

        temporary.rename(release_dir)

    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print("JSM Nano MVP — Release")
    print("----------------------")
    print("Version:", args.version)
    print("Source run:", args.run_id)
    print("Training step:", step)
    print("Validation loss:", f"{val_loss:.4f}")
    print("Release:", release_dir)
    print("Status: PACKAGED")


if __name__ == "__main__":
    main()

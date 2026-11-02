
import json
import math
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .ingestion import inspect_split, read_records, validate_splits
from .runtime import select_device, fingerprint, environment
from .dataset import (
    build_datasets,
    collate_speech,
)
from .features import FeatureConfig
from .model import (
    ModelConfig,
    NabraASR,
    ctc_loss,
)
from .evaluator import edit_distance
from .paths import (
    TRAIN_ROOT,
    VAL_ROOT,
    new_run_id,
    prepare_run,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_speech,
    )


def required_ctc_lengths(targets, target_lengths):
    result = []
    offset = 0

    for length in target_lengths.tolist():
        ids = targets[offset:offset + length].tolist()

        repetitions = sum(
            a == b
            for a, b in zip(ids, ids[1:])
        )

        result.append(length + repetitions)
        offset += length

    return result


def forward_loss(model, batch, device):
    logits, lengths = model(
        batch["features"].to(device),
        batch["feature_lengths"],
    )

    required = required_ctc_lengths(
        batch["targets"],
        batch["target_lengths"],
    )

    for available, needed in zip(
        lengths.tolist(),
        required,
    ):
        if available < needed:
            raise ValueError(
                "CTC alignment impossible: "
                f"{available} frames, "
                f"{needed} required"
            )

    loss = ctc_loss(
        logits,
        lengths,
        batch["targets"],
        batch["target_lengths"],
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite CTC loss"
        )

    return loss, logits, lengths


def prediction_stats(
    logits,
    lengths,
    transcripts,
    tokenizer,
):
    predicted = logits.detach().argmax(
        dim=-1
    ).cpu()

    total_blanks = 0
    total_frames = 0
    character_errors = 0
    character_count = 0
    word_errors = 0
    word_count = 0
    samples = []

    for index, reference in enumerate(transcripts):
        length = int(lengths[index])

        ids = predicted[index, :length].tolist()

        total_blanks += ids.count(0)
        total_frames += len(ids)

        hypothesis = tokenizer.ctc_decode(ids)

        character_errors += edit_distance(
            list(reference),
            list(hypothesis),
        )
        character_count += len(reference)

        reference_words = reference.split()
        hypothesis_words = hypothesis.split()

        word_errors += edit_distance(
            reference_words,
            hypothesis_words,
        )
        word_count += len(reference_words)

        samples.append({
            "reference": reference,
            "hypothesis": hypothesis,
        })

    return {
        "blanks": total_blanks,
        "frames": total_frames,
        "char_errors": character_errors,
        "chars": character_count,
        "word_errors": word_errors,
        "words": word_count,
        "samples": samples,
    }


def summarize(losses, stats):
    return {
        "loss": sum(losses) / len(losses),
        "cer": stats["char_errors"] / max(
            stats["chars"], 1
        ),
        "wer": stats["word_errors"] / max(
            stats["words"], 1
        ),
        "blank_rate": stats["blanks"] / max(
            stats["frames"], 1
        ),
    }


def empty_stats():
    return {
        "blanks": 0,
        "frames": 0,
        "char_errors": 0,
        "chars": 0,
        "word_errors": 0,
        "words": 0,
        "samples": [],
    }


def accumulate(total, current):
    for field in (
        "blanks",
        "frames",
        "char_errors",
        "chars",
        "word_errors",
        "words",
    ):
        total[field] += current[field]

    total["samples"].extend(current["samples"])


@torch.no_grad()
def evaluate_loader(model, loader, tokenizer, device):
    model.eval()

    losses = []
    stats = empty_stats()

    for batch in loader:
        loss, logits, lengths = forward_loss(
            model, batch, device
        )

        losses.append(float(loss.item()))

        current = prediction_stats(
            logits,
            lengths,
            batch["transcripts"],
            tokenizer,
        )

        accumulate(stats, current)

    return summarize(losses, stats), stats["samples"]


def save_checkpoint(
    path,
    model,
    optimizer,
    model_config,
    feature_config,
    tokenizer,
    epoch,
    run_id,
    source_manifest,
    seed,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "feature_config": asdict(feature_config),
        "tokenizer": tokenizer.to_dict(),
        "epoch": epoch,
        "step": max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0),
        "run_id": run_id,
        "source_manifest": source_manifest,
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "dataset_fingerprints": {k: fingerprint(v) for k, v in source_manifest.items()},
        "environment": environment(),
    }

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    epochs=100,
    batch_size=2,
    learning_rate=0.0005,
    seed=42,
    resume=None,
    run_id=None,
    blank_bias=-2.0,
    save_interval=10,
    train_source=None,
    val_source=None,
    device_name="auto",
    model_options=None,
    feature_options=None,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")

    if not math.isfinite(blank_bias):
        raise ValueError("Invalid blank bias")

    if save_interval < 1:
        raise ValueError("Invalid save interval")

    random.seed(seed)
    torch.manual_seed(seed)

    train_records = read_records(train_source or TRAIN_ROOT)
    val_records = read_records(val_source or VAL_ROOT)

    train_hashes = {
        record["audio_sha256"]
        for record in train_records
    }

    val_hashes = {
        record["audio_sha256"]
        for record in val_records
    }

    if train_hashes & val_hashes:
        raise ValueError(
            "Duplicate recordings across splits"
        )

    source_manifest = {
        "train": train_records,
        "val": val_records,
    }

    feature_config = FeatureConfig(**(feature_options or {}))

    train_data, val_data, tokenizer = build_datasets(
        train_records,
        val_records,
        feature_config,
    )

    options = {"n_mels": feature_config.n_mels, "vocab_size": tokenizer.vocab_size, **(model_options or {})}
    model_config = ModelConfig(**options)
    if model_config.n_mels != feature_config.n_mels or model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model configuration does not match dataset/features")

    start_epoch = 0
    checkpoint = None

    if resume:
        checkpoint = torch.load(
            resume,
            map_location="cpu",
            weights_only=True,
        )

        if {k: fingerprint(v) for k, v in checkpoint["source_manifest"].items()} != {k: fingerprint(v) for k, v in source_manifest.items()}:
            raise ValueError(
                "Dataset changed since checkpoint"
            )

        if checkpoint["tokenizer"] != tokenizer.to_dict():
            raise ValueError("Tokenizer mismatch")

        if checkpoint["feature_config"] != asdict(feature_config):
            raise ValueError(
                "Feature configuration mismatch"
            )

        if model_options is not None and checkpoint["model_config"] != asdict(model_config):
            raise ValueError("Model configuration mismatch")
        model_config = ModelConfig(**checkpoint["model_config"])
        if model_config.vocab_size != tokenizer.vocab_size or model_config.n_mels != feature_config.n_mels:
            raise ValueError("Checkpoint model configuration mismatch")

        start_epoch = int(checkpoint["epoch"])

        if epochs <= start_epoch:
            raise ValueError(
                "Target epochs must exceed resumed epoch"
            )

    device = select_device(device_name)

    model = NabraASR(model_config).to(device)

    if checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
    else:
        with torch.no_grad():
            model.output.bias[0] = blank_bias

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    if checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint["torch_rng_state"]
            )

    if checkpoint and checkpoint.get("cuda_rng_states") and device.type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    if checkpoint and checkpoint.get("mps_rng_state") is not None and device.type == "mps":
        torch.mps.set_rng_state(checkpoint["mps_rng_state"])

    run_id = run_id or new_run_id()
    run_dir = prepare_run(run_id)

    run_info = run_dir / "logs" / "run.json"

    if run_info.exists():
        raise FileExistsError(
            f"Run already exists: {run_id}"
        )

    run_info.write_text(
        json.dumps({
            "run_id": run_id,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_config": asdict(model_config),
            "feature_config": asdict(feature_config),
            "start_epoch": start_epoch,
            "target_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "requested_learning_rate": learning_rate,
            "blank_bias": (
                None if checkpoint else blank_bias
            ),
            "seed": seed,
            "resume": str(resume) if resume else None,
            "environment": environment(),
            "dataset_fingerprints": {k: fingerprint(v) for k, v in source_manifest.items()},
        }, indent=2),
        encoding="utf-8",
    )

    train_loader = create_loader(
        train_data,
        batch_size,
        shuffle=True,
    )

    val_loader = create_loader(
        val_data,
        batch_size,
        shuffle=False,
    )

    metrics_path = (
        run_dir / "logs" / "metrics.jsonl"
    )

    best_cer = float("inf")
    best_loss = float("inf")
    best_epoch = None
    best_checkpoint = None
    if checkpoint:
        resumed_metrics, _ = evaluate_loader(model, val_loader, tokenizer, device)
        best_cer, best_loss = resumed_metrics["cer"], resumed_metrics["loss"]
        best_epoch, best_checkpoint = start_epoch, Path(resume).resolve()

    print("NABRA Nano — Training v2")
    print("-------------------------")
    print("Run:", run_id)
    print("Device:", device)
    print("Train clips:", len(train_data))
    print("Validation clips:", len(val_data))
    print("Vocabulary:", tokenizer.vocab_size)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ), flush=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()

        losses = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            loss, _, _ = forward_loss(
                model,
                batch,
                device,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            losses.append(float(loss.item()))

        train_metrics, train_samples = evaluate_loader(
            model,
            create_loader(
                train_data,
                batch_size,
                shuffle=False,
            ),
            tokenizer,
            device,
        )

        val_metrics, val_samples = evaluate_loader(
            model,
            val_loader,
            tokenizer,
            device,
        )

        record = {
            "epoch": epoch,
            "optimizer_train_loss": (
                sum(losses) / len(losses)
            ),
            "train": train_metrics,
            "validation": val_metrics,
        }

        with metrics_path.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(record) + "\n"
            )

        current_cer = val_metrics["cer"]
        current_loss = val_metrics["loss"]

        is_best = (
            current_cer < best_cer
            or (
                current_cer == best_cer
                and current_loss < best_loss
            )
        )

        if is_best:
            best_cer = current_cer
            best_loss = current_loss
            best_epoch = epoch

        should_save = (
            is_best
            or epoch % save_interval == 0
            or epoch == epochs
        )

        if should_save:
            checkpoint_path = (
                run_dir / "checkpoints"
                / f"epoch_{epoch:04d}.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                model_config,
                feature_config,
                tokenizer,
                epoch,
                run_id,
                source_manifest,
                seed,
            )

            if is_best:
                best_checkpoint = checkpoint_path

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train CER {train_metrics['cer']:.3f} | "
            f"Val CER {current_cer:.3f} | "
            f"Blank {val_metrics['blank_rate']:.1%} | "
            f"Val loss {current_loss:.4f}",
            flush=True,
        )

        if epoch == 1 or epoch % 10 == 0 or is_best:
            print(
                "Validation prediction:",
                repr(val_samples[0]["hypothesis"]),
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_validation_cer": best_cer,
        "best_validation_loss": best_loss,
        "best_checkpoint": str(best_checkpoint),
        "final_epoch": epochs,
    }

    (
        run_dir / "logs" / "summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("-------------------------")
    print("Training complete")
    print("Best epoch:", best_epoch)
    print("Best validation CER:", best_cer)
    print("Best checkpoint:", best_checkpoint)

    return best_checkpoint

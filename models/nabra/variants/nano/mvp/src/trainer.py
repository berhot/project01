
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .ingestion import inspect_split
from .dataset import build_datasets, collate_speech
from .features import FeatureConfig
from .model import ModelConfig, NabraASR, ctc_loss
from .paths import (
    TRAIN_ROOT, VAL_ROOT, new_run_id, prepare_run,
)


def device_for_training():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_speech,
    )


def batch_loss(model, batch, device):
    features = batch["features"].to(device)
    lengths = batch["feature_lengths"]

    logits, output_lengths = model(
        features, lengths
    )

    targets = batch["targets"]
    target_lengths = batch["target_lengths"]

    # Repeated adjacent characters need
    # additional CTC alignment positions.
    offset = 0
    minimums = []

    for n in target_lengths.tolist():
        ids = targets[offset:offset + n].tolist()
        repeated = sum(
            a == b for a, b in zip(ids, ids[1:])
        )
        minimums.append(n + repeated)
        offset += n

    if any(
        available < required
        for available, required in zip(
            output_lengths.tolist(), minimums
        )
    ):
        raise ValueError(
            "Audio too short for its transcript "
            "under the current CTC encoder."
        )

    return ctc_loss(
        logits, output_lengths,
        targets, target_lengths,
    )


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    losses = []

    for batch in loader:
        loss = batch_loss(model, batch, device)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def save_checkpoint(
    path, model, optimizer, model_config,
    feature_config, tokenizer, epoch,
    run_id, source_manifest,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "feature_config": asdict(feature_config),
        "tokenizer": tokenizer.to_dict(),
        "epoch": epoch,
        "run_id": run_id,
        "source_manifest": source_manifest,
        "torch_rng_state": torch.get_rng_state(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".pt.tmp")

    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    epochs=10,
    batch_size=4,
    learning_rate=1e-3,
    seed=42,
    resume=None,
    run_id=None,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Invalid training settings")

    random.seed(seed)
    torch.manual_seed(seed)

    train_records = inspect_split(TRAIN_ROOT)
    val_records = inspect_split(VAL_ROOT)

    train_hashes = {
        r["audio_sha256"] for r in train_records
    }
    val_hashes = {
        r["audio_sha256"] for r in val_records
    }

    if train_hashes & val_hashes:
        raise ValueError(
            "Audio duplicated across train and validation"
        )

    source_manifest = {
        "train": train_records,
        "val": val_records,
    }

    feature_config = FeatureConfig()

    train_data, val_data, tokenizer = build_datasets(
        train_records, val_records, feature_config
    )

    model_config = ModelConfig(
        n_mels=feature_config.n_mels,
        vocab_size=tokenizer.vocab_size,
    )

    checkpoint = None
    start_epoch = 0

    if resume:
        checkpoint = torch.load(
            resume,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["source_manifest"] != source_manifest:
            raise ValueError(
                "Dataset changed since checkpoint"
            )

        if checkpoint["tokenizer"] != tokenizer.to_dict():
            raise ValueError("Tokenizer mismatch")

        if checkpoint["feature_config"] != asdict(feature_config):
            raise ValueError("Feature configuration mismatch")

        model_config = ModelConfig(
            **checkpoint["model_config"]
        )
        start_epoch = int(checkpoint["epoch"])

        if epochs <= start_epoch:
            raise ValueError(
                "Target epochs must exceed checkpoint epoch"
            )

    device = device_for_training()

    model = NabraASR(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    if checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint["torch_rng_state"]
            )

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
            "learning_rate": learning_rate,
            "resume": resume,
        }, indent=2),
        encoding="utf-8",
    )

    train_loader = make_loader(
        train_data, batch_size, True
    )
    val_loader = make_loader(
        val_data, batch_size, False
    )

    history = run_dir / "logs" / "metrics.jsonl"

    print("NABRA Nano — Training")
    print("Run ID:", run_id)
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

            loss = batch_loss(model, batch, device)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        val_loss = validation_loss(
            model, val_loader, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        with history.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(record) + "\n")

        output = (
            run_dir / "checkpoints"
            / f"epoch_{epoch:04d}.pt"
        )

        save_checkpoint(
            output, model, optimizer,
            model_config, feature_config,
            tokenizer, epoch, run_id,
            source_manifest,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}",
            flush=True,
        )

    print("Training complete")
    print("Run ID:", run_id)
    print("Checkpoint:", output)
    return output

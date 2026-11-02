
import argparse
import json
import random
from dataclasses import asdict

import torch

from .ingestion import inspect_split
from .features import FeatureConfig
from .dataset import SpeechDataset, collate_speech
from .tokenizer import CharTokenizer
from .model import ModelConfig, NabraASR, ctc_loss
from .evaluator import edit_distance
from .paths import (
    TRAIN_ROOT,
    VAL_ROOT,
    new_run_id,
    prepare_run,
)


def select_device(name):
    if name != "auto":
        device = torch.device(name)
        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is unavailable")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@torch.no_grad()
def diagnose(model, batch, tokenizer, device):
    was_training = model.training
    model.eval()

    logits, lengths = model(
        batch["features"].to(device),
        batch["feature_lengths"],
    )

    length = int(lengths[0].item())
    frame_logits = logits[0, :length]

    frame_ids = frame_logits.argmax(
        dim=-1
    ).cpu().tolist()

    blank_rate = (
        sum(token == 0 for token in frame_ids)
        / len(frame_ids)
    )

    decoded = tokenizer.ctc_decode(frame_ids)
    reference = batch["transcripts"][0]

    errors = edit_distance(
        list(reference),
        list(decoded),
    )

    cer = errors / max(len(reference), 1)

    if was_training:
        model.train()

    return {
        "blank_rate": blank_rate,
        "cer": cer,
        "reference": reference,
        "hypothesis": decoded,
        "output_frames": length,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps", type=int, default=400
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--report-every", type=int, default=25
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )

    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("Steps must be positive")

    if args.report_every < 1:
        raise ValueError("Report interval must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)

    train_records = inspect_split(TRAIN_ROOT)
    val_records = inspect_split(VAL_ROOT)

    training_hashes = {
        record["audio_sha256"]
        for record in train_records
    }

    if training_hashes & {
        record["audio_sha256"]
        for record in val_records
    }:
        raise ValueError(
            "Training and validation share audio"
        )

    tokenizer = CharTokenizer.from_transcripts(
        [
            record["transcript"]
            for record in train_records
        ]
    )

    feature_config = FeatureConfig()

    dataset = SpeechDataset(
        [train_records[0]],
        tokenizer,
        feature_config,
    )

    batch = collate_speech([dataset[0]])

    model_config = ModelConfig(
        n_mels=feature_config.n_mels,
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.0,
    )

    model = NabraASR(model_config).to(device)

    # Experiment-specific initialization:
    # Discourage blank-only predictions at the start.
    # CTC blanks remain enabled and trainable.
    with torch.no_grad():
        model.output.bias[0] = -2.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    reference = batch["transcripts"][0]

    target_length = int(
        batch["target_lengths"][0].item()
    )

    target_ids = batch["targets"].tolist()

    repeated = sum(
        a == b
        for a, b in zip(target_ids, target_ids[1:])
    )

    available = int(
        model.output_lengths(
            batch["feature_lengths"]
        )[0].item()
    )

    required = target_length + repeated

    if available < required:
        raise ValueError(
            f"CTC alignment impossible: "
            f"{available} frames available, "
            f"{required} required"
        )

    run_id = new_run_id()
    run_dir = prepare_run(run_id)

    print("NABRA — CTC Overfit Diagnostic")
    print("------------------------------")
    print("Run:", run_id)
    print("Device:", device)
    print("Audio:", train_records[0]["audio"])
    print("Reference:", reference)
    print("CTC frames:", available)
    print("Required frames:", required)
    print("Vocabulary:", tokenizer.vocab_size)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )

    metrics_path = (
        run_dir / "logs" / "ctc_overfit.jsonl"
    )

    best_cer = float("inf")
    best_step = 0
    success = False

    for step in range(1, args.steps + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        logits, lengths = model(
            batch["features"].to(device),
            batch["feature_lengths"],
        )

        loss = ctc_loss(
            logits,
            lengths,
            batch["targets"],
            batch["target_lengths"],
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite CTC loss at step {step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        should_report = (
            step == 1
            or step % args.report_every == 0
            or step == args.steps
        )

        if not should_report:
            continue

        metrics = diagnose(
            model, batch, tokenizer, device
        )

        record = {
            "step": step,
            "loss": float(loss.item()),
            **metrics,
        }

        with metrics_path.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )

        print(
            f"Step {step:04d} | "
            f"Loss {loss.item():.4f} | "
            f"Blank {metrics['blank_rate']:.1%} | "
            f"CER {metrics['cer']:.3f}",
            flush=True,
        )

        print(
            "Prediction:",
            repr(metrics["hypothesis"]),
            flush=True,
        )

        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            best_step = step

        if metrics["cer"] == 0.0:
            success = True
            break

    checkpoint_path = (
        run_dir
        / "checkpoints"
        / f"overfit_step_{step:06d}.pt"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "feature_config": asdict(feature_config),
        "tokenizer": tokenizer.to_dict(),
        "epoch": 0,
        "step": step,
        "run_id": run_id,
        "source_manifest": {
            "train": train_records,
            "val": val_records,
        },
        "experiment": "one-recording-ctc-overfit",
        "torch_rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, checkpoint_path)

    summary = {
        "run_id": run_id,
        "steps_completed": step,
        "exact_training_transcription": success,
        "best_training_cer": best_cer,
        "best_step": best_step,
        "checkpoint": str(checkpoint_path),
    }

    (
        run_dir / "logs" / "ctc_overfit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("------------------------------")
    print(
        "Result:",
        "PASS" if success else "NOT YET CONVERGED",
    )
    print("Best training CER:", best_cer)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()

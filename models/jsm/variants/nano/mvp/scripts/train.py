
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .dataset import TextDataset
from .ingestion import corpus_sha256
from .model import JSMModel, ModelConfig
from .paths import DEFAULT_CORPUS, new_run_id, prepare_run


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model, dataset, device, batch_size):
    model.eval()
    results = {}

    for split in ("train", "val"):
        losses = []
        for _ in range(10):
            x, y = dataset.get_batch(
                split, batch_size, device
            )
            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


def save_checkpoint(
    path, model, optimizer, config,
    tokenizer, step, corpus_hash,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config),
        "tokenizer_type": "char-v1",
        "tokenizer_chars": tokenizer.chars,
        "corpus_sha256": corpus_hash,
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_states"] = (
            torch.cuda.get_rng_state_all()
        )

    temporary = path.with_suffix(".pt.tmp")

    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Checkpoint saved: {path}", flush=True)


def train(args):
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    if args.block_size < 1:
        raise ValueError("Invalid block size")
    if args.eval_interval < 1 or args.save_interval < 1:
        raise ValueError("Intervals must be positive")
    if args.learning_rate <= 0:
        raise ValueError("Invalid learning rate")

    torch.manual_seed(args.seed)
    device = get_device()

    dataset = TextDataset(
        file_path=args.data,
        block_size=args.block_size,
    )

    corpus_hash = corpus_sha256(args.data)
    checkpoint = None
    start_step = 0

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("tokenizer_type") != "char-v1":
            raise ValueError("Tokenizer type mismatch")

        if checkpoint["corpus_sha256"] != corpus_hash:
            raise ValueError("Corpus fingerprint mismatch")

        if checkpoint["tokenizer_chars"] != dataset.tokenizer.chars:
            raise ValueError("Vocabulary mismatch")

        config = ModelConfig(**checkpoint["model_config"])

        if config.block_size != args.block_size:
            raise ValueError("Context length mismatch")
        if config.vocab_size != dataset.tokenizer.vocab_size:
            raise ValueError("Vocabulary size mismatch")

        start_step = int(checkpoint["step"])

        if args.steps <= start_step:
            raise ValueError("Target step must exceed saved step")

    else:
        config = ModelConfig(
            vocab_size=dataset.tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=0.0,
        )

    model = JSMModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    if checkpoint is not None:
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
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_states"]
            )

        print("Restored checkpoint and optimizer state.")

    run_id = args.run_id or new_run_id()
    run_dir = prepare_run(run_id)

    history_path = run_dir / "logs" / "metrics.jsonl"
    run_info_path = run_dir / "logs" / "run.json"

    if run_info_path.exists() or history_path.exists():
        raise FileExistsError(
            f"Run ID already exists: {run_id}"
        )

    run_info = {
        "run_id": run_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "corpus_source": str(Path(args.data).resolve()),
        "corpus_sha256": corpus_hash,
        "model_config": asdict(config),
        "starting_step": start_step,
        "target_step": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "device": str(device),
        "resume_from": (
            str(Path(args.resume).resolve())
            if args.resume else None
        ),
    }

    run_info_path.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("JSM Nano MVP — Training")
    print("-----------------------")
    print("Run ID:", run_id)
    print("Run directory:", run_dir)
    print("Device:", device)
    print("Vocabulary:", config.vocab_size)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
    print("Train tokens:", len(dataset.train_data))
    print("Validation tokens:", len(dataset.val_data))
    print("Starting step:", start_step)
    print("Target step:", args.steps)

    model.train()

    for step in range(start_step + 1, args.steps + 1):
        x, y = dataset.get_batch(
            "train",
            batch_size=args.batch_size,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if (
            step == start_step + 1
            or step % args.eval_interval == 0
            or step == args.steps
        ):
            metrics = estimate_loss(
                model, dataset, device, args.batch_size
            )

            record = {
                "step": step,
                "train_loss": metrics["train"],
                "val_loss": metrics["val"],
            }

            with history_path.open(
                "a", encoding="utf-8"
            ) as file:
                file.write(json.dumps(record) + "\n")

            print(
                f"Step {step}/{args.steps} | "
                f"Train: {metrics['train']:.4f} | "
                f"Val: {metrics['val']:.4f}",
                flush=True,
            )

        if (
            step % args.save_interval == 0
            or step == args.steps
        ):
            path = (
                run_dir / "checkpoints"
                / f"step_{step:06d}.pt"
            )

            save_checkpoint(
                path, model, optimizer, config,
                dataset.tokenizer, step, corpus_hash,
            )

    print("Training complete.")
    print("Run ID:", run_id)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", default=str(DEFAULT_CORPUS)
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4
    )
    parser.add_argument(
        "--eval-interval", type=int, default=100
    )
    parser.add_argument(
        "--save-interval", type=int, default=500
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

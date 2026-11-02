
import argparse
import hashlib

from pathlib import Path
from dataclasses import asdict

import torch

from scripts.dataset import TextDataset
from scripts.model import JSMModel, ModelConfig


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def corpus_sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


@torch.no_grad()
def estimate_loss(
    model,
    dataset,
    device,
    batch_size,
    eval_batches=10,
):
    model.eval()
    results = {}

    for split in ("train", "val"):
        losses = []

        for _ in range(eval_batches):
            x, y = dataset.get_batch(
                split=split,
                batch_size=batch_size,
                device=device,
            )

            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


def save_checkpoint(
    model,
    optimizer,
    config,
    tokenizer,
    step,
    data_path,
    output_path,
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config),
        "tokenizer_type": "char-v1",
        "tokenizer_chars": tokenizer.chars,
        "corpus_sha256": corpus_sha256(data_path),
        "step": step,

        # Preserve random sampling state for resume.
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_states"] = (
            torch.cuda.get_rng_state_all()
        )

    # Write to a temporary file before replacing output.
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    checkpoint_path,
    dataset,
    data_path,
    block_size,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("tokenizer_type") != "char-v1":
        raise ValueError(
            "Checkpoint tokenizer is not char-v1"
        )

    if checkpoint["tokenizer_chars"] != dataset.tokenizer.chars:
        raise ValueError(
            "Tokenizer vocabulary does not match checkpoint"
        )

    actual_hash = corpus_sha256(data_path)

    if checkpoint["corpus_sha256"] != actual_hash:
        raise ValueError(
            "Training text differs from the original corpus"
        )

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    if config.block_size != block_size:
        raise ValueError(
            "Block size differs from checkpoint"
        )

    if config.vocab_size != dataset.tokenizer.vocab_size:
        raise ValueError(
            "Vocabulary size differs from checkpoint"
        )

    return checkpoint, config


def train(args):
    torch.manual_seed(args.seed)

    device = get_device()

    dataset = TextDataset(
        file_path=args.data,
        block_size=args.block_size,
    )

    checkpoint = None
    start_step = 0

    if args.resume:
        checkpoint, config = load_checkpoint(
            checkpoint_path=args.resume,
            dataset=dataset,
            data_path=args.data,
            block_size=args.block_size,
        )

        start_step = int(checkpoint["step"])

        if args.steps <= start_step:
            raise ValueError(
                f"Target step {args.steps} must be "
                f"greater than checkpoint step {start_step}"
            )

    else:
        config = ModelConfig(
            vocab_size=dataset.tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=64,
            n_head=4,
            n_layer=2,
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

        # Restore RNG when available.
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint["torch_rng_state"]
            )
        else:
            print(
                "Note: Old checkpoint has no RNG state. "
                "Training continues, but is not an "
                "exact replay of uninterrupted training."
            )

        if (
            device.type == "cuda"
            and "cuda_rng_states" in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_states"]
            )

        # Do not silently change the original
        # learning rate when resuming.
        print("Optimizer state restored.")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("JSM Mini — Training")
    print("-------------------")
    print("Device:", device)
    print("Vocabulary:", config.vocab_size)
    print("Parameters:", f"{total_params:,}")
    print("Train tokens:", len(dataset.train_data))
    print("Val tokens:", len(dataset.val_data))
    print("Starting step:", start_step)
    print("Target step:", args.steps)

    model.train()

    for step in range(start_step + 1, args.steps + 1):
        x, y = dataset.get_batch(
            split="train",
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
                model,
                dataset,
                device,
                batch_size=args.batch_size,
            )

            print(
                f"Step {step:5d}/{args.steps} | "
                f"Train: {metrics['train']:.4f} | "
                f"Val: {metrics['val']:.4f}"
            )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=dataset.tokenizer,
        step=args.steps,
        data_path=args.data,
        output_path=args.output,
    )

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/saudi.txt",
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="artifacts/jsm_mini_char.pt",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.eval_interval < 1:
        parser.error("--eval-interval must be >= 1")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    if args.resume:
        if (
            Path(args.resume).resolve()
            == Path(args.output).resolve()
        ):
            parser.error(
                "Use a different output path when resuming "
                "to preserve the original checkpoint"
            )

    train(args)


if __name__ == "__main__":
    main()


import argparse

from src.trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Train NABRA Nano ASR"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--blank-bias",
        type=float,
        default=-2.0,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument("--train-source")
    parser.add_argument("--val-source")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--config", help="Versioned ASR model/features JSON")
    args = parser.parse_args()
    import json
    from pathlib import Path
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    if config and config.get("schema_version") != 1:
        parser.error("Unsupported config schema")

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        blank_bias=args.blank_bias,
        save_interval=args.save_interval,
        resume=args.resume,
        run_id=args.run_id,
        train_source=args.train_source, val_source=args.val_source, device_name=args.device,
        model_options=config.get("model"), feature_options=config.get("features"),
    )


if __name__ == "__main__":
    main()

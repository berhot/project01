
import argparse
from src.trainer import train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=10
    )
    parser.add_argument(
        "--batch-size", type=int, default=4
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--resume", default=None
    )
    parser.add_argument(
        "--run-id", default=None
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resume=args.resume,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

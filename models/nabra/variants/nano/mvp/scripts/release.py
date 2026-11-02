
import argparse

from src.packaging import package_release


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", required=True
    )
    parser.add_argument(
        "--version", required=True
    )
    parser.add_argument(
        "--max-cer", type=float, required=True
    )
    parser.add_argument(
        "--approve", action="store_true"
    )

    args = parser.parse_args()

    release_dir = package_release(
        checkpoint_path=args.checkpoint,
        version=args.version,
        max_cer=args.max_cer,
        approve=args.approve,
    )

    print("NABRA — Release")
    print("Version:", args.version)
    print("Directory:", release_dir)
    print("Status: PACKAGED")


if __name__ == "__main__":
    main()

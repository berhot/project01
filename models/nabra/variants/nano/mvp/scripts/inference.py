
import argparse

from src.inference import (
    get_device,
    load_checkpoint,
    load_release,
    transcribe,
)


def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument("--checkpoint")
    source.add_argument("--release")

    parser.add_argument(
        "--audio", required=True
    )

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    from src.runtime import select_device
    device = select_device(args.device)

    if args.release:
        model, tokenizer, features = load_release(
            args.release, device
        )
    else:
        model, tokenizer, features = load_checkpoint(
            args.checkpoint, device
        )

    text = transcribe(
        model, tokenizer, features, args.audio
    )

    print("NABRA — Speech Recognition")
    print("Audio:", args.audio)
    print("Device:", device)
    print("Transcript:", text)


if __name__ == "__main__":
    main()

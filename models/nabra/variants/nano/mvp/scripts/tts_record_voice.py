"""Collect a consented Arabic speaker corpus interactively; no model training."""
import argparse
from pathlib import Path

from src.tts.voice_recorder import (
    CORPUS_ROOT, check_take, export_index, microphone_session, save_take, start_session,
    validate_speaker,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--consent", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--export-index", type=Path, help="Export existing accepted takes; microphone not required")
    args = parser.parse_args()
    validate_speaker(args.speaker)
    folder, prompts, existing = start_session(args.speaker, args.prompts, args.consent)
    if args.export_index:
        print("Exported", export_index(folder, args.export_index, args.speaker), "recordings to", args.export_index)
        return
    device = int(args.device) if args.device is not None and args.device.isdigit() else args.device
    print(f"{len(existing)} completed; {len(prompts)-len(existing)} remaining. Ctrl+C preserves accepted takes.")
    try:
        with microphone_session(device) as microphone:
            for number, prompt in enumerate(prompts, 1):
                if f"{number:05d}" in existing:
                    continue
                while True:
                    print(f"[{number}/{len(prompts)}] {prompt}")
                    command = input("  Enter=record, s=skip, q=quit: ").strip().lower()
                    if command == "q":
                        return
                    if command == "s":
                        break
                    samples = microphone.take()
                    problem = check_take(samples)
                    if problem:
                        print("  Retake required:", problem)
                        continue
                    if input("  Enter=save, r=redo: ").strip().lower() == "r":
                        continue
                    save_take(folder, number, prompt, samples)
                    print("  Saved", f"{number:05d}.wav")
                    break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; accepted recordings preserved.")
    print("Saved recordings:", folder)
    print("Export with: python3 -m scripts.tts_record_voice --speaker", args.speaker,
          "--prompts", args.prompts, "--consent --export-index", folder / "index.jsonl")


if __name__ == "__main__":
    main()

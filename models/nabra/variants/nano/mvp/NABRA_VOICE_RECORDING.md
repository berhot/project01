# NABRA Nano — local microphone corpus recording

Adapted from the historical `berhot/berhot-ai/jsm-voice/scripts/record_voice_corpus.py`, preserving its prompt/record/retry/skip/quit/resume experience and single persistent microphone stream.

The old system is **not** transplanted wholesale: Whisper/Piper and Griffin-Lim remain excluded because they are not Nabra's neural models. This port uses Nabra paths, hashes and corpus preparation. The source microphone audio is preserved at **22,050 Hz PCM16**; `prepare_corpus` creates separate **16,000 Hz** copies.

## Setup (Mac)

```bash
cd /Users/farisalsaiari/Desktop/BERHOT-nabra-multi/models/nabra/variants/nano/mvp
python3 -m pip install sounddevice numpy
```

Do not change a dirty working tree to switch branches: use a new worktree for this PR. Use the shared project virtual environment. Grant Terminal microphone access in macOS Privacy & Security settings if prompted.

## Record and resume

```bash
python3 -m scripts.tts_record_voice \
  --speaker fahad \
  --prompts data/voice_prompts/saudi_starter.txt \
  --consent
```

Replace `fahad` with your own chosen speaker label. Using `--consent` asserts the speaker agrees to record their voice for model training; only record yourself or someone who has actually agreed. Repeat the same command to resume; completed prompts are skipped. Recording is local. No automatic cloud upload occurs. A different prompt list requires a different speaker/session label in this MVP.

The recorder displays one Arabic prompt at a time. Enter begins recording; Enter stops; enter saves or `r` retries; `s` skips; `q` quits. It rejects too short/long, too quiet or clipped audio. These heuristics don't prove transcription correctness or absence of noise: listen to your takes.

Recordings live under `storage/voice_corpus/nabra/<speaker>/` with WAVs, `speaker.json` consent receipt and `metadata.jsonl`. Do not push private voice recordings or large checkpoints to public GitHub.

## Export for training

```bash
ROOT=/Users/farisalsaiari/Desktop/BERHOT
python3 -m scripts.tts_record_voice \
  --speaker fahad --prompts data/voice_prompts/saudi_starter.txt --consent \
  --export-index "$ROOT/storage/voice_corpus/nabra/fahad/source-index-v1.jsonl"

python3 -m scripts.prepare_corpus \
  --index "$ROOT/storage/voice_corpus/nabra/fahad/source-index-v1.jsonl" \
  --output "$ROOT/storage/data/processed/audio/nabra-fahad-v1" \
  --val-fraction 0.2 --seed 42

python3 -m scripts.tts_preflight_multi \
  --train-source "$ROOT/storage/data/processed/audio/nabra-fahad-v1/train.jsonl" \
  --val-source "$ROOT/storage/data/processed/audio/nabra-fahad-v1/val.jsonl" --strict
```

With 30 prompts, this is a **pipeline pilot**, not a sufficient speech dataset. A later 100–300-recording corpus is a better small-scale experiment, but genuine intelligible speech will typically require considerably more diverse, clean hours of data. Recordings should have consistent microphone distance, no clipping, precise transcripts and valid permission.

## Compatibility with old jsm-voice corpus

Existing jsm-voice `storage/voice_corpus/<speaker>/metadata.jsonl` lacks the new `audio_sha256` requirement and must not be pointed at this new recorder as-is. Preserve the original corpus and implement a verified, explicit conversion rather than rewriting it.

Nabra's old diagnostic one-sentence checkpoints and release files are unchanged. Running this recorder does not perform training or certify model quality.

# NABRA Nano — held-out TTS listening protocol

This is a **diagnostic**, not evidence of understandable or natural speech.
The original recovered single-sentence speech model has not demonstrated
generalization to new sentences. This stage adds a repeatable, versioned Arabic
prompt set and signal measurements, with per-prompt WAV files for human review.

## Run later with a locally available checkpoint

From `models/nabra/variants/nano/mvp` (with the project's virtualenv):

```bash
python3 -m unittest tests.test_tts_unseen_quality -v
python3 -m unittest discover -s tests -v

ROOT=$(git rev-parse --show-toplevel)
python3 -m scripts.tts_unseen_quality \
  --checkpoint "$ROOT/storage/artifacts/nabra/tts/nano/mvp/YOUR_RUN/checkpoints/step_000500.pt" \
  --prompts data/eval/arabic_unseen_v1.txt \
  --output "$ROOT/storage/artifacts/nabra/tts/nano/mvp/unseen-eval-001" \
  --train-manifest "$ROOT/storage/data/processed/audio/YOUR_CORPUS/train.jsonl"
```

Replace the example checkpoint and corpus paths with the **actual paths on your
Mac**. The output directory must not already exist. The script will not touch
the training set, checkpoint, release or existing WAVs.

Use `--train-manifest` once per relevant JSONL training manifest. This rejects
**exact normalized text** overlaps only. It does not detect semantic paraphrases,
hidden data leakage, speaker memorization or overlap with unlisted training data.
Without training manifests, the report explicitly states that holdout overlap
has **not been checked**.

For each text, the tool saves a WAV, sample metrics (RMS, clipping and length),
audio hash and a JSON report. Signal checks do **not** establish whether the
words can be understood. A human Arabic listener should review all WAVs
blindly, transcribe the words heard, and record intelligibility/errors and
pronunciation notes. Human scoring is a separate step and is **not** automated.

The prompts are authored for diagnostics and must never be included in future
training. Run IDs/output directories should be unique. Do not claim passing
signal checks alone means production-ready quality.

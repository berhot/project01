# NABRA — Unseen Arabic TTS evaluation

This evaluates **diagnostic recovery checkpoints**, not the old noisy TTS 0.1.0 release. No weights or releases are changed. It is a **read-only evaluation** of checkpoints and generates new WAVs and reports under a new output directory.

## Run on your Mac

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
source .venv/bin/activate
cd models/nabra/variants/nano/mvp
python3 -m unittest tests.test_tts_unseen_evaluation -v

ROOT=/Users/farisalsaiari/Desktop/BERHOT
BASE="$ROOT/storage/artifacts/nabra/tts/nano/mvp"
find "$BASE" -path '*/checkpoints/*.pt' | grep -E 'recovery-(acoustic|vocoder)'
```

Choose the actual **final refined acoustic** checkpoint and the paired **phase-trained spectral vocoder** checkpoint shown in your quality recovery report; check the paths really exist. Examples from the report:

```bash
python3 -m scripts.tts_evaluate_unseen \
  --acoustic-checkpoint "$BASE/recovery-acoustic-waveform-001/checkpoints/step_002500.pt" \
  --vocoder-checkpoint "$BASE/recovery-vocoder-phase-001/checkpoints/step_004000.pt" \
  --output "$BASE/unseen-evaluation-001" \
  --device cpu --max-frames 800
```

If either path is absent, locate the actual saved pair first; do not silently substitute unpaired checkpoints. A new output folder is mandatory, and the script refuses to overwrite existing evaluations.

The script generates six Arabic prompts (seen reference, short unseen, question, longer text, numbers, colloquial dialect). It saves `.wav` files, `evaluation.json` with hashes, duration, RMS, peak, clipping, stop-prediction and attention diagnostics, and a human listening checklist.

## Interpretation

Listen to every WAV (`afplay file.wav` on macOS). Distinguish (1) recognizable words, (2) correct entire phrase, (3) hiss/artifacts, (4) early stop/truncation, (5) stability across varied text, and (6) Saudi dialect and numerals. A nonzero RMS or low spectral error **cannot** establish speech intelligibility. Report whether unseen outputs are acceptable before discussing more parameters.

This is an **unseen-text** test, **not** unseen-speaker generalization, because the recovery data came from only one synthetic voice recording. If the results fail, collect diverse, consented and licensed Arabic recordings first; do not promote the one-utterance overfit experiment to a release.

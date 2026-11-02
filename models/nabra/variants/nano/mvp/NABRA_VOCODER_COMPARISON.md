# NABRA — Controlled vocoder comparison

The standard TTS checkpoint's ground-truth-Mel vocoder sounds like noise on
both available synthetic training clips (listener-reported). The earlier recovery
experiments produced a more intelligible **single-sentence overfit** result.
This experiment compares both vocoders on the **same recording and Mel input**.

**It does not train a new model or claim generalization.** The comparison
enforces original-source SHA-256 equality and identical Mel feature configuration.
A checkpoint trained on a different source recording is rejected.

## Run on your Mac

Check out the PR branch and run the test suite first. Then, from
`models/nabra/variants/nano/mvp`:

```bash
ROOT=/Users/farisalsaiari/Desktop/BERHOT
python3 -m scripts.tts_compare_vocoders \
  --standard-checkpoint "$ROOT/storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002/checkpoints/step_000500.pt" \
  --recovery-checkpoint "$ROOT/storage/artifacts/nabra/tts/nano/mvp/recovery-vocoder-phase-001/checkpoints/step_004000.pt" \
  --split train --record-index 0 \
  --output "$ROOT/storage/artifacts/nabra/tts/nano/mvp/vocoder-comparison-001" \
  --device cpu
```

The chosen recovery checkpoint is an **existing local file**, not a Git-tracked
weight. If a source hash mismatch is reported, this is intentional—choose an
appropriately matched recovery checkpoint instead of bypassing the check.

The new output directory contains `reference.wav`, `standard.wav`,
`recovered.wav`, and `report.json`.

```bash
afplay "$ROOT/storage/artifacts/nabra/tts/nano/mvp/vocoder-comparison-001/reference.wav"
afplay "$ROOT/storage/artifacts/nabra/tts/nano/mvp/vocoder-comparison-001/standard.wav"
afplay "$ROOT/storage/artifacts/nabra/tts/nano/mvp/vocoder-comparison-001/recovered.wav"
```

Compare intelligibility by listening and check `report.json` for waveform
L1, aligned waveform SNR, spectral error and model parameter counts.
An unaligned negative waveform SNR is not itself conclusive evidence of noise.
The recovered model may outperform baseline because it was intensively
trained on that **one recording**; this does not establish a generalizing vocoder.

## Decision after listening

If the recovered output is intelligible from the same reference Mel while the
standard output is noise, prioritize adapting its phase-aware spectral
vocoder and training objective to a *multi-utterance* dataset, with a
speaker/source-separated validation split. Do **not** copy single-clip weights
or overwrite published release checkpoints. Evaluate held-out clips and
naturalness independently of objective scores. If both outputs are noise,
investigate the source audio, Mel extraction and vocoder checkpoint pairing.

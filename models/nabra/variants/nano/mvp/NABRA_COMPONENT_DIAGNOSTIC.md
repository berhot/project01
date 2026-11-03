# NABRA TTS — Component isolation experiment

The previous evaluation produced ten 4-second WAVs; the listener reported
noise in every file. Three additional examples all reached the 400-frame
generation cap. Attention moves along text, but the stop head probabilities
remain below 0.5. This experiment separates three signals:

1. **Teacher-forced Mel L1** on recorded text/audio
2. **Free-running Mel L1**, attention progress, length and stop behavior
3. **Vocoder reconstruction** from **ground-truth** Mel features, including
   aligned waveform SNR and a listening WAV

This is read-only with respect to input recordings and checkpoints. It creates
only a fresh output directory with `report.json` and optional WAVs. It does
not retrain or change any released model. Input audio hashes are checked.

From `models/nabra/variants/nano/mvp`, with the existing Python environment:

```bash
ROOT=/Users/farisalsaiari/Desktop/BERHOT
python3 -m scripts.tts_component_diagnostic \
  --checkpoint "$ROOT/storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002/checkpoints/step_000500.pt" \
  --split train --count 2 --max-frames 400 \
  --output "$ROOT/storage/artifacts/nabra/tts/nano/mvp/component-diagnostic-001" \
  --device cpu
```

**Important:** This command is for the standard `task=tts` checkpoint,
not the specialized recovery-stage checkpoints. The saved training manifest
must still refer to the real original WAVs on your Mac. The diagnostics check
each referenced WAV against its saved SHA-256.

Review `report.json`. Then compare `reference_vocoder_001.wav` with the
corresponding original training recording. If even ground-truth-Mel vocoding
sounds like noise, prioritize the vocoder. If it sounds like speech but
free-running synthesis is noisy, focus on acoustic prediction, text/audio
alignment, and inference drift. If the teacher-forced loss is high, fix that
before drawing conclusions from free-running output.

**Caution:** A lower Mel loss, better SNR, or progressing attention is not
evidence of natural or intelligible speech. Output lengths may differ from
reference lengths and L1 only compares overlapping frames. Listening tests
and diverse, consented or appropriately licensed recordings remain necessary.

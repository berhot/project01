# Nabra multi-sentence TTS pilot

This branch keeps Nabra's existing acoustic model and neural vocoder. It adds deterministic shuffled utterance selection and a corpus-readiness command. It does not train, release, or modify model weights.

## Dataset
Collect 100–300 accurately transcribed, licensed or consented recordings from one speaker for an initial pilot. Use multiple distinct Arabic sentences. Preserve original audio.

Create an index with audio, transcript, speaker_id, license/consent and provenance. Then run from the Nabra MVP:

```bash
ROOT=/Users/farisalsaiari/Desktop/BERHOT
python3 -m scripts.prepare_corpus --index "$ROOT/intake/speech/raw/index.jsonl" --output "$ROOT/storage/data/processed/audio/nabra-multi-v1" --val-fraction 0.2 --seed 42
python3 -m scripts.tts_preflight_multi --train-source "$ROOT/storage/data/processed/audio/nabra-multi-v1/train.jsonl" --val-source "$ROOT/storage/data/processed/audio/nabra-multi-v1/val.jsonl" --strict
```

The preflight default thresholds (20 train recordings, 10 distinct train prompts) are minimum software checks, not a claim of sufficient data for intelligible speech. It also requires every validation speaker to appear in training, since the model uses a trained speaker embedding.

## Training

```bash
python3 -m scripts.tts_train --train-source "$ROOT/storage/data/processed/audio/nabra-multi-v1/train.jsonl" --val-source "$ROOT/storage/data/processed/audio/nabra-multi-v1/val.jsonl" --steps 500 --save-interval 25 --device cpu --threads 2
```

Each epoch now visits all training clips in a reproducible shuffled order. New checkpoints resume with the same seed and ordering. For exact historical reproducibility, do not resume pre-change fixed-order checkpoints into this new code.

Evaluate held-out audio metrics and listen to generated unseen text. Separately assess the vocoder using ground-truth Mel features. Do not promote the previous one-sentence overfit diagnostic into an Arabic-generalizing release. No historical weights or release files are touched by this branch.

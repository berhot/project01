# NABRA Nano — End-to-end training guide

This guide documents the existing **from-scratch Nabra** ASR and TTS architecture. The compatible data practices were compared with [BERHOT's historical jsm-voice repository](https://github.com/berhot/berhot-ai/tree/main/jsm-voice): preserve original audio, trace source/derived hashes, keep repeated prompts in the same split, validate before training, and finish at checkpoint boundaries. **Do not import Whisper/Piper or the JSM-voice duration/Griffin-Lim system.**

## 1. Setup

From the BERHOT root, activate the project's Python environment and enter the Nabra MVP:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
source .venv/bin/activate
cd models/nabra/variants/nano/mvp
python3 -m unittest tests.test_corpus_preparation -v
python3 -m unittest discover -s tests -v
ffmpeg -version
```

The new `prepare_corpus` command requires FFmpeg. It is for **new, separately licensed speech sources**; the existing two training and one validation synthetic samples do not establish production quality.

## 2. Prepare a source index

Keep original audio under shared `BERHOT/intake/speech/` or another source-controlled/data location with appropriate permissions. Do **not** modify the original files.

Create a JSONL index, for example `intake/speech/raw/index.jsonl`. Each record contains a path relative to the index, an exact checked transcript, and source metadata:

```json
{"audio":"speaker01/001.wav","transcript":"مرحبا كيف حالك","speaker_id":"speaker01","source_url":null,"license":null,"consent":null,"provenance":{"source":"microphone","collection_date":"2026-11-02"}}
{"audio":"speaker02/002.mp3","transcript":"صباح الخير","speaker_id":"speaker02","source_url":null,"license":null,"consent":null,"provenance":{"source":"licensed recording"}}
```

Replace placeholders with actual evidence. Do not treat missing license or recording consent as permission for commercial training. Metadata is copied into prepared records; source and derived files get SHA-256 identities.

## 3. Standardize and split safely

```bash
ROOT=/Users/farisalsaiari/Desktop/BERHOT
python3 -m scripts.prepare_corpus \
  --index "$ROOT/intake/speech/raw/index.jsonl" \
  --output "$ROOT/storage/data/processed/audio/nabra-v1" \
  --val-fraction 0.20 --seed 42
```

This creates a **new** output directory containing `audio/*.wav` (16 kHz, mono, PCM16), `train.jsonl`, `val.jsonl`, and `report.json`. It refuses an existing destination, preserves each source's bytes, verifies hashes after conversion, and groups repeated normalized text prompts into the same split. The target must include at least **two different prompts**. Conversion failures leave an `INCOMPLETE` marker: discard or quarantine that output; choose a new version and rerun.

The split is **prompt-disjoint, not speaker-disjoint**. When evaluating generalization across speakers, create truly speaker-disjoint partitions before declaring independent validation.

## 4. Preflight

```bash
python3 -m scripts.preflight_corpus \
  --train "$ROOT/storage/data/processed/audio/nabra-v1/train.jsonl" \
  --val "$ROOT/storage/data/processed/audio/nabra-v1/val.jsonl"
```

The checks use Nabra's existing WAV/hash reader and reject duplicate waveform hashes across splits, normalized repeated prompts and invalid duration/format. If you intentionally created speaker-separated partitions, add `--require-speaker-disjoint`. Check transcriptions, tokenizer/vocabulary coverage, speaker permissions, quiet clips, and clipping manually or using Nabra's ASR/TTS dataset inspection before model training.

## 5. Train ASR (speech to text)

Nabra ASR is a log-Mel frontend, convolutional downsampling, BiGRU and CTC model. Confirm supported options via `python3 -m scripts.train --help`. One typical command:

```bash
python3 -m scripts.train --train-source "$ROOT/storage/data/processed/audio/nabra-v1/train.jsonl" \
  --val-source "$ROOT/storage/data/processed/audio/nabra-v1/val.jsonl" \
  --epochs 100 --device cpu
```

Record the printed run ID and checkpoint path. Inspect training versus held-out CER/WER, the CTC blank rate, and several unseen genuine recordings. Current reported CER on one synthetic validation clip is not evidence of broad accuracy.

## 6. Evaluate ASR / inference

```bash
python3 -m scripts.evaluate --help
python3 -m scripts.inference --help
python3 -m scripts.asr_microphone --help
```

Use the confirmed CLI flags for `--checkpoint` or `--release` and `--audio`. `asr_microphone` uses buffered bidirectional recognition; it is not a causal streaming network. Preserve held-out test audio from training and model selection.

## 7. Train TTS (text to waveform)

Nabra TTS uses an autoregressive attention-based acoustic model and a **learned neural vocoder**. It does not use Whisper/Piper, a duration-predictor/Griffin-Lim branch, or source waveform playback.

```bash
python3 -m scripts.tts_train \
  --train-source "$ROOT/storage/data/processed/audio/nabra-v1/train.jsonl" \
  --val-source "$ROOT/storage/data/processed/audio/nabra-v1/val.jsonl" \
  --steps 500 --save-interval 25 --device cpu --threads 2
```

Nabra writes new run directories under `storage/artifacts/nabra/tts/nano/mvp`, with atomic checkpoint replacement. Use a bounded number of steps and verify the actual checkpoint. **New behavior:** SIGINT (Ctrl+C) or SIGTERM requests stop; the trainer finishes the current step and proceeds until it has written its next checkpoint, then exits with a summary that records `stop_requested` and achieved step. For immediate force-kill or power loss, resume from the last completed checkpoint; never assume an interrupted write succeeded.

For resume:

```bash
python3 -m scripts.tts_train --resume "/absolute/path/to/step_000500.pt" \
  --train-source "$ROOT/storage/data/processed/audio/nabra-v1/train.jsonl" \
  --val-source "$ROOT/storage/data/processed/audio/nabra-v1/val.jsonl" \
  --steps 1000 --save-interval 25 --device cpu
```

Resume requires the **same corpus fingerprint, model config, speaker mapping and tokenizer**, and a new total target step greater than the checkpoint's step. Existing releases remain immutable; do not substitute a diagnostic checkpoint for an ordinary training checkpoint.

## 8. Evaluate TTS and listen

```bash
python3 -m scripts.tts_evaluate --help
python3 -m scripts.tts_inference --help
python3 -m scripts.tts_inference --release 0.1.0 \
  --text "مرحبا أنا أتعلم الذكاء الاصطناعي" \
  --output "$ROOT/storage/artifacts/nabra/tts/nano/mvp/voice-check.wav" --device cpu
afplay "$ROOT/storage/artifacts/nabra/tts/nano/mvp/voice-check.wav"
```

That old `0.1.0` release was confirmed as **noise**, and is kept for audit. Later **diagnostic-only** recovery checkpoints learned one recognizable sentence, still with hiss, documented in `NABRA_TTS_QUALITY_RECOVERY_REPORT.md`. This proves single-example learning, **not** general Arabic voice synthesis. Evaluate unseen text, new speakers, objective signal quality and intelligibility listening independently before releasing improvements.

## 9. Auditing and next scale-up

Keep source index, licensing, conversion report, derived hashes, train/validation/test split provenance, training config, seed, checkpoint hashes and human listening records. For a real Arabic corpus, collect diverse licensed human speakers, accents and phonetic coverage, and enforce speaker/source-disjoint tests when those claims matter.

Nabra releases and large checkpoints belong under `storage/` or remote artifact storage rather than raw Git commits. Do not overwrite historical releases.

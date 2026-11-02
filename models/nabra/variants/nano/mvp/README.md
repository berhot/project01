# NABRA Nano voice MVP

Experimental ASR, from-scratch TTS, and bounded-buffer microphone recognition.
Read the current [TTS quality recovery report](NABRA_TTS_QUALITY_RECOVERY_REPORT.md)
and the historical [MVP completion report](NABRA_MVP_COMPLETION_REPORT.md).
These models are not production-quality speech systems.

The current shared corpus has two training clips and one validation clip. All
three are synthetic Arabic speech generated earlier with macOS `say`; their
provenance is recorded in `intake/speech/synthetic-samples.json`. The TTS model
implemented here generates its own waveform from neural weights and text. It
does not call `say`, use a speech API, or replay the training recordings.

## Later: record your own Saudi Arabic voice for multi-sentence TTS

**Status: prepared, not yet recorded or trained.** The new recording workflow was
adapted from `berhot/berhot-ai/jsm-voice`. See
[NABRA_VOICE_RECORDING.md](NABRA_VOICE_RECORDING.md) for details and
[NABRA_MULTISENTENCE_TTS.md](NABRA_MULTISENTENCE_TTS.md) for training notes.
This section is for when you're ready; it does **not** change old releases,
diagnostic checkpoints, or the original BERHOT working directory.

### 1. Check out the voice-recorder worktree (once)

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git fetch origin
git worktree add ../BERHOT-nabra-record \
  -b review/nabra-voice-record \
  origin/feat/nabra-voice-recorder-port-20261102
cd ../BERHOT-nabra-record/models/nabra/variants/nano/mvp
/Users/farisalsaiari/Desktop/BERHOT/.venv/bin/python -m unittest tests.test_tts_voice_recorder -v
/Users/farisalsaiari/Desktop/BERHOT/.venv/bin/python -m unittest discover -s tests -v
```

If the worktree already exists, enter it and use `git pull --ff-only`
instead of creating it again. These steps work with the PR branch until it is merged;
afterwards use the updated project checkout.

### 2. Prepare the microphone (once)

```bash
source /Users/farisalsaiari/Desktop/BERHOT/.venv/bin/activate
python3 -m pip install sounddevice
```

Grant Terminal microphone permission on macOS when prompted. NumPy is also
required (already used elsewhere in Nabra).

### 3. Record; resume with the same command later

```bash
python3 -m scripts.tts_record_voice \
  --speaker fahad \
  --prompts data/voice_prompts/saudi_starter.txt \
  --consent
```

Replace `fahad` with your chosen speaker identifier. Only record yourself or
someone who explicitly agreed to voice-model training. The starter set has
30 original Arabic sentences. At each prompt, Enter starts/stops recording;
save the take, use `r` to redo, `s` to skip, or `q` to quit.
Repeat the same command later to resume without overwriting accepted takes.
Your microphone originals are stored locally at
`BERHOT/storage/voice_corpus/nabra/<speaker>/` (22,050 Hz PCM16).
Do not commit private voice recordings to this public GitHub repository.

### 4. Export the recordings and prepare the 16 kHz training corpus

Run from the recording worktree's Nabra MVP directory:

```bash
ROOT=/Users/farisalsaiari/Desktop/BERHOT
python3 -m scripts.tts_record_voice \
  --speaker fahad \
  --prompts data/voice_prompts/saudi_starter.txt \
  --consent \
  --export-index "$ROOT/storage/voice_corpus/nabra/fahad/source-index-v1.jsonl"

python3 -m scripts.prepare_corpus \
  --index "$ROOT/storage/voice_corpus/nabra/fahad/source-index-v1.jsonl" \
  --output "$ROOT/storage/data/processed/audio/nabra-fahad-v1" \
  --val-fraction 0.2 --seed 42

python3 -m scripts.tts_preflight_multi \
  --train-source "$ROOT/storage/data/processed/audio/nabra-fahad-v1/train.jsonl" \
  --val-source "$ROOT/storage/data/processed/audio/nabra-fahad-v1/val.jsonl" \
  --strict
```

The export and conversion preserve source WAVs and refuse to overwrite
existing output versions. The built-in 30 prompts are a pipeline pilot;
**100–300 varied, accurately transcribed consented recordings** are a more
useful first training experiment, and still not a guarantee of intelligible
speech. For additional sentences, plan a separate prompt-list/session
workflow rather than changing the list in an already-started session.

### 5. Train a fresh experimental multi-sentence Nabra TTS run

Only after corpus preflight passes on a sufficiently varied dataset:

```bash
python3 -m scripts.tts_train \
  --train-source "$ROOT/storage/data/processed/audio/nabra-fahad-v1/train.jsonl" \
  --val-source "$ROOT/storage/data/processed/audio/nabra-fahad-v1/val.jsonl" \
  --steps 500 --save-interval 25 --device cpu --threads 2
```

Evaluate the checkpoint on held-out recordings and listen to new Arabic
sentences before deciding whether to scale training. The recovered
single-sentence overfit checkpoints are **not** a general-purpose TTS release.
No actual speaker recording, real-data training, or intelligibility verification
has been performed as part of this documentation change.

## TTS quality recovery — 2026-11-02

The original TTS `0.1.0` output was heard as noise. Bounded recovery experiments
isolated vocoder reconstruction failure and a teacher-forcing/inference gap. An
optional learned complex-spectral vocoder head, phase-sensitive supervision, and
free-running acoustic training recovered the existing training sentence. A final
acoustic refinement through the frozen vocoder reduced the background hiss.

The user listened to the final text-to-speech output and confirmed:

> words clear , hiss reduced not 100% quality

**Single-sentence Arabic intelligibility: PASS. Release readiness: FAIL.**
Residual hiss remains, unseen-text pronunciation is unverified, and this is an
overfit proof on one existing synthetic recording. The separate ground-truth-Mel
vocoder sample still awaits direct human listening. No release was replaced or created.

| Check | Observed result |
| --- | --- |
| Ground-truth-Mel vocoder | 22.68 dB aligned SNR, waveform MAE 0.00700 |
| Free-running acoustic learning | Mel MAE fell from 0.32648 to 0.05232 in Experiment B; 440 predicted / 438 target frames |
| Refined full neural synthesis | 4.40 seconds, 16.26 dB aligned SNR, no clipping |
| Noise refinement | Quiet-region residual RMS fell about 30%; user reports reduced hiss |
| Full regression suite | 72 passed, 0 failed, 0 skipped with optional MPS checks enabled |
| Preservation | 137 historical data/artifact/release files unchanged; original TTS WAV remains byte-identical |

From this MVP directory:

```bash
python -m pip install -r requirements-diagnostics.txt
ROOT=$(git rev-parse --show-toplevel)
ART="$ROOT/storage/artifacts/nabra/tts/nano/mvp"
afplay "$ART/recovery-full-002/samples/tts_full_overfit.wav"
afplay "$ART/recovery-vocoder-phase-001/samples/vocoder_reconstruction.wav"
```

The complete [recovery report](NABRA_TTS_QUALITY_RECOVERY_REPORT.md) records the
root-cause evidence, every bounded experiment, checkpoint/audio hashes, human
feedback, measured timings, playback paths, tests and remaining limitations.
Use `scripts.tts_overfit_vocoder`, `scripts.tts_overfit_acoustic`, and
`scripts.tts_overfit_full` for diagnostics; all support `--help`. Training supports
explicit step limits, seed, learning rate, device, periodic samples and resume.
Use new run IDs for additional experiments. The source and report are versioned;
generated WAVs, tensors, figures and checkpoints remain in local artifact storage.

## Historical implementation report — before TTS quality recovery

The following records the earlier delivery; current TTS findings are above.

| Workstream | Delivered | Verification |
| --- | --- | --- |
| ASR | Experimental release `0.2.0`, checkpoint validation, evaluation and release integrity | Exact checkpoint/export tensor and transcription equivalence; validation CER 0.411765 and WER 1.333333 |
| Tests | Ingestion, features, tokenizer, model, training/resume, releases, TTS, microphone buffering and human-speech integration | Final MPS-enabled run: 62 passed; CPU run: 60 passed, 2 optional MPS checks skipped |
| TTS | Independent acoustic model, neural vocoder, training/resume, evaluation, WAV inference and experimental release `0.1.0` | 500 training steps; 4.76-second Arabic WAV; exact checkpoint/release waveform equivalence |
| Microphone ASR | Device selection, bounded buffers, silence detection, partial/final events and latency reports | Recorded-input checks and live capture passed; live spoken transcription still requires observation |

The released models were trained on two synthetic Arabic clips totaling 9.3588
seconds. Three separately cached human recordings were used for integration
checks only. These measurements establish working software paths, not production
speech quality. Buffered bidirectional recognition is not causal streaming.

**Verification verdict: NOT READY — REMAINING ENGINEERING WORK.** An Arabic
listener still needs to assess TTS intelligibility, and a live voiced microphone
session must be observed. Neither check is marked complete. Historical releases,
checkpoints and the existing input recordings were preserved byte for byte during
the implementation. The detailed [completion report](NABRA_MVP_COMPLETION_REPORT.md)
contains the A–M requirement scorecard, commands, metrics, hashes, limitations,
and remaining verification steps.

Git delivery includes this MVP's source, configs, tests and documentation, plus
three small synthetic speech fixtures and their provenance in `intake/speech/`.
Generated weights, checkpoints, downloaded human audio, local audit logs and
machine-specific intake JSONL manifests are not included in this commit. Release
paths and generated WAV links describe local outputs; a fresh clone must train
and package models or retrieve those artifacts separately. No R2 upload or model
deployment has been performed.

### Git delivery record

- Implementation commit: [`1171e61`](https://github.com/berhot/project01/commit/1171e61e4e3457395c754877af1392ac009fcb66).
- Destination: `origin/main` at `https://github.com/berhot/project01.git`.
- Push result on 2026-11-02 (Asia/Riyadh): successful; GitHub accepted the update
  from `2393b08` to `1171e61`, and local `main` tracks `origin/main`.
- This delivery record is a follow-up documentation commit to the implementation
  commit above. Test results above refer to the completed implementation run;
  subsequent changes only added documentation.
- Local generated release bundles and absolute-path intake manifests remain
  untracked and were not published to GitHub.

## Setup and tests

From the BERHOT root:

```bash
source .venv/bin/activate
python -m pip install -r models/nabra/variants/nano/mvp/requirements.txt
cd models/nabra/variants/nano/mvp
python -m unittest discover -s tests -v
```

Optional accelerator checks on Apple Silicon:

```bash
NABRA_TEST_MPS=1 python -m unittest discover -s tests -v
```

A separately cached human-recorded integration fixture is available from Google
FLEURS Arabic under CC BY 4.0. To fetch or verify the three hash-pinned files:

```bash
python -m scripts.fetch_test_speech
```

The explicit downloader needs `curl` and `ffmpeg`. Tests never download data;
the two human-speech tests skip if this optional cache is absent. Attribution and
original downloaded audio are retained in `storage/cache/datasets/fleurs-ar-smoke-v1/`.
The fixture repartitions three upstream validation samples solely for engineering
checks; it is not an official FLEURS benchmark. Shared intake is unchanged.

Tests write temporary checkpoints/releases into temporary directories. Existing
speech files are only read. Controlled synthetic waveforms are used for DSP unit
tests; they are not evidence of Arabic recognition. No external models are downloaded.

## Layout and data contract

- `src/`: ASR implementation, shared utilities, buffered recognition, and independent `tts/` modules.
- `scripts/`: CLI entry points; `configs/`: versioned model and stream configurations.
- `tests/`: unittest unit, regression, and end-to-end tests.
- `BERHOT/intake/speech/{train,val}`: shared speech intake, not a Nabra-specific subdirectory.
- `BERHOT/storage/artifacts/nabra/nano/mvp/`: ASR runs.
- `BERHOT/storage/artifacts/nabra/tts/nano/mvp/`: TTS runs.
- `BERHOT/storage/releases/nabra/nano/<version>/`: ASR releases, preserving existing paths.
- `BERHOT/storage/releases/nabra/tts/nano/<version>/`: independent TTS releases.

Each input WAV must be mono 16 kHz PCM16, 0.25–20 seconds, paired with a UTF-8
transcript using the same basename. Transcripts must contain the words actually
spoken. No automatic transcript editing is performed. Training and validation
must not contain duplicate audio hashes. Speaker-disjoint and source-disjoint
splits require a real corpus with those identities supplied.

ASR `--train-source`/`--val-source` and TTS equivalents accept paired directories
or JSONL manifests. Each manifest record contains `audio`, `transcript`, and
preferably `audio_sha256`; relative audio paths resolve relative to the manifest.
Generated manifests include duration, sample rate, and provenance hashes. TTS
also preserves `speaker_id` and `provenance` (source type, license, consent/source
references when known). Unknown identities and licenses remain null. The
`<unspecified>` conditioning slot is not an invented speaker identity.

The MVP rejects unsupported sample rates instead of silently resampling corpus
files. Convert into a new file if needed, preserving the original:

```bash
ffmpeg -n -i original.wav -ar 16000 -ac 1 -c:a pcm_s16le converted.wav
```

## ASR

```bash
python -m scripts.inspect
python -m src.features
python -m scripts.train --config configs/asr_nano.json --epochs 100 --device auto
```

Use `--resume PATH --epochs NEW_TOTAL` to resume into a new run. The optimizer
and random state are restored; dataset, tokenizer, and feature mismatches are
rejected. An explicit model config must also match. The best checkpoint is
selected by validation CER, then validation loss, not by the last epoch.

The verified improved checkpoint is:

```bash
BERHOT_ROOT=$(git rev-parse --show-toplevel)
CHECKPOINT="$BERHOT_ROOT/storage/artifacts/nabra/nano/mvp/nabra-20261102T121628Z-f4747217/checkpoints/epoch_0075.pt"
python -m scripts.evaluate --checkpoint "$CHECKPOINT" --device cpu
python -m scripts.inference --checkpoint "$CHECKPOINT" --audio "$BERHOT_ROOT/intake/speech/train/clip001.wav" --device cpu
python -m scripts.inference --release 0.2.0 --audio "$BERHOT_ROOT/intake/speech/val/clip101.wav" --device cpu
```

`0.2.0` is already packaged with an explicit experimental CER gate of `0.5`.
Its observed validation CER is `0.4117647058823529` on one synthetic clip. This
is not an estimate of real-speaker generalization. Existing releases cannot be
overwritten. New releases use `scripts.release --checkpoint ... --version ...
--max-cer ... --approve` and require a hash-bound evaluation report.

The preserved `python -m scripts.overfit_ctc` command runs a one-recording diagnostic.
Its implementation lives in `src/overfit.py`; successful memorization is not a
recognition-quality assessment.

## TTS

The acoustic model uses a character encoder, autoregressive decoder, learned
location-aware attention, reduction blocks, and a learned stop head. No forced
character durations are fabricated. Text normalization applies Unicode NFC and
whitespace normalization and preserves Arabic diacritics. A fixed alphabet
allows supported unseen text, but unseen characters/words have not necessarily
been learned from the tiny corpus.

TTS uses absolute log-Mel targets with fixed scaling, separate from ASR's
per-utterance feature normalization. Its compact convolutional vocoder upsamples
Mel frames by 160 and learns waveform L1 plus multi-resolution STFT losses.
There is no pretrained vocoder or adversarial discriminator. This is a small
research MVP, not an implementation of a commercial TTS architecture.

```bash
BERHOT_ROOT=$(git rev-parse --show-toplevel)
# Choose a new output directory; existing manifests are not overwritten.
python -m scripts.tts_inspect --output "$BERHOT_ROOT/storage/data/processed/audio/tts-inspection-v1"
python -m scripts.tts_train --config configs/tts_nano.json --steps 500 --device cpu --threads 2
```

Training accepts `--train-source`, `--val-source`, `--resume`, `--seed`, and
`--run-id`. `--steps` is the total optimizer-step target. Training resumes into
a new run and validates dataset fingerprints, tokenizer, speakers, and configs.
The current trainer optimizes one variable-length recording per step; the
collator/model also support masked batches, but multi-GPU training is not implemented.

The completed TTS run and release can be used now:

```bash
TTS_RUN="$BERHOT_ROOT/storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002"
python -m scripts.tts_inference --release 0.1.0 \
  --text 'مرحبا بكم في نبْرة للذكاء الاصطناعي' \
  --output "$TTS_RUN/samples/new-arabic-example.wav" --device cpu
```

Output files are never silently overwritten. Metadata alongside each WAV records
whether the stop head fired or the length cap was reached. Use `--max-frames` to
bound synthesis (100 Mel frames per second). The verified sample stopped at
4.76 seconds with the default 800-frame limit. Intelligibility remains unverified.

For a new checkpoint:

```bash
python -m scripts.tts_evaluate --checkpoint /path/to/checkpoint.pt --device cpu
python -m scripts.tts_release --checkpoint /path/to/checkpoint.pt \
  --report /path/to/evaluation.json --version 0.2.0 \
  --max-mel-l1 1.0 --max-spectral-loss 5.0 --approve
```

These numerical gates are explicitly engineering gates, not intelligibility or
production certification. TTS `0.1.0` already exists and is immutable. Acoustic
weights, vocoder weights, tokenizer, feature/model configs, speakers, evaluation,
and SHA-256 manifest are included. Hashes detect corruption; manifests are not
cryptographically signed attestations.

## Buffered microphone ASR

The existing bidirectional GRU needs future context. This implementation is
**buffered near-real-time recognition, not a causal streaming encoder**. It
re-decodes only the active utterance at a configurable interval. A silence event
or the maximum segment duration finalizes it. Hard boundaries may split words;
there is no cross-segment acoustic state or overlap decoder.

```bash
python -m scripts.asr_microphone --list-devices
python -m scripts.asr_microphone --release 0.2.0 \
  --microphone 'MacBook Air Microphone' --config configs/asr_buffered.json
```

Select the enumerated device by name or current index; indexes can change. Capture
requests 16 kHz mono float audio through PortAudio/CoreAudio and reports an error
if that format is unavailable. Microphone permission must be enabled for the
terminal/application. `Ctrl-C` drains the bounded input queue and emits a final
segment. `--seconds 10` limits capture time.

Recorded audio can be delivered incrementally without a microphone:

```bash
python -m scripts.asr_stream --release 0.2.0 \
  --audio "$BERHOT_ROOT/intake/speech/train/clip001.wav" \
  --chunk-ms 100 --config configs/asr_buffered.json
```

Events contain `segment_id`, `kind`, `text`, `stable_prefix`, and decode latency.
Replace partial text for a segment; commit its final text once. The stable prefix
is agreement with the previous partial, not a guarantee that later text cannot
change. Capture uses a 32-chunk queue, inference uses an eight-second default
segment, and the CLI retains at most 128 preview events. `--output report.json`
also writes the complete event history incrementally to `report.json.events.jsonl`.
Queue overflow fails explicitly rather than silently dropping/reordering audio.

## Limits and next steps

No production-quality Arabic ASR/TTS claim is made. The current corpus uses a
single synthetic voice; dialect, gender, speaker, and recording-environment
coverage are absent. A representative human-recorded training corpus and broader listening evaluation
are still needed; the three additional human fixtures only establish integration. The live microphone check opened successfully but encountered no
speech, so live spoken transcription remains to be verified.

CPU and short MPS training paths are tested. MPS CTC and TTS spectral-loss
operations use differentiable CPU fallbacks where needed. CUDA selection exists
but no CUDA hardware was available. DDP, AMP, gradient accumulation/checkpointing,
streaming causal models, sharded loading, and direct R2 fetching are future work.
TTS currently caches features/waveforms in memory; change that before large-corpus
training. Versioned configs and manifests provide upgrade points without claiming
those scaling features are implemented.

Architecture references: [Tacotron](https://arxiv.org/abs/1703.10135) for learned
text/acoustic attention, and [MelGAN](https://arxiv.org/abs/1910.06711) for the
convolutional Mel-to-waveform direction. This MVP uses its own simplified models
and reconstruction objectives; it does not reproduce either full system.

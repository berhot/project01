# NABRA MVP completion report

Verification date: 2026-11-02 (Asia/Riyadh). Project: `BERHOT/models/nabra/variants/nano/mvp`.

## A. Executive summary

The ASR release lifecycle, automated regression suite, from-scratch TTS engineering
pipeline, and bounded-buffer microphone implementation are operational. **62 tests
passed in the final MPS-enabled suite.** The CPU-only suite ran the same 62 tests:
60 passed and the two optional MPS tests skipped. No test failed.

ASR `0.2.0` is an experimental release of the existing epoch-75 checkpoint. TTS
`0.1.0` is independently packaged from a genuine 500-step neural training run.
The mandatory Arabic synthesis produced a 4.76-second WAV; checkpoint and release
float tensors and PCM WAV bytes matched exactly. Both ASR and TTS additionally
passed temporary end-to-end tests using human-recorded Arabic FLEURS samples.

Engineering success does not establish model quality. The released models were
trained on two synthetic Arabic voice clips (9.3588 seconds total); their original
validation is one synthetic clip (2.5851 seconds). The three human recordings are
additional integration fixtures, not training for the published releases and not
a representative benchmark. ASR generalization remains poor. TTS intelligibility
has not been independently assessed. Microphone capture opened successfully but
encountered no speech in the observed interval; live spoken transcription still
needs an operator check. Buffered recognition is explicitly **not causal streaming**.

All 47 protected files in the preservation inventory remain byte-identical,
including the original shared speech files/manifests, ASR `0.1.0`, and the improved
ASR run. Historical releases/checkpoints were not overwritten. No JSM changes,
cloud uploads were made. This report describes verification before Git publication;
see the adjacent README for the subsequent commit/push delivery record.

## B. Milestone scorecard

Statuses: PASS = implemented and executed successfully; PARTIAL = implemented but
some verification is outstanding; BLOCKED = an external verification dependency
is missing; NOT IMPLEMENTED = absent by design or unfinished. No percentages used.

| Requirement | Status | Evidence / boundary |
| --- | --- | --- |
| A1–A3: inspect checkpoint, config, run and hash-bound evaluation | PASS | Checkpoint hash `bfbad351749d4ed9acfe49c4a90d1a1b9c3ddc0b170df2bfb3a7cc827a823317`; existing report matches exact bytes. |
| A4: actual audio transcription | PASS | CPU inference on original paired WAVs; synthetic origin explicitly recorded. Human waveform processing also tested. |
| A5–A7: ASR 0.2.0, preserve 0.1.0, required files | PASS | New immutable directory; four required files present; old release unchanged. |
| A8–A10: exported tensors, hashes, checkpoint/release inference | PASS | Every tensor equal; all file hashes verified; two exact transcript comparisons. |
| A11: deterministic equivalence | PASS | Exact CPU text equality for training and validation WAVs. |
| A12: experimental status and explicit observed gate | PASS | CER 0.4117647058823529 <= explicitly approved 0.5; production_ready=false. |
| B1: ingestion cases | PASS | Missing files/transcripts, empty labels, invalid/truncated WAV, rates/channels/widths, duration bounds, duplicates, unmatched labels, UTF-8, hashes/manifests. |
| B2: acoustic features | PASS | dtype/shape/finite/stable features, 1-sample and 21-second feature inputs, config restoration and checkpoint/release equivalence. Ingestion separately limits clips to 20 seconds. |
| B3: tokenizer/CTC | PASS | Encoding, decoding, blanks, repeated/blank-separated characters, serialization, OOV and vocabulary mismatch rejection. |
| B4: dataset/batching | PASS | WAV pairs, variable lengths, padding, flattened targets/lengths, empty dataset rejection, repeated-label alignment feasibility. |
| B5: neural model | PASS | Shapes, lengths, finite logits/loss, nonzero gradients, optimizer updates, exact state restoration. |
| B6: training | PASS | CPU training, checkpointing, optimizer resume, corrupt loads, dataset/tokenizer/model/features mismatch rejection, logging and best-checkpoint selection. |
| B7: evaluation | PASS | CER/WER edits and empty hypotheses; exact checkpoint/report binding; stale report rejection and immutable report re-verification. |
| B8: releases | PASS | Safetensors, exact weights, provenance/hashes, loading/equivalence, overwrite/version/path/corruption/missing-file and metric-gate rejection. |
| B9: real labeled human speech integration | PASS | Two additional offline tests train/evaluate/package/infer with CC-BY-4.0 FLEURS Arabic; temporary artifacts, unchanged publisher transcripts. |
| C: separate TTS modules and thin commands | PASS | `src/tts/`, five entry points, versioned config; existing ASR layout remains compatible. |
| C: manifest-based paired data, sample rates, normalization, speakers, provenance | PASS | Hash-verified manifests; strict 16 kHz input; NFC+whitespace, diacritics preserved; unknown speaker/license fields stay unknown. |
| C: train/val partitions and fingerprints | PASS | Fixed manifest assignment, duplicate audio rejection, content-based fingerprints and strict resume checks. No automatic speaker-disjoint split claim. |
| C: trainable acoustic model, alignment and masked losses | PASS | Learned attention, autoregressive decoder, stop head, masked Mel/BCE and guided-attention losses. No forced durations invented. |
| C: from-scratch neural vocoder | PASS | Convolutional upsampling trained with waveform L1 and multi-resolution STFT objectives; no playback, OS TTS, pretrained model, noise or tone substitution during synthesis. |
| C: actual training, resume, checkpoints and independent release | PASS | 100 CPU steps then resume to total 500; TTS release 0.1.0; checkpoint/optimizer/config/fingerprint tests. |
| C: waveform integrity and distinct/deterministic text behavior | PASS | Finite PCM16 mono 16 kHz output; text and weight sensitivity; repeat and checkpoint/release equality. |
| C: mandatory Arabic WAV | PASS | 76,160 samples, 4.76 s; learned stop fired at a cap of 800 frames. |
| C: subjective intelligibility review | PARTIAL | `afplay` completed successfully. No human listening assessment recorded; playback success is not intelligibility evidence. |
| C: independent-ASR intelligibility scoring | NOT IMPLEMENTED | Optional evaluation; no external ASR was used as a substitute for Nabra or as a fabricated quality score. |
| D: strategy honesty | PASS | Bounded buffered bidirectional inference; no causal/low-latency encoder claim. |
| D: live microphone capture and configurable device | PASS | Enumerated devices; MacBook Air Microphone opened with 16 kHz mono stream, 2.8 seconds captured in a 3.125-second check. |
| D: live spoken transcription | PARTIAL | No speech crossed the configured VAD threshold in the capture check; no partial/final live speech events observed. |
| D: chunks, buffer management, VAD, partial/final and stabilization | PASS | Ordered incremental input, finite bounds, energy VAD, per-segment replacement events, consecutive-partial common prefix. |
| D: no duplicated committed segments / graceful shutdown | PASS | Unique final segment IDs, idempotent close, interruption and device-open failure tests. Queue is bounded and overflow fails explicitly. |
| D: feature consistency, latency and RTF | PASS | Wave-recognizer matches offline preprocessing; measured on synthetic and human prerecorded audio delivered in chunks. |
| D: genuinely causal streaming encoder/state | NOT IMPLEMENTED | Explicitly selected the permitted buffered strategy; current model requires future context. |
| Configuration, safe loading, atomic saves, path/version validation | PASS | Versioned configs, weights_only checkpoint loading, hash verification, atomic checkpoint/release staging, traversal rejection. |
| Reproducibility and environment records | PASS | Seeds, stable inputs, content fingerprints, optimizer/RNG state and dependency/source-hash recording. Bit-identical cross-device training is not promised. |
| CUDA execution | BLOCKED | No CUDA device in this environment. Device selection is implemented but CUDA execution was not verified. |
| Multi-GPU/DDP/AMP/sharded loading/direct object storage | NOT IMPLEMENTED | Explicit future upgrade work; no speculative distributed framework added. |

## C. Actual architecture

### ASR

16 kHz mono PCM16 WAV -> 80-bin log-Mel power features (400-sample FFT,
160-sample hop, per-recording mean/std normalization) -> two stride-2 temporal
Conv1d/GELU layers -> two-layer bidirectional GRU (hidden size 128) -> character
logits -> CTC greedy decoding. Vocabulary: 22 including blank. **633,622 parameters.**
Output temporal length is two successive ceiling divisions by two. CPU CTC loss
fallback preserves gradient flow on MPS. The overfit diagnostic still exists;
its ML implementation was moved into `src/overfit.py` behind the original command.

### TTS acoustic model

NFC/whitespace-normalized Arabic text with diacritics retained -> fixed character
embedding (PAD/EOS, vocabulary 353) plus speaker-conditioning slot -> bidirectional
GRU text encoder -> autoregressive GRUCell decoder with additive, cumulative
location-aware learned attention -> four Mel frames per step and stop probability.
**210,433 parameters.** The fixed alphabet supports input encoding; it does not
mean every character or pronunciation has been learned. No forced alignments or
uniform character durations are claimed. Teacher forcing, masked Mel L1, stop
BCE, and a soft diagonal attention prior are used in training.

### Vocoder

Absolute log-Mel power (80 bins, 16 kHz, FFT 400, hop 160, log scaling /8) ->
convolutions and transposed convolutions with strides 5,4,4,2 -> mono tanh waveform.
**25,565 parameters.** Loss: waveform L1 plus spectral convergence/log-magnitude
loss at FFT sizes 256,512,1024. No discriminator, pretrained vocoder, stored clip
lookup, oscillator, or random-noise input is used. This is a simplified neural
reconstruction vocoder, not a reproduction of the full MelGAN system. TTS
absolute features intentionally differ from ASR's per-utterance normalization.

### Buffered recognition

PortAudio/CoreAudio input requests 16 kHz mono float samples. A bounded 32-chunk
queue separates capture from inference. Audio is assembled into fixed 20 ms
frames, segmented by RMS silence and an eight-second maximum, and partially
re-decoded every 0.75 seconds by default. Final events have one segment ID each.
The CLI keeps at most 128 preview events; an optional JSONL log holds full history
on disk. Partial-prefix agreement is revisable. Hard segment boundaries can split
words. No recurrent acoustic state is carried between segments. This is not
causal streaming, and low compute RTF does not eliminate utterance/VAD latency.

### Data, checkpoints and releases

Shared `intake/speech` paths are preserved. Directory or JSONL inputs are supported;
relative manifest paths resolve beside the manifest. TTS supports supplied speaker
IDs and provenance, rejects nearly silent inputs, and fingerprints content and
metadata. No real transcript, speaker identity, or license is fabricated.

ASR checkpoints retain historical format compatibility with added environment,
step, fingerprint and random-state fields. TTS checkpoints are task/schema tagged
and contain both model states, optimizer state, features/configs, tokenizer,
speakers, data fingerprints, steps/epochs, and random state. Releases stage to a
temporary directory, compare exported tensors, write hashes and metadata, and
refuse existing versions. Hash integrity is not a cryptographic signature scheme.

References: [Tacotron](https://arxiv.org/abs/1703.10135) informs learned text/acoustic
attention; [MelGAN](https://arxiv.org/abs/1910.06711) informs convolutional
Mel-to-waveform generation. Neither complete architecture is claimed here.

## D. Files changed

The following source/config/documentation files differ from the snapshot taken
at the start of this task. Pre-existing user edits are not attributed wholesale
to this task; this comparison uses that initial working-tree snapshot.

| File relative to MVP | Change | Purpose |
| --- | --- | --- |
| `README.md` | modified | Setup, commands, architectures, data provenance, streaming semantics, and limitations. |
| `configs/asr_buffered.json` | created | Versioned bounded-buffer/VAD configuration. |
| `configs/asr_nano.json` | modified | Versioned ASR model and feature configuration. |
| `configs/human_speech_fixture.json` | created | Pinned Google FLEURS fixture rows, hashes, license, and transformation record. |
| `configs/tts_nano.json` | created | Versioned acoustic, Mel, and vocoder configuration. |
| `requirements.txt` | modified | Valid Python runtime dependencies; replaces the misplaced shell command. |
| `scripts/asr_microphone.py` | created | Thin CLI entry point for asr microphone. |
| `scripts/asr_stream.py` | created | Thin CLI entry point for asr stream. |
| `scripts/evaluate.py` | modified | Thin CLI entry point for evaluate. |
| `scripts/fetch_test_speech.py` | created | Thin CLI entry point for fetch test speech. |
| `scripts/inference.py` | modified | Thin CLI entry point for inference. |
| `scripts/overfit_ctc.py` | modified | Thin CLI entry point for overfit ctc. |
| `scripts/train.py` | modified | Thin CLI entry point for train. |
| `scripts/tts_evaluate.py` | created | Thin CLI entry point for tts evaluate. |
| `scripts/tts_inference.py` | created | Thin CLI entry point for tts inference. |
| `scripts/tts_inspect.py` | created | Thin CLI entry point for tts inspect. |
| `scripts/tts_release.py` | created | Thin CLI entry point for tts release. |
| `scripts/tts_train.py` | created | Thin CLI entry point for tts train. |
| `src/evaluator.py` | modified | Checkpoint-bound evaluation with fresh verification of existing reports. |
| `src/features.py` | modified | Feature config validation, truncated PCM rejection, finite/short-waveform handling. |
| `src/fixtures.py` | created | Explicit optional download and integrity verification of human speech fixtures. |
| `src/inference.py` | modified | Tokenizer/features compatibility and required release hashes/path/schema checks. |
| `src/ingestion.py` | modified | Missing/truncated audio rejection, JSONL source reading, hash checks, split validation. |
| `src/model.py` | modified | ASR model dimension and dropout validation; architecture/state keys unchanged. |
| `src/overfit.py` | created | Existing CTC diagnostic implementation moved behind its original CLI. |
| `src/packaging.py` | modified | Evaluation schema/metric gates and explicit experimental release labels. |
| `src/runtime.py` | created | Device validation, atomic JSON, content fingerprints, dependency/source-code provenance. |
| `src/stream_cli.py` | created | Recorded/live argument handling, versioned config, bounded event preview and incremental event logs. |
| `src/streaming.py` | created | Bounded segment processing, VAD, partial/final updates, microphone selection/capture, latency metrics. |
| `src/trainer.py` | modified | Configurable directory/manifest inputs, devices, strict resume/config checks, optimizer/RNG and dataset provenance. |
| `src/tts/__init__.py` | created | Independent TTS package. |
| `src/tts/acoustic_model.py` | created | Trainable autoregressive attention acoustic predictor and masked training losses. |
| `src/tts/cli.py` | created | Shared TTS CLI argument handling. |
| `src/tts/dataset.py` | created | Absolute log-Mel features, manifest validation, silence checks, speaker/provenance metadata, collation. |
| `src/tts/evaluator.py` | created | Hash-bound acoustic/vocoder evaluation and deterministic distinct-text synthesis checks. |
| `src/tts/inference.py` | created | Strict checkpoint construction, bounded text-conditioned synthesis and PCM16 output. |
| `src/tts/packaging.py` | created | Immutable, gated Safetensors TTS releases, tensor equality and integrity verification. |
| `src/tts/tokenizer.py` | created | Fixed character inventory, NFC/whitespace policy preserving Arabic diacritics. |
| `src/tts/trainer.py` | created | Training, atomic checkpoints, optimizer/random-state resume, metrics and dataset fingerprints. |
| `src/tts/vocoder.py` | created | Trainable convolutional waveform decoder and L1/multi-resolution spectral reconstruction losses. |
| `tests/helpers.py` | created | Executable unittest coverage for helpers. |
| `tests/test_dataset.py` | modified | Executable unittest coverage for dataset. |
| `tests/test_features.py` | created | Executable unittest coverage for features. |
| `tests/test_human_speech.py` | created | Executable unittest coverage for human speech. |
| `tests/test_ingestion.py` | modified | Executable unittest coverage for ingestion. |
| `tests/test_model.py` | modified | Executable unittest coverage for model. |
| `tests/test_pipeline.py` | modified | Executable unittest coverage for pipeline. |
| `tests/test_streaming.py` | created | Executable unittest coverage for streaming. |
| `tests/test_tokenizer.py` | modified | Executable unittest coverage for tokenizer. |
| `tests/test_tts.py` | created | Executable unittest coverage for tts. |
| `NABRA_MVP_COMPLETION_REPORT.md` | created | This requirement-by-requirement evidence and limitations report. |

Generated files are outside the source tree: ASR release `0.2.0`, two TTS runs,
TTS release `0.1.0`, generated WAVs, separately cached FLEURS fixtures, and the
verification logs/manifests. `completion-verification/generated-files.json`
provides the complete generated-file inventory with hashes. `source-changes.json`
provides source-file hashes. No shared input recording or transcript changed.

## E. Commands executed

Working directory for module commands: `models/nabra/variants/nano/mvp`.
Interpreter: root `.venv/bin/python`. Absolute paths were used during execution;
variables below abbreviate those same paths for readability.

```bash
BERHOT_ROOT=/Users/farisalsaiari/Desktop/BERHOT
ASR="$BERHOT_ROOT/storage/artifacts/nabra/nano/mvp/nabra-20261102T121628Z-f4747217/checkpoints/epoch_0075.pt"
TTS1="$BERHOT_ROOT/storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-001"
TTS2="$BERHOT_ROOT/storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002"
AUDIT="$BERHOT_ROOT/storage/artifacts/nabra/nano/mvp/completion-verification"

python -m unittest discover -s tests -v
NABRA_TEST_MPS=1 python -m unittest discover -s tests -v
python -m unittest tests.test_human_speech -v
NABRA_TEST_MPS=1 python -m unittest tests.test_model.ModelTests.test_mps_backward -v
python -m scripts.evaluate --checkpoint "$ASR" --device cpu
python -m scripts.inference --release 0.2.0 --audio "$BERHOT_ROOT/intake/speech/val/clip101.wav" --device cpu
python -m scripts.tts_inspect --output "$AUDIT/tts-input"
python -m scripts.tts_train --config configs/tts_nano.json --steps 100 --device cpu --threads 2 --run-id tts-mvp-verification-001 --train-source "$AUDIT/tts-input/train.jsonl" --val-source "$AUDIT/tts-input/val.jsonl"
python -m scripts.tts_train --config configs/tts_nano.json --resume "$TTS1/checkpoints/step_000100.pt" --steps 500 --save-interval 100 --device cpu --threads 2 --run-id tts-mvp-verification-002 --train-source "$AUDIT/tts-input/train.jsonl" --val-source "$AUDIT/tts-input/val.jsonl"
python -m scripts.tts_evaluate --checkpoint "$TTS1/checkpoints/step_000100.pt" --device cpu
python -m scripts.tts_evaluate --checkpoint "$TTS2/checkpoints/step_000500.pt" --device cpu
python -m scripts.tts_release --checkpoint "$TTS2/checkpoints/step_000500.pt" --version 0.1.0 --report "$TTS2/evaluations/step_000500.json" --max-mel-l1 1.0 --max-spectral-loss 5.0 --approve
python -m scripts.tts_inference --checkpoint "$TTS2/checkpoints/step_000500.pt" --text 'مرحبا بكم في نبْرة للذكاء الاصطناعي' --output "$TTS2/samples/arabic-checkpoint.wav" --max-frames 800 --device cpu
python -m scripts.tts_inference --release 0.1.0 --text 'مرحبا بكم في نبْرة للذكاء الاصطناعي' --output "$TTS2/samples/arabic-release.wav" --max-frames 800 --device cpu
python -m scripts.asr_microphone --list-devices
python -m scripts.asr_microphone --release 0.2.0 --microphone 'MacBook Air Microphone' --seconds 3 --output "$AUDIT/microphone-capture.json"
python -m scripts.asr_stream --release 0.2.0 --audio "$BERHOT_ROOT/intake/speech/train/clip001.wav" --config configs/asr_buffered.json --output "$AUDIT/stream-final.json"
python -m scripts.asr_stream --release 0.2.0 --audio "$BERHOT_ROOT/storage/cache/datasets/fleurs-ar-smoke-v1/val/fleurs-7.wav" --config configs/asr_buffered.json --output "$AUDIT/stream-human.json"
python -m scripts.fetch_test_speech
```

Also executed: `pip install 'sounddevice>=0.5'` in the existing virtual environment;
`compileall` over source/scripts/tests; `git diff --check`; local `afplay` of the
TTS release WAV (exit 0). Inline Python verification invoked
`package_release(ASR, '0.2.0', 0.5, approve=True)`, compared every exported tensor,
recomputed evaluations and hashes, checked checkpoint/release text and WAV
identity, and verified protected historical files. Actual raw evidence is under
`completion-verification/`.

Failed/corrected development commands are recorded rather than hidden:

- Two early `unittest discover -s tests -v` launches ran from the BERHOT root
  instead of the MVP directory and exited 1. They were rerun from the correct
  directory; this was command context, not a passing test result.
- One inline file-generation command stopped with `NameError: json is not defined`;
  the missing import and remaining config/entry-point writes were completed.
- The first completion-report generation read microphone statistics at the wrong
  JSON level and raised `KeyError: capture_seconds`; the report generator was
  corrected to read the nested `stats` object and rerun.
- A patch application was rejected due to unmatched patch context. The
  patch was reapplied against the inspected file contents; no success was
  claimed for rejected patches.
- System-Python `urllib` could not verify the Hugging Face certificate chain.
  System `curl` succeeded with normal TLS verification; TLS verification was
  not disabled.
- The FLEURS training-row API request returned HTTP 500. The downloaded fixture
  instead explicitly repartitions three upstream validation rows, with that
  limitation and upstream provenance recorded. No official benchmark score is
  claimed. The publisher's Arabic transcription strings were preserved.

## F. Automated test results

| Run | Total | Passed | Failed | Skipped | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final CPU suite | 62 | 60 | 0 | 2 optional MPS tests | 4.759 s |
| Final suite with `NABRA_TEST_MPS=1` | 62 | 62 | 0 | 0 | 6.480 s |
| Standalone human-speech integration | 2 | 2 | 0 | 0 | 1.749 s |

These are 62 unique test methods; the rows are separate executions, not 126 unique
tests. Parameterized subcases are not counted as separate methods. CPU is used
for the normal suite; the optional tests execute one ASR MPS backward step and
one TTS MPS training/synthesis step. Tests use temporary artifact/release roots.
Human fixture tests are offline and skip if the explicit optional cache is absent.
No test downloads a model or calls an external speech API.

Platform: `macOS-26.3-arm64-arm-64bit`. Python `3.12.3`.
Dependencies: `{"numpy": "2.5.4", "safetensors": "0.8.0", "sounddevice": "0.5.6", "torch": "2.15.0"}`.
MPS available: `True`. CUDA available: `False`.
Full logs: `final-cpu-tests.log`, `final-with-mps-tests.log`, `human-speech-tests.log`.

The human fixture is [Google FLEURS Arabic](https://huggingface.co/datasets/google/fleurs),
revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, under CC BY 4.0. Upstream
validation rows 5 and 9 form the engineering training fixture (27.2 seconds),
and row 7 the held-out fixture (8.52 seconds). This repartition is **not a benchmark**.
Audio was converted to PCM16, original downloads retained, text unchanged, and
speaker identity left unspecified. Attribution accompanies the cache and the
versioned fixture config pins file and transcript hashes.

## G. Observed training results

### ASR historical improved run, reverified

Run `nabra-20261102T121628Z-f4747217`, 100 completed epochs, selected epoch 75.
No accuracy-improvement claim is based on a new ASR training run in this task.

| Metric | Epoch 75 |
| --- | ---: |
| Training CTC loss | 0.5950419902801514 |
| Training CER | 0.05970149253731343 |
| Training WER | 0.3 |
| Validation CTC loss | 2.8525662422180176 |
| Validation CER | 0.4117647058823529 |
| Validation WER | 1.3333333333333333 |

Epoch 100 training CER was 0.014925373134328358, but validation CER worsened to
0.47058823529411764, confirming why epoch 75 was selected. Original training used
Apple MPS per the supplied history; fresh evaluation and export equivalence were
executed on CPU. Historical training wall time was not measured in this task.
The overfit checkpoint still outputs the exact memorized training sentence; that
is not evidence of generalization.

### TTS actual new runs

- Run 001: 100 CPU optimizer steps, measured training loop 8.4535 seconds.
- Run 002: restored model/optimizer/RNG, continued steps 101–500, measured loop
  33.4743 seconds. These timings exclude import/setup and are hardware-specific.
- Data: original two synthetic training clips, 9.3588 seconds total; one synthetic
  validation clip, 2.5851 seconds. Unknown dataset license is not invented.
- Acoustic parameters: 210,433; vocoder parameters: 25,565.

| Metric | Step 100 training sample | Step 500 training sample | Step 500 held-out synthetic clip |
| --- | ---: | ---: | ---: |
| Mel L1 | 0.2096827626 | 0.1068818793 | 0.2160443216562271 |
| Stop BCE | 0.0370685682 | 0.0034455452 | 0.05301210656762123 |
| Waveform L1 | 0.1492585838 | 0.0989994779 | 0.11464906483888626 |
| Multi-resolution spectral loss | 2.4572374821 | 2.1173279285 | 2.1075470447540283 |

Training rows are individual step examples/crops, not aggregate corpus metrics.
The held-out Mel L1 did not improve relative to step 100 (0.2151803374); spectral
loss improved from 2.5495882034 to 2.1075470448. No intelligibility inference is
made from these losses. Human-speech integration trains separate tiny models for
one step in temporary directories; those are execution tests, not released models.

## H. Actual inference results

### ASR checkpoint versus release

Training reference: `مرحبا أنا أتعلم الذكاء الاصطناعي`

Checkpoint and ASR `0.2.0`, exactly equal:
`مرحبا أنا أتعلم الذكا الاصطناعي`

Validation reference: `أنا أتعلم العربية`

Checkpoint and ASR `0.2.0`, exactly equal:
`ا أتعم الا لوة`

ASR `0.1.0` still loads and returns an empty transcript on the training recording.
The historic overfit checkpoint returns its exact memorized sentence. Both are
preserved, not replaced or misrepresented as improved.

### TTS checkpoint versus release

Input: `مرحبا بكم في نبْرة للذكاء الاصطناعي`

- Checkpoint WAV: `storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002/samples/arabic-checkpoint.wav`
- Release WAV: `storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002/samples/arabic-release.wav`
- Mono PCM16, 16,000 Hz, 76,160 samples, **4.76 seconds**.
- Neural float-waveform RMS: 0.05472341179847717; peak: 0.5282715559005737.
- Finite, nonempty, exact checkpoint/release float waveform and PCM-byte equality.
- WAV SHA-256: `5b749c05c7c0366c0d3168421fa688e66d951f97ed2f0dcc09d32281c242661b`.
- Learned stop fired with max_frames=800. Earlier evaluation used a 400-frame
  cap and both texts reached that cap; the later 800-frame check demonstrated
  stopping at frame 476. Neither result is hidden.
- Different Arabic text produced different deterministic waveforms; changing
  neural weights also changed output in regression tests.
- Playback command completed (exit 0). **No human or independent-ASR
  intelligibility assessment is recorded.** Generated audio is experimental
  model output, not certified intelligible speech.

### Buffered and microphone paths

Original synthetic training WAV: final transcript matched offline ASR `0.2.0`.
Measured compute RTF: 0.017584; decode calls:
6; input audio 4.3716875 seconds.

Human FLEURS waveform: 8.52 seconds delivered incrementally, ten decode calls,
compute RTF 0.010828. The final output was
`الذودل الذطنالذطنال نالسود`, demonstrating poor generalization rather than good
recognition. The pipeline emitted ordered partial/final events and released its
buffer. Energy VAD retained a 7.52-second segment after discarding silence.

Microphone: MacBook Air Microphone (enumerated index 0 during this run; do not
hardcode it), device default 48 kHz, opened through PortAudio at 16 kHz mono.
Capture duration 3.1249 seconds; 2.8 seconds
received; zero decode calls/events because no speech exceeded VAD threshold.
This verifies capture and silence handling, not live spoken accuracy. Recorded
human/synthetic chunk tests exercise the recognizer independently of capture.

## I. Release inventory

| Task / version | Directory | Source / gate | Readiness |
| --- | --- | --- | --- |
| ASR 0.1.0 | `storage/releases/nabra/nano/0.1.0/` | Existing historical release, unchanged | Loads, but blank output; not production |
| ASR 0.2.0 | `storage/releases/nabra/nano/0.2.0/` | Run `nabra-20261102T121628Z-f4747217`, epoch 75; exact checkpoint SHA; explicit CER <= 0.5 | Experimental engineering release |
| TTS 0.1.0 | `storage/releases/nabra/tts/nano/0.1.0/` | Run `tts-mvp-verification-002`, step 500; Mel L1 <= 1.0 and spectral loss <= 5.0; explicit approval | Experimental; no intelligibility certification |

The TTS gates are documented numerical engineering gates, not a surrogate for
speech quality. Both reports bind the exact source checkpoint. Release manifests
record approval, experimental quality status, and file hashes. No existing release
was overwritten; repeat version creation is rejected in tests.

### ASR 0.1.0 (preserved)

| File | SHA-256 |
| --- | --- |
| `config.json` | `85168da526cd3697fd7616f35eeb15a4a0ee331410ce4160595c9e9ba1e494f8` |
| `manifest.json` | `9f5602533e60f2f70c1ad0802937e983e5fd6269379b090ad3dc85db8523cbd2` |
| `model.safetensors` | `8b7abd641ad626c611204c45f8110c90b57242a8c58f06642097f710d515fd2e` |
| `tokenizer.json` | `0ac8d909fbd594b938dad8623cf804572832d928251c8992328a2da71ca17136` |
### ASR 0.2.0

| File | SHA-256 |
| --- | --- |
| `config.json` | `85168da526cd3697fd7616f35eeb15a4a0ee331410ce4160595c9e9ba1e494f8` |
| `manifest.json` | `f49f896408979b56c23fd67b7ec880cd2898236c06de92879c2bc06834544e49` |
| `model.safetensors` | `f36cb4937479a9d5884aebff9058cfdddc8239bcc1d3b3e69bc3ab5766c2768a` |
| `tokenizer.json` | `0ac8d909fbd594b938dad8623cf804572832d928251c8992328a2da71ca17136` |
### TTS 0.1.0

| File | SHA-256 |
| --- | --- |
| `acoustic.safetensors` | `9a1e77b6a6cecb530f09b11cf964c70edb1e4a51d8549cdc33e2eb9e164a0ef2` |
| `config.json` | `d18fc10db3bd1530290311c108eade560ea83cf63a22f0db73dfeaee2c7e439e` |
| `evaluation.json` | `d052fad4220ad05fe70269e0c8f5588898373face453c5e276f3e02007b59f88` |
| `manifest.json` | `89136d968d3369e12bc862dde77afcce216d3d0864a355a3c71e927f319fb442` |
| `speakers.json` | `74df0bb9f3e406e460c28431eff3d48cc5a274fed5728ae099183392ac0f06d0` |
| `tokenizer.json` | `5c5b33a9aad0b06518aecf0c1e88cb54aa37f9696d7a7ff88450204ced7811a3` |
| `vocoder.safetensors` | `34563fec8563c09670585ab9d2482d792124ffe439fdc75a05497371378bbc39` |

## J. Known limitations

- Released models learned from approximately 9.36 seconds of one synthetic voice;
  the additional human fixtures do not change their training or generalization.
- ASR CTC can collapse to blank output; the old release demonstrates this. Bias
  initialization and overfit diagnostics do not establish robust recognition.
- Arabic dialect, speaker, gender, channel and environment coverage are inadequate.
- TTS acoustic attention/stop behavior is fragile on unseen text; inference is
  length-bounded and reports truncation rather than concealing it.
- TTS waveform reconstruction is real neural generation, but quality/intelligibility
  remain unverified. No MOS or independent ASR score was fabricated.
- TTS speaker IDs are supported as supplied metadata/conditioning; meaningful
  multi-speaker training has not been demonstrated.
- Energy VAD is basic; quiet speech may be missed and noise may trigger segments.
- Buffered bidirectional inference has utterance/right-context latency; low RTF
  is compute throughput, not causal latency. Hard cuts can split words.
- Live voiced microphone transcription was not observed; only capture/silence.
- TTS eagerly caches waveforms/features and uses one training record per step.
  Large corpora need lazy/sharded loading and improved batching.
- CPU and brief MPS paths are tested. CUDA, long MPS runs, multi-GPU, AMP, DDP,
  accumulation/checkpointing and R2 integration are not verified/implemented here.
- Current checkpoints/manifests can contain absolute local audio paths. Content
  fingerprints ignore path relocation, and new manifests support relative paths,
  but automatic migration of historical checkpoint source paths is not implemented.
- Inference releases are content-hashed, not cryptographically signed. Actual
  deployment policy and storage permissions are outside this local task.

## K. Production readiness by capability

| Capability | Engineering status | Production status |
| --- | --- | --- |
| ASR pipeline | PASS: ingestion through release/inference, synthetic and human fixture tests | Requires broader corpus/operational validation |
| ASR recognition quality | Poor measured generalization | Not ready |
| TTS pipeline | PASS: trainable acoustic/vocoder, resume, evaluate, release, synthesize | Requires larger-data and long-run validation |
| TTS generation quality | PARTIAL: technical output and stopping verified | Not ready; listening/intelligibility review missing |
| Local microphone pipeline | Capture and recorded-chunk processing verified; live voiced end-to-end check outstanding | Not ready for a live user-facing claim |
| True low-latency streaming model | NOT IMPLEMENTED; buffered strategy chosen | Not ready |
| Release management | PASS: immutable versions, hashes, exact export and gates | Local experimental lifecycle verified; deployment/signing not implemented |
| Automated testing | PASS: 62/62 with optional MPS enabled | Regression foundation established |
| Larger GPU training | Configurable CPU/MPS/CUDA selection; CUDA unavailable | Needs actual rented-GPU profiling, data scaling and recovery tests |

## L. Exact remaining work

Before declaring **every requested verification** complete:

1. Have an Arabic-speaking listener assess `arabic-release.wav`; record audibility,
   intelligibility, artifacts and whether it represents recognizable speech. If
   unintelligible, document that result and train on a substantially larger,
   appropriately licensed corpus. File existence and successful playback do not
   substitute for this assessment.
2. Perform a live voiced microphone check with a known spoken sentence, verify
   partial/final updates, and retain the event/latency log. Device capture itself
   is already verified; the available capture interval contained no speech.

Reproducible commands (run from the MVP directory):

```bash
BERHOT_ROOT=$(git rev-parse --show-toplevel)
afplay "$BERHOT_ROOT/storage/artifacts/nabra/tts/nano/mvp/tts-mvp-verification-002/samples/arabic-release.wav"
python -m scripts.asr_microphone --list-devices
python -m scripts.asr_microphone --release 0.2.0 --microphone 'MacBook Air Microphone' --seconds 15 --config configs/asr_buffered.json --output "$BERHOT_ROOT/storage/artifacts/nabra/nano/mvp/live-voiced-review.json"
```

Speak during the 15-second capture. Use a new output path for subsequent checks.
For production quality (separate from the engineering MVP), acquire diverse
human speech, establish speaker/source-disjoint splits, evaluate intelligibility
independently, improve acoustic/vocoder capacity and training, and validate
actual GPU deployment/scaling. Implement a causal encoder only if true streaming
latency becomes a requirement; it was not silently claimed in this MVP.

## M. Final verdict

**NOT READY — REMAINING ENGINEERING WORK**

This is a conservative verification verdict, not a claim that the implemented
pipelines failed. All four engineering workstreams are implemented, both releases
are valid, human-speech integration passes, and all 62 tests pass with MPS enabled.
The remaining gaps are the live voiced microphone end-to-end observation and a
recorded Arabic listening assessment of generated TTS audio. Those checks need
human participation and cannot be inferred from numeric losses, playback exit
status, synthetic overfit, or successful file creation. Model quality remains
experimental regardless of the engineering test results.

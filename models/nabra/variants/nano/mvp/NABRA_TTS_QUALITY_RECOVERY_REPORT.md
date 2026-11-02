# NABRA TTS speech quality recovery report

2026-11-02, Asia/Riyadh. Scope: one-sentence neural learning proof, not a production TTS release.

**Outcome:** the user listened to the refined text-to-speech output and confirmed:

> words clear , hiss reduced not 100% quality

This establishes recognizable Arabic for the single training sentence, with residual hiss.
No claim of naturalness, clean studio audio, unseen-text intelligibility, new-speaker
quality, or production readiness follows from this result. No new release was created.

## A. Root-cause analysis

The user's earlier listening result for TTS `0.1.0` was **noise, not recognizable speech**.
That overrides any earlier implication that successful WAV generation established quality.
The following measurements use the unchanged training pair `intake/speech/train/clip001`.

| Candidate cause | Actual observation | Conclusion |
| --- | --- | --- |
| Insufficient data | Original release used two synthetic clips, 9.3588 s. Recovery uses just one 4.3716875 s clip and eventually produces user-confirmed clear words. | Data remains grossly insufficient for generalization, but cannot by itself explain failure to learn even one sentence. |
| Acoustic convergence | Original teacher-forced Mel MAE 0.099907; free-running MAE 0.326483. Experiment B reduces these to 0.049333 and 0.052320. | Under-convergence and inference mismatch were measured contributors. |
| Attention collapse | Original expected position advances from 0.895 to 28.917 across 33 tokens, with no backward steps under the recorded 0.1-token tolerance. Alignment image shows an advancing diagonal. | No evidence of collapsed attention on this pair. This is learned soft alignment, not verified character/phoneme timing. |
| Mel mismatch | Dataset and diagnostics use identical absolute log-power Mel features: 16 kHz, FFT 400, hop 160, 80 bands, log scale 8. Tests verify tensor equality and gradients. | No preprocessing-scale inconsistency found. ASR normalization is not used for synthesis. |
| Vocoder failure | Even ground-truth Mel produces waveform MAE 0.106641, STFT loss 2.159830, SNR -1.06 dB. | The vocoder was independently defective as a reconstruction model; acoustic improvements alone could not solve that. |
| Amplitude normalization | Source RMS 0.139803; old vocoder RMS 0.073543. No clipping. Learned recovery reaches 0.138936 without a playback gain correction. | Weak learned amplitude, not evidence of a PCM scaling bug. No normalization fix was substituted for learning. |
| Duration / stop | Old same-text output: 436 frames vs 438 target; recovered output: 440. Learned stopping occurs before the safety limit. | Large duration failure was not the main cause on this sentence. |
| Teacher-forcing mismatch | Free-running original error is about 3.27 times teacher-forced error. Training on the model's own previous outputs substantially closes this gap. | Explicitly addressed with differentiable free-running training, not forced inference durations. |
| Architecture / objective | Small raw-waveform generator and a larger residual generator remain weak within their bounded runs. A learned complex-spectral head with magnitude-dominated losses still gives -0.28 dB SNR. Resuming the same head with normalized complex/waveform MSE reaches 22.68 dB. | Phase-sensitive reconstruction supervision is the strongest measured intervention. Architecture/capacity and objective both changed; this is not a factorial ablation proving one parameter alone caused the failure. |

Random 32-frame training crops also change padding/context relative to full-utterance
inference: on an observed interior crop the old network differs by mean absolute
waveform value 0.00044573. Full-context overfit removes that
confound, but this small observed difference does not establish cropping as the main cause.

The initial complete recovery sample was then heard as Arabic with TV-like hiss.
Its quiet-region residual RMS was 0.006268, versus
0.003273 for ground-truth-Mel reconstruction. Output
RMS above 4 kHz was 0.002883, below the source's
0.005360. Thus the evidence does not support
simply attributing the hiss to an overall high-frequency boost. Acoustic errors
propagating through the phase-sensitive vocoder remained a measurable bottleneck.

## B. Vocoder experiment

All runs used the same original waveform and its actual ground-truth Mel tensor.
The source audio was never overwritten. No Griffin-Lim, recorded phase, prerecorded
waveform, speech service, OS speech synthesis, or pretrained speech model was used
for recovery inference. Existing synthetic source provenance is retained.

| Run | New optimizer steps | Parameters | Waveform MAE | Multi-resolution STFT loss | Reconstructed Mel MAE | SNR dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `recovery-vocoder-original-001` | 1000 | 25,565 | 0.108754 | 1.677875 | 0.152647 | -1.15 |
| `recovery-vocoder-residual-001` | 1500 | 107,073 | 0.113021 | 1.460399 | 0.116008 | -1.55 |
| `recovery-vocoder-spectral-001` | 1000 | 468,114 | 0.097829 | 1.592525 | 0.115624 | -0.28 |
| `recovery-vocoder-phase-001` | 3000 | 468,114 | 0.006999 | 0.824356 | 0.069618 | 22.68 |

The original run initialized from the preserved 500-step checkpoint. The residual
variant started from random weights with 64 channels and a 16-channel minimum.
The spectral variant started from random weights with 128 channels. It predicts
201 real and 201 imaginary coefficients per Mel frame with a convolutional network;
a differentiable inverse STFT converts those learned coefficients to samples.
It never receives target Fourier phase during synthesis. This is a neural spectral
vocoder, not conventional Mel inversion or a claim to reproduce a published model.

The spectral run used 1,000 steps, then resumed its optimizer for 3,000 additional
steps. Learning rate was 0.001, AdamW without weight decay, seed 42, CPU with two
threads, gradient norm clipping at 1. The successful resumed objective was:

```text
normalized complex-spectrum MSE
+ normalized waveform MSE
+ 0.1 * reconstructed log-Mel MAE
```

Normalization uses target energy during training only. Periodic WAVs, metrics,
checkpoints and optimizer state were saved. Final gradient norm was
0.254337; measured parameter-update L2 norm was
0.050867. No nonfinite losses or gradients
were observed. The restored model reproduces output exactly; zeroing its learned
weights produces zero output. Neither operation loads a recorded waveform for synthesis.

Final reconstruction:

```text
/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/nabra/tts/nano/mvp/recovery-vocoder-phase-001/samples/vocoder_reconstruction.wav
```

- Samples: 69,947; 16 kHz, mono, PCM16; duration 4.3716875 s.
- RMS: 0.138936; peak: 0.734274; clipping fraction: 0.
- WAV SHA-256: `a8808c4291ef1ed83254a13b9f819ea13c5c7db21328f439a5040447e319af3e`.
- Human response for this specific file: **Not listened yet**. Its own listening verdict remains pending.

The reconstruction's >20 dB aligned SNR, amplitude recovery, valid gradients and
large waveform-error reduction supported proceeding to B. Those measurements were
not treated as proof of intelligibility. C later received its own human confirmation.

## C. Acoustic model experiment

The existing 210,433-parameter character encoder, recurrent attention decoder,
reduction factor 4, and learned stop head were retained. There are no fabricated
phoneme durations, forced inference alignments, or prerecorded audio targets used
as decoder inputs at inference.

Experiment B started from the existing 500-step acoustic checkpoint. It performed
1,500 additional steps at learning rate 0.0005 on the single pair. Each update mixed
equal teacher-forced and fully free-running losses. In the latter path, gradients
propagate through predicted previous frames instead of always using target frames.
The existing masked Mel, stop-BCE and guided-attention losses remain in use.

| Measurement | Old model | B, 1,500 recovery steps | After waveform-aware refinement, 2,500 recovery steps |
| --- | ---: | ---: | ---: |
| Teacher-forced frame MAE | 0.099907 | 0.049333 | 0.050873 |
| Free-running frame MAE | 0.326483 | 0.052320 | 0.051529 |
| Free-running frame RMSE | 0.446681 | 0.089211 | 0.084640 |
| Predicted / target frames | 436 / 438 | 440 / 438 | 440 / 438 |
| Learned stop / frame cap | true / false | true / false | true / false |

The first acoustic result reduced free-running error by about 84%, matched duration
within 0.46%, and retained advancing attention. This supports **PASS for single-pair
text-to-Mel learning**, without asserting generalization or exact phoneme alignment.
The images show the learned broad acoustic pattern while fine harmonic structure
remains imperfect. Color limits are shared between the target and predicted Mel plots.

Final saved visualizations/tensors, relative to the BERHOT root:

```text
storage/artifacts/nabra/tts/nano/mvp/recovery-acoustic-waveform-001/samples/target_mel.png
storage/artifacts/nabra/tts/nano/mvp/recovery-acoustic-waveform-001/samples/predicted_mel.png
storage/artifacts/nabra/tts/nano/mvp/recovery-acoustic-waveform-001/samples/attention.png
storage/artifacts/nabra/tts/nano/mvp/recovery-acoustic-waveform-001/samples/step_002500.pt
```

The tensor file contains the actual target Mel, free-running predicted Mel, and
learned attention. Source transcripts are unchanged. All observed values were finite.

## D. Complete synthesis experiment

The output path is exactly the requested neural chain:

```text
Arabic text -> character tokenizer -> free-running acoustic model
-> predicted Mel -> learned complex-spectral vocoder -> inverse STFT -> PCM16 WAV
```

The waveform is generated before the diagnostic loads the original audio for
comparison. The original waveform is not an input to either synthesis component.
The vocoder has learned coefficients, not a stored source phase or waveform buffer.
Single-pair memorization is intentional in this diagnostic; it is not evidence of
learning arbitrary Arabic pronunciation.

First full output used B's acoustic checkpoint and A's recovered vocoder. The user
reported Arabic with TV-like background noise. That feedback triggered a bounded
1,000-step acoustic refinement at learning rate 0.0001. Vocoder weights were frozen.
The added objective was 0.5 times normalized waveform MSE, with fivefold weight in
reference-quiet regions. Quiet regions are defined during training by 321-sample
mean-square energy below 1% of global reference mean-square energy. No source-derived
noise mask, silence envelope, denoiser, or waveform post-filter runs during inference.

| Measurement | First full synthesis | Refined full synthesis |
| --- | ---: | ---: |
| Duration | 4.40 s | 4.40 s |
| Samples | 70,400 | 70,400 |
| Waveform MAE | 0.048727 | 0.014841 |
| Spectral loss | 1.569388 | 1.059514 |
| Reconstructed Mel MAE | 0.212284 | 0.144226 |
| Aligned SNR | 5.77 dB | 16.26 dB |
| RMS | 0.100818 | 0.134516 |
| Peak | 0.575595 | 0.722265 |
| Clipping | 0 | 0 |
| Quiet-region error RMS | 0.006268 | 0.004387 |

Quiet-region residual RMS fell approximately
30.0%. This does
not prove a matching perceptual reduction across all speech/noise frequencies.
Metrics compare the common waveform prefix without gain normalization or time warping;
the additional 453 generated samples are reported in duration, not hidden in a crop.
Waveform statistics are pre-PCM values; there was no clipping in these outputs.

The final file is:

```text
/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/nabra/tts/nano/mvp/recovery-full-002/samples/tts_full_overfit.wav
```

Transcript: **مرحبا أنا أتعلم الذكاء الاصطناعي**

The user was explicitly asked whether these words were understandable and hiss was
reduced, and answered **“words clear , hiss reduced not 100% quality”**. Therefore
recognizability is now confirmed for this specific output, while remaining hiss
is also confirmed. Audio hashes bind the listening record to the reviewed files:

- Final WAV: `374b9b26695371a1a30ae0c2d171c501eee2d0230559fa3a0fed3b69ce949124`.
- Acoustic checkpoint: `f7f8cb3dc756f53e6098f488a863df3014a066e7944f601d2e4a5171841d0797`.
- Vocoder checkpoint: `2daa878c2ea38843446a1f92e3d5fab3a93efef66b4cc0df797ba3258a2cb1ff`.

Repeated synthesis is deterministic on the tested CPU path. Different text produces
a different waveform, saved as `different_text.wav` beside the final sample. Weight
ablation tests confirm that changing either model changes output. These establish
model/text dependency, not correct pronunciation of unseen text. Human feedback is
saved separately from the earlier machine-only evaluation in
`quality-recovery-audit/listening-review.json`; the earlier evaluation's pending label
is a historical pre-listening state, not the final human verdict.

Playback commands:

```bash
afplay "/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/nabra/tts/nano/mvp/recovery-vocoder-phase-001/samples/vocoder_reconstruction.wav"
afplay "/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/nabra/tts/nano/mvp/recovery-full-002/samples/tts_full_overfit.wav"
```

The earlier hissier comparison remains at:

```text
/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/nabra/tts/nano/mvp/recovery-full-001/samples/tts_full_overfit.wav
```

## E. Architecture changes

All paths below are relative to this MVP directory.

| File | Change and reason |
| --- | --- |
| `src/tts/vocoder.py` | Backward-compatible optional residual capacity and learned complex-spectral head. Default raw-waveform architecture/state names remain compatible. CPU inverse-STFT fallback preserves MPS gradients. |
| `src/tts/acoustic_model.py` | Optional differentiable free-running/scheduled teacher forcing; optional inference attention capture; optional predicted frames from loss for waveform supervision. Default behavior remains compatible. |
| `src/tts/trainer.py` | Normalize legacy vocoder config defaults during resume so old checkpoints remain usable after optional config fields were added. |
| `src/tts/diagnostics.py` | Shared bounded train/resume, optimizer/RNG state, data/hash checks, finite checks, clipping, gradient/update logs, periodic audio/tensors, Mel/attention plots, full synthesis, phase objectives, frozen-vocoder refinement, noise analysis and thin CLI routing. |
| `scripts/tts_overfit_vocoder.py` | Thin entry point for Experiment A. |
| `scripts/tts_overfit_acoustic.py` | Thin entry point for B and waveform-aware refinement. |
| `scripts/tts_overfit_full.py` | Thin entry point connecting both learned components and reporting C. |
| `tests/test_tts_diagnostics.py` | Ten additional regression tests, including optional accelerator execution and frozen-vocoder behavior. |
| `requirements-diagnostics.txt` | Existing runtime requirements plus Matplotlib for standalone diagnostic figures; ordinary inference does not require it. |
| `README.md` | Current outcome, links, playback and diagnostic workflow; historical completion information remains identifiable. |
| `NABRA_TTS_QUALITY_RECOVERY_REPORT.md` | This evidence-based recovery report. |

No ASR source, intake organization, existing checkpoint, or historical release was
changed. All 137 protected data/artifact/release files have matching baseline
SHA-256 hashes. TTS `0.1.0` still loads and produces byte-identical original WAV output;
ASR `0.1.0` and `0.2.0` load and transcribe as before. Nothing was repackaged or deployed.
Diagnostic checkpoint schema/task identifiers deliberately differ from release checkpoints.

The neural spectral-head direction is supported by the primary
[iSTFTNet paper](https://arxiv.org/abs/2203.02395): learned spectral quantities can
be converted to a waveform with an inverse STFT. This simplified implementation is
not iSTFTNet. The optional residual-convolution trial was informed by the
[HiFi-GAN paper](https://arxiv.org/abs/2010.05646); no adversarial discriminator,
pretrained weights, or HiFi-GAN quality claim is included here.

## F. Tests and reproducible commands

Run from `models/nabra/variants/nano/mvp` with the repository virtual environment:

```bash
python -m pip install -r requirements-diagnostics.txt
NABRA_TEST_MPS=1 python -m unittest discover -s tests -v
```

Final actual result: **72 tests passed, 0 failed, 0 skipped, 6.827 seconds** with MPS
checks enabled. This includes the existing ASR/TTS/human-speech/streaming suite and
ten new tests. Logs: `quality-recovery-audit/full-suite-refinement.log`.
Earlier complete runs passed 71 tests before the frozen-vocoder test was added.

New coverage includes identical ground-truth Mel extraction, differentiable features,
Mel/vocoder length consistency, residual and spectral heads, attention masking,
stop/cap behavior, teacher-forced versus free-running equality under matched limits,
gradients, deterministic checkpoint restoration, exact split-run versus uninterrupted
CPU resume, optimizer health, data mismatch rejection, PCM validity, weight dependency,
frozen-vocoder preservation/resume validation, and CPU/MPS gradient paths.

Failed development checks are retained: the first diagnostic suite ran seven tests
and reported four errors because its helper `train()` collided with a `Sandbox.train`
list. A repair command initially used a root-relative file path from the MVP working
directory, raised `FileNotFoundError`, and the second test attempt repeated the same
helper errors. Renaming the helper to `train_diagnostic()` fixed this. The third
seven-test run passed; later expanded full suites also passed. These were test-harness
errors, not a training NaN or evidence that the old noise issue was solved by tests.

Below are the actual bounded experiment commands, expressed portably. Use new run IDs
when rerunning: existing run directories are intentionally never overwritten.

```bash
ROOT=$(git rev-parse --show-toplevel)
ART="$ROOT/storage/artifacts/nabra/tts/nano/mvp"
AUDIO="$ROOT/intake/speech/train/clip001.wav"
OLD="$ART/tts-mvp-verification-002/checkpoints/step_000500.pt"
python -m scripts.tts_overfit_vocoder --audio "$AUDIO" --steps 1000 --initialize "$OLD" --run-id recovery-vocoder-original-001 --save-interval 250
python -m scripts.tts_overfit_vocoder --audio "$AUDIO" --steps 1500 --model-options '{"channels":64,"min_channels":16,"residual":true}' --run-id recovery-vocoder-residual-001 --save-interval 250
python -m scripts.tts_overfit_vocoder --audio "$AUDIO" --steps 1000 --model-options '{"channels":128,"head":"spectral"}' --run-id recovery-vocoder-spectral-001 --save-interval 250
python -m scripts.tts_overfit_vocoder --audio "$AUDIO" --steps 4000 --model-options '{"channels":128,"head":"spectral"}' --resume "$ART/recovery-vocoder-spectral-001/checkpoints/step_001000.pt" --objective phase --run-id recovery-vocoder-phase-001 --save-interval 500
python -m scripts.tts_overfit_acoustic --audio "$AUDIO" --steps 1500 --learning-rate 0.0005 --initialize "$OLD" --run-id recovery-acoustic-001 --save-interval 500
VOC="$ART/recovery-vocoder-phase-001/checkpoints/step_004000.pt"
python -m scripts.tts_overfit_full --acoustic-checkpoint "$ART/recovery-acoustic-001/checkpoints/step_001500.pt" --vocoder-checkpoint "$VOC" --output "$ART/recovery-full-001/samples"
python -m scripts.tts_overfit_acoustic --audio "$AUDIO" --steps 2500 --learning-rate 0.0001 --resume "$ART/recovery-acoustic-001/checkpoints/step_001500.pt" --vocoder-checkpoint "$VOC" --run-id recovery-acoustic-waveform-001 --save-interval 500
python -m scripts.tts_overfit_full --acoustic-checkpoint "$ART/recovery-acoustic-waveform-001/checkpoints/step_002500.pt" --vocoder-checkpoint "$VOC" --output "$ART/recovery-full-002/samples"
```

The original checkpoint is a preserved local artifact, not a Git-hosted dependency.
The models, audio, figures and audit logs stay under ignored local artifact storage.
Source/config/test/documentation changes can be versioned independently.

## G. Training data

One existing correctly labeled synthetic Arabic recording, **69,947 samples /
4.3716875 seconds**, is used throughout recovery. It was already generated with
macOS `say`, Majed voice, before this task. No new OS speech generation was performed.
The original transcript and waveform bytes were preserved. This is not human speech.

- Audio SHA-256: `d7e2627f95bd7df534b1b8c917a471bf13f9d6f8da3eb19c1261193a002c6466`.
- Transcript SHA-256: `090a3bafbef5a2499e9de7e4e4c36e6f06ad805ae25c4b1c2bd5df502eed94a5`.
- Provenance: `intake/speech/synthetic-samples.json`.
- License/consent for commercial corpus use: not established; no invented permission.
- No held-out split or external benchmark claim applies to this single-pair overfit.
- The existing optional FLEURS human recordings are used only by regression tests;
  they were not added to these recovery training runs. Their pinned provenance and
  CC-BY-4.0 attribution remain in the fixture metadata.

Natural human Arabic synthesis quality needs properly sourced, licensed human speech,
representative speakers and held-out evaluation. Recognizing a memorized synthetic
sentence is a narrower milestone.

## H. Performance

All recovery training used CPU, two threads, deterministic seed 42. Step limits
were fixed per run; no indefinite training job remains running. Resume target step
counts are cumulative, so the table reports actual new updates.

| Run | New steps | Trainable parameters | Observed training elapsed time |
| --- | ---: | ---: | ---: |
| `recovery-vocoder-original-001` | 1000 | 25,565 | 33.40 s |
| `recovery-vocoder-residual-001` | 1500 | 107,073 | 299.54 s |
| `recovery-vocoder-spectral-001` | 1000 | 468,114 | 38.51 s |
| `recovery-vocoder-phase-001` | 3000 | 468,114 | 85.92 s |
| `recovery-acoustic-001` | 1500 | 210,433 | 246.98 s |
| `recovery-acoustic-waveform-001` | 1000 | 210,433 | 162.04 s |

Total new optimizer updates: 9,000. Summed recorded training durations:
866.40 s. Some tasks overlapped and
the timings include periodic sampling/checkpointing; they are not clean benchmark
comparisons. Final synthesis uses 210,433 acoustic plus 468,114 vocoder parameters.
During final refinement, only the acoustic parameters were trainable.

MPS one-step training/backpropagation was exercised by tests, including the spectral
head and differentiable CPU STFT/iSTFT fallbacks. These fallbacks mean this is not
an accelerator-only path. CUDA selection is implemented but no CUDA device was
available for verification. Peak memory was not measured. Long-run MPS determinism,
GPU speed/scaling and larger-corpus memory behavior are unverified.

## I. Remaining limitations

1. Residual hiss remains in the reviewed output. The user explicitly confirmed
   clear words and reduced hiss, not complete audio quality.
2. Direct human review of A's standalone reconstruction WAV remains pending.
3. One-pair overfit is not Arabic language generalization. Different text changes
   output, but that alternative recording has not been validated for correct words.
4. Spectral supervision and a larger head resolve the observed one-pair failure;
   capacity, objective, data and robustness still need systematic ablation and scale tests.
5. Phonetic detail, residual noise and naturalness require broader data and evaluation;
   adversarial/perceptual objectives or a stronger vocoder may help but were not claimed
   or silently substituted in this experiment.
6. No new experimental or production release is approved. Preserve the prior release
   for audit, but its user-confirmed noise result disqualifies it as usable speech.
7. Human-quality tests need licensed natural recordings, speaker/source-disjoint data,
   phonetic coverage and multiple listeners. Larger GPUs are for scaling those studies;
   this one-sentence learning proof itself ran on the available CPU.

## J. Final verdict

| Requested verdict | Status | Scope / reason |
| --- | --- | --- |
| Neural vocoder reconstruction | **PENDING HUMAN REVIEW** | Technical reconstruction improved to 22.68 dB SNR; this standalone WAV has not been listened to by the user. |
| Text-to-Mel learning | **PASS** | Free-running reconstruction, learned attention, duration and stopping verified on the single pair. |
| Full TTS synthesis | **PASS** | The user confirmed clear words from the final output generated through both trained neural components. Single-sentence overfit only. |
| Arabic intelligibility | **PASS** | User explicitly confirmed the prompted Arabic words are clear in the reviewed final WAV. This is not a general Arabic benchmark or clean-audio rating. |
| Experimental release readiness | **FAIL** | Residual hiss and narrow overfit validation remain. No new release was created or old release replaced. |

**The main learning proof is achieved:** Nabra's own neural weights can produce at
least one recognizable Arabic sentence. Its TTS MVP is not yet a general-purpose,
clean-quality speech product. The verified sentence and remaining quality defect
are both recorded; neither is hidden by a new version number.

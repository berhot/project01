# Sora Nano RGB pixel diffusion — Dataset v2 result (2026-11-03)

**Status: failed semantic/prompt-control acceptance gate.** This is a
user-reported Mac experiment, not an independently replicated benchmark.

## Experiment

- Dataset: generated 32×32 RGB images of 4 colors × 3 shapes (12 classes),
  each with English/Arabic phrase variants
- Model: `rgb-unet-text-v1` with Dataset v2 expanded word vocabulary
- Training: 10,000 steps, 12 images/batch, 32 base channels, CFG text
  dropout probability 0.15, CPU (2 threads)
- Checkpoint (local user's Mac, **not committed**):
  `storage/artifacts/sora/nano/mvp/pixel-20261103T001206Z-69742dd9/checkpoint.pt`
- Last logged noise MSE: 0.016549; several recent steps <0.01
- Evaluation: deterministic full-step DDPM posterior mean, scales
  1.0 / 3.0 / 5.0 / 7.5, random seeds 42 and 123
- Human image review: **not received/verified**
- Automated tests on user's Mac: 34 passed before the long training run

## User-reported toy heuristic scores

Each scale and seed covers 24 PNGs (12 concepts × 2 languages).
All listed seed pairs yielded the same summary percentages.

| CFG scale | Color accuracy | Shape accuracy | Strict prompt accuracy | Bilingual both correct |
|---|---:|---:|---:|---:|
| 1.0 | 25.00% | 33.33% | 8.33% | 8.33% |
| 3.0 | 25.00% | 33.33% | 8.33% | 8.33% |
| 5.0 | 50.00% | 33.33% | 16.67% | 16.67% |
| 7.5 | 58.33% | 41.67% | 25.00% | 25.00% |

Bilingual *prediction consistency* ranged from 91.67% to 100%,
but that is not evidence that the predictions were correct.
At CFG 7.5 only **6 / 24** output labels were classed as correct for
each seed; bilingual-both-correct was only **3 / 12** class pairs.

For an always-identical prediction on a perfectly class-balanced
12-class evaluation, strict prompt accuracy is 1/12=8.33%:
guidance 1.0 and 3.0 scored **exactly** at that baseline.
The identical aggregate results for seeds 42/123 justify investigating
a dominant-class or weak-conditioning failure. They do not prove which
failure occurred; inspect per-class mistakes and actual images.

## Do not claim success from low loss

This low MSE is *noise-prediction accuracy*, not conditional image
quality. The metrics are low-complexity image heuristics; blurry edges,
background variations and wrong shapes can cause inaccurate labels.
A manual review of the saved 24-image grids is required.

## Diagnostic: inspect existing outputs without retraining

After pulling the updated branch, run:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

python3 -m scripts.pixel_diagnose \
  --evaluation-dir /tmp/sora-v2-eval-1793665643 \
  --output /tmp/sora-v2-audit-1793665643.json
```

This does **not** create model outputs, train or modify weights, or
trigger GitHub Actions. It reads the existing PNGs and reports,
computes a predicted-class histogram, color/shape confusion,
wrong-but-agreeing EN/AR pairs and per-pair RGB differences.

**Next gate:** inspect the diagnostic and real evaluation grids before
altering the generator. Possible causes to investigate include overly
weak prompt dependence in denoising at high noise, poor 12-class shape
representation, evaluation heuristics, and data/label correlations.
Do not pay for another 10,000-step rerun or introduce real image data
until that failure is distinguished.

The previous v1 six-class pilot reached 91.67% strict heuristic
prompt accuracy at CFG 7.5 on two seeds, but v1/v2 have different
numbers of classes and must not be compared as an equal-difficulty
benchmark.

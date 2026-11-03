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

## New controlled experiment: v1→v2 warm-start (2026-11-03)

User-reported Mac outcome for **2,500-step v2 fine-tuning**, initialized
from the earlier **10,000-step v1 six-class RGB checkpoint**.

- Source v1 checkpoint:
  `storage/artifacts/sora/nano/mvp/pixel-20261102T230751Z-185cee42/checkpoint.pt`
- New model checkpoint (saved locally; **not committed**):
  `storage/artifacts/sora/nano/mvp/pixel-20261103T004620Z-c92b5c0b/checkpoint.pt`
- Configuration: v2 12 classes, base channels 32, 100 diffusion steps,
  batch 12, CFG dropout 0.15, LR 0.0001, CPU 2 threads
- Transfer: v1 RGB denoiser + original word embeddings; newly introduced
  v2 words initialized from old learned embeddings and fine-tuned
- Last logged training noise MSE: **0.009091**
- Evaluated with same v2 12-class toy heuristic; 24 model-generated
  images × two seeds (42 and 123) × four guidance scales
- Full tests on user's Mac before warm-start: **41 passed**
- **No warm-start image grid attached yet for human review**

### Direct v2-to-v2 comparison (guidance scale 7.5)

| Metric | From-scratch 10,000 steps | Warm-start 2,500 steps |
|---|---:|---:|
| Color accuracy | 58.33% | **95.83%** |
| Shape accuracy | 41.67% | **58.33%** |
| Strict prompt accuracy | 25.00% | **58.33%** |
| Both English and Arabic correct | 25.00% | **58.33%** |
| EN/AR prediction agreement | 91.67% | 91.67% |

Scores above are the same percentages on both random seeds for
guidance scale 7.5. Prompt accuracy rose from 6/24 to 14/24;
both-language-correct concepts rose from 3/12 to 7/12 per seed.

**Improvement is supported, but the causal contribution of transfer
alone cannot be isolated:** the two runs also differed in learning rate,
number of steps, and training initialization. We need a matched
ablation to claim exact transfer benefit.

At guidance 5.0, warm-start strict prompt accuracy was 58.33%
for both seeds; at guidance 3.0 it was 37.5% and 50.0%; at
guidance 1.0 only 8.33%. In v2, shape accuracy of 58.33% is now
the major bottleneck; better color control cannot compensate for
wrong circles/squares/triangles. All scores remain *toy heuristics*.

### Next no-retrain diagnosis

Pull the updated diagnostics and inspect actual PNGs:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

python3 -m scripts.pixel_diagnose \
  --evaluation-dir /tmp/sora-v2-warmstart-1793667091 \
  --output /tmp/sora-v2-warmstart-shape-audit.json
open /tmp/sora-v2-warmstart-1793667091/guidance_7p5_seed_42/preview_grid.png
```

New diagnostic fields `shape_accuracy_by_expected_shape`,
`prompt_accuracy_by_expected_shape`, and
`prompt_accuracy_by_concept` will distinguish a triangle-only
failure from broad shape collapse. **Stop before another long run**
and determine which geometry is failing. The tool only reads
already-generated PNGs/reports, not model weights.

## Triangle curriculum trial: 800 additional v2 steps (2026-11-03)

User loaded the 2,500-step v2 warm-start model, reset the optimizer,
and trained an **independent new run for 800 steps** at LR 0.00005 with
50% triangle examples. Original and new checkpoints remain distinct.

- Source: `pixel-20261103T004620Z-c92b5c0b/checkpoint.pt`
- New checkpoint: `pixel-20261103T063642Z-168cc81f/checkpoint.pt`
- Last training noise MSE: 0.016052
- At CFG **5.0**, both seed 42 and 123:
  - Color accuracy: 95.83%
  - Shape accuracy: 66.67%
  - Strict prompt accuracy: 62.50% (15/24)
  - Bilingual both correct: 58.33%
  - Diagnostic per-shape: **circles 100%, squares 100%, triangles 0%**
- At CFG 7.5:
  - Color 95.83%; shape 58.33% (seed 42) / 62.50% (seed 123)
  - Prompt accuracy 54.17% / 58.33%
  - **Triangles 0%** at both seeds
- The actual 24-image CFG 7.5 preview was shared for visual review.
  It clearly shows **no recognizable triangles**; nearly every
  triangle prompt becomes a square/rectangular colored object.

**Conclusion: triangle oversampling alone did not solve geometry control.**
CFG 5 gave better overall 12-class score than 7.5. Both model training
and inference execute normally, but the new triangle semantics
have not been learned. This is not a metric-only misclassification.

## Next bounded architecture experiment (unvalidated)

The new `rgb-unet-spatial-text-v2` architecture is a **separate,
opt-in** model: trainable per-pixel attention over individual Arabic/
English words at 32×32 and 16×16, instead of only averaging all
words into one global vector. Its attention output begins as a zero
residual so existing trained RGB weights can be transferred exactly.
Additionally, it offers an RGB silhouette Dice training loss computed
from existing **training-image pixels** at lower-noise timesteps.
No shape renderer is used at inference.

The existing `rgb-unet-text-v1` model and all checkpoints are
unchanged. The new experiment must prove that it can generate
triangles without forgetting circles, squares, or colors. The first
test is 2 training steps (unit test) and 100 CLI steps, **not**
another 10,000 steps. The next quality comparison would use a
same-dataset 800-step experiment (LR 0.00005) from the original 2,500-step
warm-start checkpoint. Only undertake it if all automated tests pass.

This is a research hypothesis, not a fix already shown to work.

## Spatial text + silhouette 800-step trial: still zero triangles

Reported by user on 2026-11-03 after successful 800-step spatial
architecture training from the original 2,500-step v2 base:

- Checkpoint (Mac only; never commit):
  `storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt`
- New model: `rgb-unet-spatial-text-v2`, 32 channels, 100 diffusion
  timesteps, 50% triangle training examples, silhouette Dice weight 0.15,
  CFG dropout 0.15, learning rate 0.00005.
- **CFG 5.0** on seeds 42/123: color 100%, shape 66.67%,
  strict prompt 66.67%, bilingual both correct 66.67%.
- Per-shape diagnostic: **circle 100%, square 100%, triangle 0%**.
  Per-concept: all four triangle colors have 0% prompt accuracy.
- **CFG 7.5**: color 95.83%, shape 58.33%, strict prompt
  54.17%; triangle remains 0%.
- Actual 24-image CFG 5.0 preview shows eight correctly colored
  circles/squares and four colors of rectangle-like objects for
  triangle requests, in both Arabic and English.
- Noise MSE at final step 0.010998; silhouette training Dice loss
  0.081245; these cannot establish generative shape accuracy.

**Conclusion:** New spatial token attention + silhouette loss slightly
improved overall toy accuracy but **did not teach triangles**. This
isolates the persistent category failure under the current evaluation
and deterministic posterior-mean sampling. Keep the new model as
an experimental checkpoint, not a successful foundation release.

### No-retraining test of a sampling hypothesis

Previous evaluations used `posterior_mean`, which uses only the
reverse DDPM mean and **omits** Gaussian posterior variance.
This is a deterministic heuristic, not a full DDPM sample.
The experimental `--sampler ancestral` adds that stochastic variance
at all 100 reverse steps using a seeded torch.Generator; `ddim`
remains an alternative and the old sampler is unchanged.

The new `--shared-noise` flag also forces the **same starting Gaussian
noise** across all 12 English/Arabic prompts (the old benchmark
used a different seed for each class). Thus a change from
`blue square` to `blue triangle` can be compared directly,
rather than confounded with a different initial noise image.
This is a NEW comparison protocol; do not directly equate its
score with the old seed+class benchmark.

Run only after local tests pass:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

CKPT="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$CKPT" || { echo "Missing spatial checkpoint"; false; }
OUT="/tmp/sora-ancestral-shared-$(date +%s)"

python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler ancestral \
  --guidance-scales 5 --seeds 42 --shared-noise \
  --output-dir "$OUT"

open "$OUT/guidance_5_seed_42/preview_grid.png"
python3 -m scripts.pixel_diagnose \
  --evaluation-dir "$OUT" \
  --output "$OUT/ancestral_diagnosis.json"
```

If ancestral DDPM still fails to produce visually recognizable
triangles, stop sampler tweaking and design a genuinely
shape-discriminative denoising objective / training ablation
before another long GPU or CPU run. A sampler change alone is
not proof that learned shape semantics exist.

## Sampler regression fix (2026-11-03)

A user ran the new `test_pixel_ancestral.py` and got 1 Gaussian
draw instead of 8. Investigation found a reverse timestep loop typo
in the **recent** sampler update:

```python
# BROKEN (empty range)
for t in range(total - 1, -1):
    ...

# FIXED (all T reverse steps)
for t in range(total - 1, -1, -1):
    ...
```

The wrong loop made both `ancestral` and `posterior_mean` return
clipped initial Gaussian noise without performing denoising.
The sampler implementation was corrected and tests now assert
**T actual denoiser calls with timesteps T-1,T-2,...,0**,
plus T total random draws for ancestral (one initialization and T-1
reverse-noise draws) versus one random draw for deterministic mean.

**Crucial provenance:** inspecting historical commits shows the
reverse loop was correct at the time of the user's prior v2 spatial
model image evaluations. The incorrect loop was introduced during
the later sampler extension and was caught before the user ran
the new ancestral/shared-noise comparison. Hence prior user-supplied
trained checkpoint weights and reported visual grids do not need
to be discarded because of this regression.

The corrected new tests are **awaiting a Mac run**, and this
does NOT establish ancestral sampling will generate triangles.
The old checkpoints remain untouched; GitHub workflow remains
manual-only.

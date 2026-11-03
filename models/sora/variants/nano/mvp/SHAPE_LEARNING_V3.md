# Sora Nano — Shape Learning V3 (experimental)

This is a small, opt-in **training recipe on the existing
`rgb-unet-spatial-text-v2` architecture**, NOT a new foundation model or
proof of successful triangle generation. Sora Nano still produces 32×32
synthetic single-shape images and has not met release-quality requirements.

## Why V3 is needed

The user's actual blue circle looked angular and the yellow triangle was a
square despite older toy heuristic metrics reporting correct circles.
Prior investigation establishes:

- Baseline fixed sampler: 66.67% 24-prompt accuracy, **0% triangles**.
- High-noise t=80/99 reconstruction: **0% triangles** in both languages
  for red partial reverse, whereas t=20/50 preserved them.
- The model can recover genuine triangle masks when given a lightly-noised
  real triangle. The failure is in text-conditional high-noise generation,
  not in the availability of training-target triangle geometry.
- Dynamic x0 clipping degraded results and produced **0% triangles**.
- The earlier paired-midpoint objective damaged circles/colors/squares.

## What the code changes

The new `--shape-v3` training flag:

1. Requires the **known-good v2 spatial checkpoint** as `--init-from-v2`.
2. Requires `--preserve-weight > 0` with a frozen baseline teacher so old
   circle/square and unconditional predictions remain protected.
3. Requires batches divisible by 12, without `--triangle-fraction`. Each
   12-image batch contains all 12 colors × shapes; consecutive steps
   alternate Arabic and English for each concept.
4. Disallows `--paired-shape-weight`, because the old midpoint-based
   objective caused unacceptable regression.
5. Keeps the original RGB noise MSE plus low-noise silhouette Dice loss.
   Adds a **valid-forward-q(x_t|x0) objective** at t/T between 0.55 and
   0.85, with both correct and wrong shape words evaluated on the **same**
   genuine noisy RGB pixels and the **same** epsilon target.
6. Applies a small weight to correct-caption noise MSE and an explicit
   ranking margin to favor the correct shape caption. This is an
   experiment and might still fail; tests can establish code correctness,
   not geometric quality.
7. Saves a **new** run under central
   `BERHOT/storage/artifacts/sora/nano/mvp/pixel-<timestamp>-<id>/`,
   with `shape_learning_recipe` and hyperparameter provenance. The
   baseline checkpoint is never overwritten.

New strict `scripts.pixel_shape_quality` evaluates actual saved
generated PNGs against ideal **circle / square / both triangle directions**
silhouette references. The templates are used solely by the *evaluator*,
not the model's training inference. It demands a meaningful IoU and
separation from competing shapes, no cropped shape at the image border,
and correct color. It is not human visual validation.

## Stage A — Validate implementation (no training)

From Sora MVP:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git fetch origin feat/image-nano-foundation-mvp-20261102
git merge --ff-only FETCH_HEAD
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
```

The previous Mac suite passed **101** tests; the V3 change adds
**10 new tests**. Expect 111 tests. Stop if tests fail; share the
traceback. GitHub source code was reviewed, but this latest V3 code
has **not been run on the user's Mac**.

## Stage B — Limited 20-step pilot only

Run this only after Stage A passes:

```bash
BASE="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$BASE" || { echo "Baseline checkpoint missing"; false; }

set -o pipefail
LOG="/tmp/sora-v3-pilot-$(date +%s).log"

python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 \
  --conditioning spatial \
  --shape-v3 \
  --init-from-v2 "$BASE" \
  --preserve-weight 1.0 \
  --preserve-null-weight 0.50 \
  --paired-shape-weight 0 \
  --v3-noise-weight 0.15 \
  --v3-contrast-weight 0.10 \
  --v3-margin 0.003 \
  --v3-low-fraction 0.55 \
  --v3-high-fraction 0.85 \
  --silhouette-weight 0.15 \
  --steps 20 --batch-size 12 \
  --learning-rate 0.00001 \
  --base-channels 32 \
  --diffusion-steps 100 \
  --cfg-drop-prob 0.15 \
  --seed 42 --device cpu \
  --cpu-threads 2 --log-every 10 | tee "$LOG"
```

No oversampling; 4 circles, 4 squares, and 4 triangles in each batch.
This pilot **cannot be presumed to generate better triangles**.
Inspect the resulting printed checkpoint path and training metrics.

## Stage C — Actual image/shape acceptance (ONLY after pilot)

Evaluate *the same* baseline and candidate prompt set, seed 42, CFG 5,
fixed clipping and posterior-mean sampling. Store both evaluation
directories centrally, with unique run IDs. These are learned RGB model
outputs, not synthetic dataset renderings.

```bash
BASE="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
# Replace with the ACTUAL checkpoint printed by V3 training:
V3_CKPT="/absolute/path/to/new/v3/checkpoint.pt"
OUT="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp"

python3 -m scripts.pixel_metrics \
  --checkpoint "$BASE" --guidance-scales 5 --seeds 42 \
  --sampler posterior_mean --x0-clipping fixed \
  --output-dir "$OUT/pixel-v3-review-base-$(date +%s)"

python3 -m scripts.pixel_metrics \
  --checkpoint "$V3_CKPT" --guidance-scales 5 --seeds 42 \
  --sampler posterior_mean --x0-clipping fixed \
  --output-dir "$OUT/pixel-v3-review-candidate-$(date +%s)"
```

Then use the returned `guidance_5_seed_42` directories from both
evaluation reports to compare strict geometry and the existing toy
model-promotion quality gate:

```bash
python3 -m scripts.pixel_shape_quality \
  --baseline-dir "<BASE_EVAL>/guidance_5_seed_42" \
  --evaluation-dir "<V3_EVAL>/guidance_5_seed_42"

python3 -m scripts.pixel_quality_gate \
  --baseline-report "<BASE_EVAL>/guidance_5_seed_42/report.json" \
  --candidate-report "<V3_EVAL>/guidance_5_seed_42/report.json"
```

Both should be rejected unless triangles appear while circles,
squares, colors and bilingual behavior remain strong. In addition,
review the actual 24 generated images for angular circles, crooked
squares, bad colors and triangles. A geometry score is NOT a human
visual assessment. Accept for more research only on repeatable
improvements; never automatically publish to `storage/releases`.

## Avoiding accidental regressions

- Do not run 800/2500/10000-step training until the 20-step pilot shows
  a promising trend. Increase only with controls and after user review.
- Do not change the original fixed posterior-mean sampler or baseline.
- Do not use the failed paired midpoint objective with V3.
- Do not assume a low noise MSE means valid geometry.
- No external image models, procedural shape output, or fabricated
  evaluation scores are allowed.
- PR #10 stays draft; GitHub Actions stays manual-only.

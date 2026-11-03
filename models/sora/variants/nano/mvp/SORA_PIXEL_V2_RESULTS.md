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

## Matched-noise sampler experiment: no triangle recovery (2026-11-03)

The user evaluated the **same trained 800-step spatial-text checkpoint**
(`pixel-20261103T065647Z-ee054d71/checkpoint.pt`) with identical
initial noise for every shape/color prompt, at CFG 5.0 and seed 42.

| Metric | Posterior mean | Ancestral DDPM |
|---|---:|---:|
| Color accuracy | 100% | 100% |
| Shape accuracy | 66.67% | 41.67% |
| Strict prompt accuracy | 66.67% | 41.67% |
| Bilingual both correct | 66.67% | 41.67% |
| Triangle accuracy | 0% | 0% |

Ancestral report: `/tmp/sora-ancestral-shared-1793709484`; posterior
mean report: `/tmp/sora-mean-shared-1793709484`.
Square-vs-triangle same-noise foreground-mask IoU was **0.8172**
(ancestral). The screenshot shows distorted but still non-triangular
blobs in the triangle rows, while some squares also degraded.

**Decision:** ancestral noise does not solve geometry control in
this checkpoint. Retain the prior deterministic posterior-mean
baseline, with its limitations clearly labeled.

## Next test: matched-layout square/triangle counterfactuals (unvalidated)

A new opt-in **training objective** creates paired procedural
TRAINING samples with the **same color, position, size and
background** but different square/triangle silhouettes and different
Arabic/English shape words. Both conditioned versions receive
the **same noisy RGB tensor** at a moderate diffusion timestep.
Their mathematically valid denoising targets differ, making a
shape-blind noise predictor wrong on at least one member of the
pair. An added differentiable silhouette-difference term focuses
on the missing triangular corner regions. Ordinary RGB noise
training/CFG remain in the objective, and the inference sampler
is unmodified. No renderer is used at image inference.

Opt in via `--paired-shape-weight 0.35 --paired-shape-pairs 4
--paired-silhouette-weight 0.5` with `--conditioning spatial
--dataset-version v2`. The default weight is **0**, so previous
training remains unchanged. Checkpoint metadata records these
choices and preserves the original spatial source checkpoint.

The related **read-only** `scripts.pixel_shape_probe` measures
how the pretrained denoiser's noise predictions respond to
switching square to triangle while holding noisy pixels,
background, color, and target geometry aligned. The probe
reports expected/predicted noise-difference RMS, their ratio
and cosine alignment. These are synthetic-diagnostic metrics,
not actual generated-image shape accuracy.

**No full training was run and no improvement is claimed.**
The new tests and probe must first be executed on the user's Mac.
If they pass, attempt a 50-step integration smoke, followed by
a bounded 300-step trial only if the user approves further compute.
A triangle generated at the correct color in both languages,
across multiple seeds, remains the acceptance gate.

## Paired square/triangle experiment: 50-step integration success (2026-11-03)

User's Mac: **62/62 tests passed** on the first paired-loss implementation.
The existing 800-step spatial checkpoint was probed (read-only)
on eight held-out square/triangle pairs:

- Source checkpoint:
  `pixel-20261103T065647Z-ee054d71/checkpoint.pt`
- Initial probe: expected epsilon-difference RMS **0.369722**;
  predicted difference RMS **0.034817**; ratio **0.094170**;
  mean cosine alignment **−0.058700**.
- Those initial eight pairs covered every color, but each color
  appeared in only one language. Therefore the original numbers
  support weak shape-conditioning response but are not a fully
  bilingual balanced evaluation. The probe has now been corrected
  to use matched validation batches at steps 1 and 3 for eight pairs,
  giving **each of four colors once in English and once in Arabic**.
  Re-run both checkpoints using the revised probe; **do not directly
  compare revised ratios with the older 0.094170**.
- First opt-in training test: 50 steps from the source checkpoint,
  new separate checkpoint:
  `pixel-20261103T124924Z-b7835e13/checkpoint.pt`
- Paired shape-difference loss logged at step 1: **0.369545**;
  step 50: **0.148050**; these are from different training batches.
- Final logged regular noise MSE: **0.015162**, paired noise MSE:
  **0.041782**; logged total loss at step 50: **0.082147**.
- **No post-training validation probe or actual generated 24-image
  gallery has been supplied.** Nothing yet proves triangle
  generation or improvement.

### Next verification without retraining

After pulling latest changes, rerun all tests, then use the same
balanced probe protocol for **both** source and fine-tuned checkpoints:

```bash
BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
NEW="storage/artifacts/sora/nano/mvp/pixel-20261103T124924Z-b7835e13/checkpoint.pt"

python3 -m scripts.pixel_shape_probe --checkpoint "$BASE" \
  --pairs 8 --seed 42 --output "/tmp/sora-shape-base-balanced-42.json"
python3 -m scripts.pixel_shape_probe --checkpoint "$NEW" \
  --pairs 8 --seed 42 --output "/tmp/sora-shape-finetuned-balanced-42.json"
```

Then generate one 24-image CFG-5 posterior-mean preview from NEW,
with independent normal per-class seeds to stay comparable with prior
model evaluation; visually inspect triangle silhouettes. This is
still 32×32 synthetic single-object research, not general image AI.

## Critical regression: 50-step matched shape objective harms generation (2026-11-03)

User's Mac **63/63 tests passed** (including the bilingual-balanced probe).
The previously trained spatial-v2 800-step baseline and the new
50-step paired-loss checkpoint were compared using identical validation
pairs, CFG scale 5.0, posterior-mean sampler and seed 42.

| Measurement | Base spatial model | 50-step paired-loss model |
|---|---:|---:|
| Matched prompt difference RMS | 0.032862 | 0.217834 |
| Expected difference RMS | 0.367838 | 0.367838 |
| Prompt difference / expected | 0.089338 | 0.592200 |
| Prompt response cosine alignment | -0.009628 | 0.633999 |
| Color accuracy (24 generated images) | 100% | 62.50% |
| Circle accuracy | 100% | **0%** |
| Square accuracy | 100% | 100% |
| Triangle accuracy | 0% | **0%** |
| Strict prompt accuracy | 66.67% | **29.17%** |
| Bilingual both correct | 66.67% | 25.00% |

The actual contact sheet shows nearly blank/pale circles, colorful
rectangles where squares are requested, and incoherent noisy blocks
where triangles should be. **This is a major regression, not a success.**

- Source best spatial checkpoint:
  `storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt`
- Failed 50-step paired checkpoint (preserve for research, not best):
  `storage/artifacts/sora/nano/mvp/pixel-20261103T124924Z-b7835e13/checkpoint.pt`
- User outputs: `/tmp/sora-paired-review-1793710487`
- Training paired weight was 0.35, larger than the ordinary
  low-MSE denoising signal. The paired task also uses a special
  midpoint-noise input rather than normal forward diffusion images.
  Those mechanisms are **plausible** causes of catastrophic forgetting,
  but a matched ablation is necessary to distinguish them.

### New safeguards (code committed; Mac execution pending)

An opt-in frozen checkpoint **teacher-preservation** objective protects
circle/square predictions and the null CFG branch on ordinary
q(x_t|x0) noised RGB training examples. This is teacher regularization,
not replacing the true pixel diffusion model, and it does not insert
procedural shapes at inference. Enable with
`--preserve-weight 1.0 --preserve-null-weight 0.5`, requiring a matching
spatial-v2 source via `--init-from-v2`. Defaults remain off.

In the next trial use **paired weight 0.02, not 0.35**, and stop at
**20 steps**. The no-retraining `scripts.pixel_quality_gate` compares
candidate vs source reports using the SAME prompts/seeds/sampler/CFG.
It requires a recognizable triangle heuristic improvement while
retaining circles, squares, colors and total prompt correctness.
The gate itself is NOT visual proof; inspect the actual 24 PNGs.

**Do not continue to 300 or 10,000 steps if either the gate fails
or the visual triangles are still missing.** Preserve all checkpoints;
PR remains draft and GitHub Actions manual-only.

## Frozen teacher + tiny paired loss: preserved old classes but zero triangles (2026-11-03)

User Mac ran **71 tests successfully** and completed 20-step spatial-v2
fine-tune using frozen old-shape teacher, paired-shape weight 0.02,
teacher preservation weight 1.0, learning rate 0.00002.
The checkpoint remains separate:
`storage/artifacts/sora/nano/mvp/pixel-20261103T131058Z-65110681/checkpoint.pt`.

The user provided actual 24-image CFG-5, seed-42 generated contact
sheet. Circles and squares are visually preserved, but **all eight
English/Arabic triangle examples still resemble squares or blocks**.
The toy segmentation results remained:
- **100% color**
- **100% circles**
- **100% squares**
- **0% triangles**
- **66.67% strict prompt** (16/24 outputs)
- **66.67% bilingual both correct** (8/12 pairs)

This matches the good 800-step spatial baseline on the same model
benchmark. The frozen teacher therefore prevented the severe earlier
regression, but the candidate **must not be promoted as solving
triangles**. The shape-focused training alone is not ready for more
compute.

A compatibility bug in the new `scripts.pixel_quality_gate` rejected
comparing the older base report (which predated the
`shared_noise_across_concepts` key) to the new report (explicit
`false`). The earlier evaluation always used independent per-prompt
seeds, so an omitted legacy key is semantically `false`. Fixed the
gate to default the missing key to `false` and added regression tests
that still reject explicit `true` or an invalid string value.
**The corrected tests have not yet been run on the user's Mac.**

After pulling, rerun the gate on the existing saved reports; no
training and no image regeneration are needed:

```bash
python3 -m unittest discover -s tests -p 'test_pixel_quality_gate.py' -v
python3 -m scripts.pixel_quality_gate \
  --baseline-report /tmp/sora-spatial-eval-1793689086/guidance_5_seed_42/report.json \
  --candidate-report /tmp/sora-safe20-eval-1793711582/guidance_5_seed_42/report.json
```

Expected outcome: print a valid JSON decision,
`accepted_by_toy_gate: false`, because triangles remain 0%; exit
status **2** is an intentional gate rejection, not an exception.
All original checkpoints remain in place and PR #10 stays draft.

## 74 passing tests, gate correctly rejects preserved-but-nontriangular run

The user reran the updated Mac tests: **74/74 passed**. After the
legacy shared-noise metadata fix, the strict toy quality gate returned
valid JSON with `accepted_by_toy_gate: false`:
- Circle, square and color preservation checks all passed.
- Overall prompt accuracy was unchanged at **0.6667**.
- **Triangle accuracy remained 0.0** in both baseline and 20-step
  frozen-teacher candidate.

The model therefore **must not be promoted**. Its preservation
result is still useful as a separate research checkpoint.

### Next read-only diagnosis: actual forward-DDPM shape-word sensitivity

Inspection of `src/pixel_dataset_v2.py` confirms that triangle training
targets are actual PIL-filled three-vertex polygons; triangle labels
and tokens exist in both Arabic and English. This alone cannot
establish that the model learned them.

New `scripts.pixel_conditioning_audit` avoids the unnatural
midpoint-noise paired objective. It uses **ordinary held-out forward
diffusion** on actual square/triangle validation images:
- Keep the SAME RGB x_t, timestep, color and noise epsilon.
- Predict noise with the correct shape word and the swapped word
  (square vs triangle; مربع vs مثلث).
- Report mean MSE advantage of the *correct* prompt, fraction of
  comparisons in which it wins, and model response RMS.
- Cover 4 colors, 2 languages, 2 target shapes, 2 seeds and 4
  timesteps = **128 evaluations**.
- Check actual training-target geometry via foreground fill ratio;
  triangles should occupy much less of their bounding box than squares.

This is diagnostic ONLY. It loads the existing trained checkpoints and
does not train, save weights or call any procedural renderer during
inference. A high correct-prompt win rate does not guarantee good
generated images, but a near-zero/negative advantage would reveal
a conditioning weakness on an ordinary diffusion input.

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
SAFE="storage/artifacts/sora/nano/mvp/pixel-20261103T131058Z-65110681/checkpoint.pt"

python3 -m scripts.pixel_conditioning_audit \
  --checkpoint "$BASE" \
  --output /tmp/sora-conditioning-base-42-123.json
python3 -m scripts.pixel_conditioning_audit \
  --checkpoint "$SAFE" \
  --output /tmp/sora-conditioning-safe-42-123.json
```

**Do not launch another long training run** until the per-shape
correct-vs-swapped conditioning result explains which timestep,
language or shape fails. The 32×32 synthetic model is not yet a
general bilingual image generator.

## 78 passing tests; ordinary diffusion conditioning audit (2026-11-03)

User Mac: **78/78 tests passed**; then ran the same held-out 128-case
ordinary-q(x_t|x0) audit on the 800-step baseline and teacher-preserved
20-step model.

| Comparison | Spatial baseline | Teacher-preserved |
|---|---:|---:|
| Correct square word wins (Arabic) | 87.50% | 84.38% |
| Correct square word wins (English) | 84.38% | 81.25% |
| Correct triangle word wins (Arabic) | 68.75% | 71.88% |
| Correct triangle word wins (English) | 71.88% | 68.75% |

The changes are small and not consistently better in either language.
On the baseline, *across shapes combined*, correct prompt win rates
by diffusion timestep were: t=5 **90.62%**, t=20 **90.62%**,
t=50 **68.75%**, t=80 **62.50%**. Accuracy at high noise is weaker,
but that alone does not explain triangle-only failures. The mean MSE
advantage of the correct prompt is tiny (~0.00013–0.00063 across
timesteps), so text conditioning is weak compared to RGB denoising.

The training shapes were confirmed to be distinct: foreground
bounding-box fill ratio was **1.000** for square versus **0.522**
for triangle. Over the audited 120-step triangle-heavy curriculum,
triangle EN/AR counts were equal at 360 each; old shapes were skewed
(circle AR=120, EN=240; square AR=240, EN=120).
This is not evidence that triangle targets are missing.

Existing binary outcome remains unchanged: no generated triangles in
any user-supplied 24-image gallery; 20-step candidate **not promoted**.

### New *read-only* reverse trajectory experiment (code awaits Mac tests)

The `scripts.pixel_reverse_trace` diagnostic traces the exact
`posterior_mean` reverse diffusion from one *shared* initial Gaussian
noise image for a square and triangle of the same color and language.
At selected steps, it reports:
- Pixel-level response of learned noise prediction to changing **only
  the shape word** with the noisy RGB tensor held fixed
- How the independent square and triangle reverse trajectories diverge
- Predicted RGB `x0` foreground masks, square-vs-triangle IoU,
  heuristic classes, and fraction of raw `x0` values requiring clipping
- A side-by-side `reverse_trace.png` contact sheet of learned
  intermediate RGB predictions, and independent final output PNGs

Unlike the training dataset preview, these images are **model
predictions**, NOT procedural PIL-drawn shapes. There is no
training and no checkpoint writing.

The trace duplicates the established mean posterior equations on
purpose and includes regression tests to compare its final RGB tensors
against the normal production `generate` function, so a diverging
analysis implementation can be caught. Full tests have not yet run on
the user's Mac; the new module must be validated before trusting it.

Suggested first run:
```bash
python3 -m unittest discover -s tests -v
BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
OUT="/tmp/sora-reverse-trace-$(date +%s)"
python3 -m scripts.pixel_reverse_trace \
  --checkpoint "$BASE" --color red --language en \
  --seed 42 --guidance-scale 5 --cpu-threads 2 \
  --output-dir "$OUT"
open "$OUT/reverse_trace.png"
```

Inspect whether prompt-only noise response disappears at large
timesteps, whether raw `x0` is dominated by clipping, and whether
triangle and square masks converge despite different text. These
measurements will decide whether the next research area is schedule/
sampler dynamics, the word-conditioned noise network, or the
data/geometry representation. **Do not start another 300/10000-step
fine-tuning run on the existing objective.**

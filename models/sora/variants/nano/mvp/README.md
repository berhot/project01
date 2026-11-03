# BERHOT Sora Nano — first trainable image model MVP

**Status:** experimental engineering MVP, not production image generation.

This is BERHOT's **separate image/design foundation family** at
`models/sora/variants/nano/mvp`. It does not use Nabra, JSM, or FANN weights.
Sora is the provisional image/design family name; verify name/trademark clearance before public launch.

## What works

A tiny, **from-scratch caption-conditioned denoising diffusion model (DDPM)**:
image/caption training pairs, noise-prediction training, checkpoints, deterministic
seeded reverse diffusion inference, PNG export and tests. No hosted image API,
third-party image generator or downloaded pretrained weights are involved.

This intentionally starts with **32×32 colored circles and squares**, six
caption classes and Arabic caption aliases. The images are created procedurally
with no external dataset or copyright uncertainty. It is a *software-pipeline
proof*, **not** a general text-to-image model, graphic designer, poster
generator, text renderer, editor, or evidence of photorealistic quality.

Supported English prompts: `red circle`, `green circle`, `blue circle`,
`red square`, `green square`, `blue square`. Arabic aliases include
`دائرة حمراء` and `مربع أزرق`. Unsupported prompts fail explicitly.

## Local Mac setup

From your original BERHOT directory, after the PR is merged:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
source .venv/bin/activate
cd models/sora/variants/nano/mvp
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Training is intentionally short for the first smoke test. Run more steps to
assess whether it learns the six categories; early samples may look like noise.

```bash
python3 -m scripts.train --steps 200 --batch-size 8 --device cpu
```

The command prints a **new run-specific checkpoint path**, e.g.
`storage/artifacts/sora/nano/mvp/image-.../checkpoint.pt` relative to this
MVP directory. Use the actual printed path in inference:

```bash
python3 -m scripts.infer \
  --checkpoint storage/artifacts/sora/nano/mvp/<YOUR_RUN_ID>/checkpoint.pt \
  --prompt "red circle" --seed 42 --output /tmp/image-nano-red-circle.png
open /tmp/image-nano-red-circle.png
```

The small baseline uses a 50-step noise schedule, 32-channel conv denoiser
and a six-label learned embedding. It has no open-vocabulary text encoder:
Arabic aliases map to the same six supervised labels. CPU is supported;
Apple MPS may require additional testing. Training runs in Python, not
in a web browser. Files under the local `storage/` tree must not be
committed to GitHub.

## Next quality gates

1. Validate clean, reproducible shape generation on held-out random seeds
   instead of relying on training loss only.
2. Introduce licensed/owned real image-caption pairs with provenance,
   duplicate checks, standardized formats and train/validation separation.
3. Add a real multilingual text encoder and trainable conditioning, then
   higher-resolution latent diffusion for general visual concepts.
4. Treat legible typography, controlled compositions, editing/inpainting,
   style consistency and design layers as **separate benchmarks/capabilities**.

Do not describe this MVP as production-ready or as understanding arbitrary
prompts. Nabra is also still experimental: its unseen-speech outputs remain
unintelligible, despite software and single-recording vocoder successes.

## Visual validation and schedule fix

The initial 200/1000-step linear-schedule experiment produced colored noise,
not recognizable circles/squares. It **failed visual validation** despite
falling training loss. The prior 50-step linear schedule did not sufficiently
corrupt images at the final timestep, although inference began at pure noise.

This version uses a cosine schedule reaching approximately pure noise and the
DDPM posterior variance during reverse sampling. **Old checkpoints are not
compatible**; the inference CLI rejects them. Preserve the old files for
comparison and train an entirely new run using the updated branch.

The correction is a hypothesis, not evidence of image quality improvement until
new training and six-prompt visual inspection pass. This PR remains a draft.

## New experiment: conditional VAE (shape-CVAE v1)

**Why:** The original DDPM and its corrected cosine schedule both generated
multicolored noise after 1,000 steps. Falling diffusion loss was not evidence
of visual quality. Keep those DDPM checkpoints intact. This *separate*
conditional variational autoencoder (CVAE) trains on the same six generated
32×32 shape classes with a conditional encoder/decoder and KL regularization.
It generates pixels from **learned weights and sampled latent noise**; the
procedural shape drawing code is used **only in training**, never for inference.
The experiment is not diffusion and must not be described as a working DDPM.

After fetching the updated PR in your **existing BERHOT folder**, use:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT/models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
python3 -m scripts.cvae train --steps 1000 --batch-size 16 --device cpu
```

Use the actual **new CVAE** checkpoint printed by training (not an old DDPM
checkpoint) to create a sample and a six-image preview:

```bash
python3 -m scripts.cvae sample \\
  --checkpoint <NEW_CVAE_CHECKPOINT> --prompt 'دائرة حمراء' \\
  --output /tmp/sora-cvae-red-circle.png
python3 -m scripts.cvae evaluate \\
  --checkpoint <NEW_CVAE_CHECKPOINT> \\
  --output-dir /tmp/sora-cvae-preview-01 --seed 42
open /tmp/sora-cvae-preview-01/preview_grid.png
```

The evaluator saves six separate true model outputs plus a contact sheet and
JSON manifest, but **does not automatically certify image quality**.
Recognition of shapes and colors must be checked visually, and different
seeds should also be tested. Successful output is limited to these six classes.
This experiment is a comparison baseline while future diffusion improvements
remain under development; it is not a general image-design product.

## Experimental spatial-field model — sharp controlled shapes

The earlier DDPM **failed** visual evaluation, and the CVAE completed
training but generated soft / poorly defined shapes in the Mac user's test.
The new `spatial-field-v1` is a **separate neural renderer**, not a general
image-generation model. It receives one of the same six discrete prompt
labels **and explicit seeded geometry** (center X, center Y, size). A learned
per-pixel neural field predicts a shape occupancy mask and RGB colors.
Training uses synthetic images as supervised targets; **inference does not
call the synthetic image renderer or copy target images**. The layout and
foreground/background compositing are explicit architectural priors.

### Test and train in the existing BERHOT folder

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
python3 -m scripts.spatial_field train \
  --steps 1100 --batch-size 24 --device cpu --cpu-threads 2
```

The trainer prints a **new** `storage/artifacts/sora/nano/mvp/spatial-.../checkpoint.pt`.
Copy that exact path and create the six-image contact sheet:

```bash
CKPT="storage/artifacts/sora/nano/mvp/spatial-REPLACE_WITH_ACTUAL_RUN/checkpoint.pt"
python3 -m scripts.spatial_field evaluate \
  --checkpoint "$CKPT" --seed 42 \
  --output-dir /tmp/sora-spatial-seed42
open /tmp/sora-spatial-seed42/preview_grid.png
python3 -m scripts.spatial_field sample \
  --checkpoint "$CKPT" --prompt "دائرة حمراء" --seed 123 \
  --output /tmp/sora-spatial-arabic.png
open /tmp/sora-spatial-arabic.png
```

The evaluation writes `report.json` with per-class foreground-mask
intersection-over-union (IoU) and pixel mean absolute error (MAE), using
synthetic images for **evaluation only**. Choose a new `--output-dir` when
re-running; the exporter refuses existing paths to avoid overwrites.

In an independent CPU experiment using a near-identical spatial field
implementation, **1100 steps and 24 held-out layouts across seeds 42, 123,
997, 2025** yielded mean mask IoU **0.9885**, minimum **0.9380**, and mean
RGB MAE **0.0069**. The experiment checkpoint was not committed to GitHub.
These are results for synthetic shapes with known geometry, **not** evidence
of open-ended design abilities. Evaluate the repository implementation
separately; results may differ across devices / random initialization.

The model is deliberately constrained. **It cannot compose arbitrary
posters, logos, lettering, scenes, or open-vocabulary prompts**, and its
geometry is supplied externally. Treat it as a quality-control milestone
for the next, more general image-generation architecture, not a release.

## Learned layout DDPM + learned neural renderer

The previous `spatial-field-v1` requires external (x, y, size). The next
experiment `layout-ddpm-v1` **learns a probability distribution over those
three geometry parameters** from synthetic training examples and generates
them from seeded diffusion noise at inference. The separate, already-trained
`spatial-field-v1` then predicts the 32×32 image pixels from its weights.

The inference path **does not call** `layout_for_seed()`,
`synthetic_image()`, or `training_layouts()`. Different seeds can generate
different learned positions and sizes for the same six text labels. The
geometry bounds are still hard constraints, and the model is explicitly
**not diffusion over images**, nor an open-ended text-to-image foundation model.

### Reproduce in the original BERHOT folder

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

# First train the spatial renderer (skip if you already have a valid checkpoint).
python3 -m scripts.spatial_field train --steps 1100 --batch-size 24 --device cpu

# Independently train the learned layout prior.
python3 -m scripts.layout_diffusion train \
  --steps 2000 --batch-size 256 --device cpu --cpu-threads 2

# Select existing checkpoints from these two different architectures.
SPATIAL=$(find storage/artifacts/sora/nano/mvp \
  -path '*/spatial-*/checkpoint.pt' -type f | sort | tail -1)
LAYOUT=$(find storage/artifacts/sora/nano/mvp \
  -path '*/layout-*/checkpoint.pt' -type f | sort | tail -1)
test -f "$SPATIAL" && test -f "$LAYOUT"

python3 -m scripts.layout_diffusion evaluate \
  --layout-checkpoint "$LAYOUT" \
  --spatial-checkpoint "$SPATIAL" \
  --seed 42 --output-dir /tmp/sora-learned-layout-seed42
open /tmp/sora-learned-layout-seed42/preview_grid.png
```

Both checkpoint files must have their correct architecture tags. Older
CVAE/DDPM checkpoints are **not interchangeable**. The evaluator writes six
model-generated images and a JSON report with generated geometry. Do not
confuse a matching six-class toy-model result with real image/design ability.
Future work needs image-level diffusion on licensed image-caption datasets,
multi-object composition, and genuine Arabic/English language conditioning.

## RGB pixel-space diffusion + Arabic/English word conditioning (experimental)

The file `src/pixel_diffusion.py` is the **first actual image-level diffusion**
in this family: training noise is added to every RGB pixel in a 32×32 image,
a trainable multiscale U-Net estimates the noise, and inference begins from
Gaussian image noise and denoises into pixels using DDIM. **Neither a spatial
shape renderer nor a geometry DDPM is called during inference.**

Prompts are encoded using **trainable word embeddings**, with independent Arabic
and English vocabulary entries and learned positional features. Arabic prompt
handling is not hardcoded translation to six integer caption IDs.
For example, supported prompts include `red circle`, `a red square on white
background`, `دائرة حمراء`, `مربع أزرق`, and `مربع أخضر على خلفية بيضاء`.
Out-of-vocabulary prompts deliberately fail with a clear error rather than
pretending the model understands an arbitrary request.

**Current dataset:** only six combinations of red/green/blue × circle/square,
with randomized object placements, sizes, caption phrasing, and either Arabic
or English descriptions. Training images are procedurally generated in Python
and require no external sources. There are **no pretrained image models,
remote image APIs, or inference-time procedural graphics**.

### Run from the original BERHOT folder

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

# Initial full experiment (may be slow on CPU):
python3 -m scripts.pixel_diffusion train \
  --steps 3000 --batch-size 12 --device cpu --cpu-threads 2
```

Copy the actual `pixel-.../checkpoint.pt` path printed by training:

```bash
CKPT=$(find storage/artifacts/sora/nano/mvp \
  -path '*/pixel-*/checkpoint.pt' -type f | sort | tail -1)
test -f "$CKPT"

python3 -m scripts.pixel_diffusion sample \
  --checkpoint "$CKPT" --prompt 'دائرة حمراء' \
  --seed 42 --sampler posterior_mean \
  --output /tmp/sora-pixel-arabic.png

OUT="/tmp/sora-pixel-eval-$(date +%s)"
python3 -m scripts.pixel_diffusion evaluate \
  --checkpoint "$CKPT" --seed 42 --sampler posterior_mean \
  --output-dir "$OUT"
open "$OUT/preview_grid.png"
```

The evaluation exports 12 **real model-sampled PNGs** covering all six concepts
in both languages, along with a JSON report. It does **not** certify visual
quality. Saving into an existing output location is intentionally rejected.

### Known limitations and results

The first independent 850-step CPU training run of an early 16-channel
pixel-U-Net reached noise MSE ~0.07. **DDIM samples were still visibly noisy**.
Using full-step deterministic DDPM posterior means instead produced colored,
rounded objects, but **red remained greenish, and circles and squares were
still poorly distinguished**. These are actual experimental outputs, not
proof of prompt fidelity. This is a real failure of image fidelity despite correct
end-to-end software operation. It is not appropriate to merge or ship this
as a visually successful diffusion model. The new GitHub CI uses only a
short **smoke training run** and tests checkpoint → inference → PNG export.
The smoke run cannot establish image quality. Longer experiments, stronger
denoising objectives, better spatial/semantic conditioning, additional data,
quantitative quality measures, and visual review remain required.

Do not confuse this image-level diffusion with `layout-ddpm-v1`, which
diffuses only three geometry parameters, or with the shape CVAE and neural
renderer. All baselines are preserved separately.

### Sampling methods (important)

`--sampler posterior_mean` (default) uses **all diffusion timesteps** and
iterates the DDPM reverse posterior mean without injecting additional noise.
It is a deterministic reconstruction heuristic and **not an unbiased DDPM
sample**. It produced more coherent 32×32 objects on the pilot checkpoint,
but the text-conditioning quality was inadequate.

`--sampler ddim --sample-steps 50` uses a standard deterministic DDIM update
and can skip steps. It produced multicolored noise in the first pilot. The
pipeline exposes both honestly; no choice of sampler replaces better training
or stronger language conditioning. Old checkpoints from other Sora
architectures are rejected.

## Classifier-free guidance (CFG) experiment — bilingual RGB pixels

The old 3000-step RGB checkpoint generated largely **the same red rounded
object** regardless of color, shape, or language. The trained noise predictor
works, but conditioning quality **failed**. The next experiment adds:

- **Training text dropout** (`--cfg-drop-prob 0.15`): fully masks prompts for
  roughly 15% of examples, so the network learns unconditional predictions.
- **Guidance-scale inference** (`--guidance-scale`): combines conditional and
  unconditional noise predictions at every RGB denoising step. Scale `1.0`
  is the original conditional prediction; `3.0`, `5.0`, `7.5` test stronger
  conditioning, which can also degrade or oversaturate results.
- **Model width selection**: `--base-channels 32` (also accepts the legacy
  `--channels`). Earlier checkpoints are not overwritten. A new training
  run is necessary for guidance > 1; the inference CLI explicitly rejects
  older pixel checkpoints that lack null-conditioning training.
- **Toy metrics**: foreground color, approximate circle/square fill-ratio,
  strict prompt match and English/Arabic pair agreement. Ambiguous/undetected
  shapes count as incorrect. These are **heuristics for only this toy
  dataset**. Good scores do not prove realistic image quality.
- **Guidance comparison**: creates 12 real pixel-diffusion images for each
  scale and seed, together with per-image scores, contact sheets, and a
  JSON summary. No synthetic shape renderer is called in inference.

### Train in your ORIGINAL BERHOT directory, no extra worktree

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

python3 -m scripts.pixel_diffusion train \
  --steps 10000 --batch-size 12 --device cpu --cpu-threads 2 \
  --cfg-drop-prob 0.15 --base-channels 32
```

Training 10,000 steps can be expensive on a CPU. Test with 200–500 steps first
to verify the new training path, then train a **new run**. Record the actual
printed `pixel-.../checkpoint.pt` path. The previous 3000-step checkpoint
is preserved but **cannot provide meaningful classifier-free guidance**.

```bash
CKPT=$(find storage/artifacts/sora/nano/mvp \
  -path '*/pixel-*/checkpoint.pt' -type f | sort | tail -1)
test -f "$CKPT"
OUT="/tmp/sora-cfg-$(date +%s)"
python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler posterior_mean \
  --guidance-scales 1 3 5 7.5 --seeds 42 123 \
  --output-dir "$OUT"
open "$OUT/guidance_3_seed_42/preview_grid.png"
cat "$OUT/summary.json"
```

For a single Arabic/English experiment, use `python3 -m
scripts.pixel_diffusion evaluate --checkpoint "$CKPT" --guidance-scale 3
--sampler posterior_mean --output-dir /tmp/unique-output`. The evaluation's
`report.json` now contains `metrics` and each image's heuristic prediction.

**Success gate**: visually recognizable shapes, correct red/green/blue colors
and circle/square classification for both languages across several seeds;
prompt accuracy and bilingual **both-correct** rate should corroborate visual
inspection. Do not merge as a visually validated model just because noise
loss decreases or a smoke CI passes.

## Dataset v2 — twelve concepts, bilingual (separate from v1)

**Added, not promoted as a visually trained model.** Dataset v2 expands
the procedural **training distribution** from 3 colors × 2 shapes to
**4 colors × 3 shapes = 12 concepts**:

- Colors: red / green / blue / yellow
- Shapes: circle / square / triangle
- Backgrounds: warm-white / light-gray / cream
- Position and size vary per sample; triangles vary orientation
- English and Arabic captions, three phrasings per concept/language
- Correct Arabic adjective gender, e.g. `دائرة صفراء` and `مثلث أصفر`

New files: `src/pixel_dataset_v2.py` (streamed synthetic training samples),
`scripts/dataset_v2.py` (inspection previews), and
`tests/test_pixel_dataset_v2.py`. An optional `dataset_version` field
in the RGB model config chooses v1 or v2 vocabulary and checkpoint embedding
size. **v1 token indices are unchanged**, and previous v1 checkpoints stay
loadable. Old v1 checkpoint weights cannot generate `yellow triangle` without
training a new v2 model. Checkpoint loading rejects a mismatched vocabulary.

### Inspect held-out dataset examples before training

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

OUT="/tmp/sora-v2-data-$(date +%s)"
python3 -m scripts.dataset_v2 preview \
  --split validation --seed 42 --output-dir "$OUT"
open "$OUT/preview_grid.png"
```

**The dataset preview is drawn with PIL. It is NOT model inference.**
Training samples are created on demand and held in memory per batch; no
million-file dataset is downloaded or stored on your Mac. A separate
deterministic PRNG seed namespace is used for the validation split; this
is a controlled synthetic holdout, not proof of real-world generalization.

### First verify v2 training works, then train the new model

```bash
python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 \
  --steps 100 --batch-size 12 \
  --base-channels 32 --cfg-drop-prob 0.15 \
  --device cpu --cpu-threads 2

# Once test results are good, start a fresh, longer v2 experiment:
python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 \
  --steps 10000 --batch-size 12 \
  --base-channels 32 --cfg-drop-prob 0.15 \
  --device cpu --cpu-threads 2
```

Copy the `pixel-.../checkpoint.pt` output from the actual **v2 training
run**. The original v1 experiment remains stored separately.
After training finishes:

```bash
CKPT="PASTE_YOUR_ACTUAL_V2_CHECKPOINT_PATH"
OUT="/tmp/sora-v2-cfg-$(date +%s)"
python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler posterior_mean \
  --guidance-scales 1 3 5 7.5 --seeds 42 123 \
  --output-dir "$OUT"
open "$OUT/guidance_5_seed_42/preview_grid.png"
cat "$OUT/summary.json"
```

Each grid now has **24 model-generated images** (12 concepts, both languages).
Across 4 guidance scales and 2 seeds this yields **192 RGB PNGs**. Heuristic
scoring uses dataset-specific color/shape rules, counting unknowns as
failures. Circle-vs-triangle scores and background robustness need human
inspection; higher guidance can produce distorted shapes. Do not compare
v1 6-class prompt accuracy directly to v2 12-class accuracy as if they
were the same benchmark.

### Acceptance gates

1. Unit tests pass, 24 v2 dataset examples inspect correctly, and
   v2 training/inference writes distinct versioned checkpoints.
2. Multiple held-out seeds show recognizably correct **all 12** colors/shapes
   in both languages; review the full galleries, not just a training loss.
3. Report per-class errors, strict prompt accuracy, Arabic/English
   both-correct rate, and failures such as missing/merged shapes.
4. Keep PR #10 in **draft** until actual v2 checkpoints receive visual review.
   GitHub Actions remains **manual-only** to prevent email floods.

**Known limitation:** This is still synthetic **single-object 32×32**
pixel diffusion, not a realistic image generator, vector art engine, or
general-purpose Arabic/English text-to-image foundation model. Next steps
only after quality evaluation: more diverse licensed image-caption pairs,
proper cross-attention/encoder design, higher resolution, and multi-object
composition. Also review the provisional Sora family name before launch.

## Dataset v2 10k-step result: accuracy regression, diagnostic required

The user ran 10,000 v2 steps with CFG dropout 0.15 and base width 32.
The training noise MSE fell to ~0.01, but the **strict 12-class toy
prompt accuracy was only 8.33% at guidance 1/3, 16.67% at 5,
and 25% at 7.5**, for each of seeds 42 and 123. This does
**not** pass the v2 visual/semantic gate. See
`SORA_PIXEL_V2_RESULTS.md` for checkpoint metadata and failure report.

Run **the saved-output diagnostic without retraining**:

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

The diagnostic loads existing v2 evaluation PNGs and reports and
prints dominant-class counts, per-class confusion, EN/AR false agreement
and same-seed image differences. It cannot establish human visual quality.
Keep v1 and v2 checkpoints; wait to design the next training objective
until the per-class failures and the galleries are inspected.

## Targeted v2 recovery experiment: warm-start from successful v1 RGB weights

**Why this experiment exists:** After 10,000 steps, dataset v2 generated
largely red squares instead of triangles and correct colors; the diagnostic
found one dominant class in 54.17% of images at CFG 7.5 and in 100%
at CFG 1 and 3. The screenshot confirms these are not merely heuristic
classification errors. v1 had achieved much stronger six-class control,
so we can test whether **transferring learned v1 RGB denoising weights**
helps rather than training v2 blindly from random initialization again.

**This is a hypothesis, not a confirmed improvement.** The source is
the user's actual v1 `rgb-unet-text-v1` checkpoint (32 channels, 100
diffusion steps), not the v2 failed run or an external image model.
All old U-Net layers and old word embeddings are transferred. New
words `yellow`, `triangle`, `أصفر`, `صفراء`, `مثلث`, `light`,
`فاتحة` receive initial vectors by averaging related v1 word vectors
and then are fully trainable. This is a semantic initialization for
new parameters, **not a hardcoded shape/color renderer**.
Architecture/diffusion-step/vocabulary mismatches are rejected.

**Your old checkpoints stay in place.** A new run gets its own timestamped
checkpoint, and metadata records the v1 source checkpoint. Nothing is
merged into another model family. Image generation still starts from
Gaussian RGB noise and uses learned U-Net weights.

### Run only a short smoke test first

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

# Lists checkpoint versions and training steps without changing anything.
python3 -m scripts.pixel_checkpoints list

# Pick the trained 10k-step v1 RGB checkpoint; never use failed v2 as source.
V1=$(python3 -m scripts.pixel_checkpoints latest \
  --dataset-version v1 --min-steps 10000)
echo "Source v1: $V1"
test -f "$V1"

python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 --init-from-v1 "$V1" \
  --steps 100 --batch-size 12 \
  --base-channels 32 --diffusion-steps 100 \
  --cfg-drop-prob 0.15 --learning-rate 0.0001 \
  --device cpu --cpu-threads 2
```

**If the tests and short smoke run pass**, the independent research trial
(not a production release) is **2,500 steps**, with identical hyperparameters
except `--steps 2500`. Compare at least two seeds and guidance scales 5 and
7.5. Record whether the new model improves the failed v2's 25% prompt
accuracy **on the same 12-class test**, and compare actual 24-image grids.
If there is no improvement, stop rather than automatically increasing
to 10,000+ steps. The old v1-v2 comparisons are *not* the same task;
the new v2-to-v2 comparison is.

If warm-start does not fix missing triangles, investigate a stronger spatial
shape-conditioning design, color/shape auxiliary objectives and time-dependent
prompt sensitivity before spending more training compute. The RGB diffusion
architecture and existing v1/v2 baselines are preserved independently.

## Shape curriculum experiment: fine-tune the improved 2,500-step v2 checkpoint

User's **actual model preview** after v1→v2 warm-start shows circle and
square requests now largely recognizable, but **all triangle requests
remain square-like**. At CFG 7.5, the warm-start v2 had **95.83% color
accuracy but only 58.33% shape accuracy**. This is a genuine geometry
failure, not simply a thresholding artifact. Do not add more classes yet.

New opt-in features:
- `--init-from-v2 PATH` loads an existing v2 RGB diffusion checkpoint,
  checks architecture and vocabulary, and starts a **new** checkpoint
  run. The optimizer is **reset**; this is fine-tuning, NOT an exact
  optimizer-state training resume. Both v1 and v2 inputs cannot be
  supplied simultaneously. Source checkpoint bytes are never modified.
- `--triangle-fraction 0.50` makes 50% of the generated training
  examples triangles, with the remaining half split between circles
  and squares. All four colors are retained and randomized placements
  remain. This is a **training-only** modification; inference remains
  the learned RGB pixel denoiser. Default `--triangle-fraction` is
  unset and the earlier balanced training sequence is unchanged.
- All flags are optional, preserving existing checkpoint semantics.

### Short integration check, no long training

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

SOURCE="storage/artifacts/sora/nano/mvp/pixel-20261103T004620Z-c92b5c0b/checkpoint.pt"
test -f "$SOURCE" || { echo "Source v2 checkpoint missing"; false; }

python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 \
  --init-from-v2 "$SOURCE" \
  --triangle-fraction 0.50 \
  --steps 100 --batch-size 12 \
  --base-channels 32 --diffusion-steps 100 \
  --cfg-drop-prob 0.15 --learning-rate 0.00005 \
  --device cpu --cpu-threads 2
```

The 100-step smoke run is only a software check. If it succeeds,
run one independent **800-step** trial from `SOURCE` (not from the
100-step smoke checkpoint) with the same flags and
`--steps 800`. The lower learning rate is deliberate, to reduce
forgetting of circles/squares/colors. Record the NEW checkpoint path
printed by training, not a placeholder.

Evaluate at guidance scales 5 and 7.5, seeds 42 and 123, then inspect
`preview_grid.png` and run
`python3 -m scripts.pixel_diagnose --evaluation-dir "$OUT"
--output /tmp/sora-triangle-audit-NEW.json`. The diagnostic
includes per-shape and per-concept strict accuracies.

**Controlled comparison:** benchmark against the existing 2,500-step
warm-start v2, **not** against v1. The targeted intervention is the
triangle-heavy curriculum, but optimizer reset, training steps and LR
also change; improvement does not isolate any one factor. Observe
whether triangle recall improves **without degrading** color and
circle/square accuracy. If triangles remain squares, don't
automatically increase the number of steps: inspect the learned
text-conditioned denoising representation and consider a shape-focused
auxiliary loss or stronger spatial conditioning in a separate model
revision. Do not interpret a falling noise MSE as proof of shape
understanding.

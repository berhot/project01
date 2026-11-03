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

## Experimental shape conditioning: spatial token attention (separate architecture)

### Why this is needed

The user's 800-step triangle-heavy v2 fine-tuning **failed** the actual
geometry check: at guidance 5.0, circles 100%, squares 100%, and **triangles
0%**, on both seeds 42 and 123. Lower noise MSE, more triangle exposure and
larger CFG did not fix this. The image gallery visibly confirmed rectangular
outputs for nearly all triangle prompts.

The original `rgb-unet-text-v1` conditions its entire RGB denoiser on
a single **mean-pooled word vector**. The new opt-in
`rgb-unet-spatial-text-v2` can attend **per pixel** to individual
English/Arabic text tokens at both 32×32 and 16×16 feature scales,
allowing shape terms to influence local geometry instead of only a
global feature vector. This is still entirely learned neural denoising
with input RGB noise at inference, **no shape drawing at inference**.

The experimental objective also uses a **training-only silhouette Dice
loss** from the real RGB training pixels, only for less-noisy timesteps.
It encourages foreground layout and boundaries while keeping ordinary
noise-prediction MSE. Both loss parts are logged separately.
No triangle target template enters the inference pipeline. This loss
is a research approximation and can itself have failure modes.

**Backwards compatibility:**
- `--conditioning pooled` is default; all old v1/v2 checkpoints remain
  the same `rgb-unet-text-v1` architecture.
- `--conditioning spatial` is opt-in for v2 only; it produces
  `rgb-unet-spatial-text-v2` checkpoints.
- `--init-from-v2` accepts the user's trained v2 pooled checkpoint and
  transfers matching RGB weights. The new attention output projections
  start at **zero**, so the new model initially predicts the SAME RGB
  noise as the source until the attention learns.
- Source checkpoints are never overwritten. The new run resets AdamW
  state (fine-tune, not an optimizer-accurate resume).
- Classifier-free guidance and the bilingual 12-class evaluation apply
  to both architectures.
- The Sora visual GitHub Actions workflow remains **manual-only**.

### Tests and a bounded 100-step verification

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp

python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T004620Z-c92b5c0b/checkpoint.pt"
test -f "$BASE" || { echo "Missing original 2500-step v2 baseline"; false; }

python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 --conditioning spatial \
  --init-from-v2 "$BASE" \
  --triangle-fraction 0.50 --silhouette-weight 0.15 \
  --steps 100 --batch-size 12 --base-channels 32 \
  --diffusion-steps 100 --cfg-drop-prob 0.15 \
  --learning-rate 0.00005 --device cpu --cpu-threads 2
```

The 100-step run is for **integration tests only**, not a promise that
triangles are fixed. It writes a NEW checkpoint with the
`rgb-unet-spatial-text-v2` architecture, and logs
`noise_mse`, `silhouette_dice_loss`, and `total_loss`.
If tests/short training fail, stop and fix that issue before launching
longer training.

If 100 steps succeed, a controlled comparison is one **800-step**
independent spatial run initialized from `BASE`, with the same LR,
triangle mix and seed as the old 800-step pooled trial, and CFG scale
5 plus seed 42/123. The only intended experimental changes are the
spatial architecture and silhouette objective (both active together);
a separate ablation with `--silhouette-weight 0` would be necessary
to isolate individual contributions.

The previous model's **triangle accuracy is 0%**. A genuine success
requires visible triangles and improved per-shape/strict prompt
metrics without losing circle, square or color accuracy. Do not claim
success from low training loss alone. If triangles remain rectangles,
stop the experiment and reconsider representation or data design,
rather than repeatedly extending CPU training.

## Diagnose triangle failure without another training run: ancestral DDPM

The 800-step spatial-text v2 plus silhouette training experiment still
produced **zero recognizable triangles** on the user's Mac. At guidance
5.0, all circles and squares were correct but triangle accuracy
remained 0% on both seeds. The actual RGB contact sheet confirms this;
this is not a false alarm from the toy image heuristic.

Previous evaluation used deterministic `posterior_mean` reverse steps,
which omit the DDPM posterior Gaussian variance. This may bias samples
toward average shapes. We now offer an experimental `ancestral`
sampler that injects **seeded Gaussian posterior variance** at every
reverse step except the final one (with normal clipped x0 prediction).
This is closer to standard stochastic DDPM sampling. It does NOT
change model weights. It can still fail entirely if the learned
distribution has no usable triangle semantics.

Also add `--shared-noise` to the `evaluate` and
`scripts.pixel_metrics` CLIs: by default each color+shape starts with
a different seeded Gaussian image, but this option reuses the same
initial noise and random-number stream across all prompts, allowing
a controlled *prompt-sensitivity* comparison. This changes evaluation
seed policy; avoid directly comparing old and new aggregate metrics
without noting that difference.

### Test the existing trained model; NO retraining

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

CKPT="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$CKPT" || { echo "Checkpoint not found"; false; }

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

The model still generates **24 actual RGB output PNGs** at 100
diffusion steps; no synthetic rendering is used at inference.
Results should be inspected visually for triangles, along with
`shape_accuracy.triangle`. If no triangle improvement occurs,
the next meaningful experiment is training-time **shape
discrimination** with a controlled paired-caption objective,
not a blind increase in steps or sampling guidance.

### Recommended controlled sampler comparison (same model, same initial noise)

Rather than comparing a new shared-noise score directly with the old
per-class-seeded score, run **both** samplers under the same new
evaluation protocol:

```bash
CKPT="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
MEAN="/tmp/sora-mean-shared-$(date +%s)"
ANCESTRAL="/tmp/sora-ancestral-shared-$(date +%s)"

python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler posterior_mean \
  --guidance-scales 5 --seeds 42 --shared-noise --output-dir "$MEAN"

python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler ancestral \
  --guidance-scales 5 --seeds 42 --shared-noise --output-dir "$ANCESTRAL"

open "$MEAN/guidance_5_seed_42/preview_grid.png"
open "$ANCESTRAL/guidance_5_seed_42/preview_grid.png"

python3 -m scripts.pixel_diagnose \
  --evaluation-dir "$ANCESTRAL" \
  --output "$ANCESTRAL/shape_audit.json"
```

The diagnostic additionally reports per-color/English/Arabic
**square-vs-triangle foreground mask intersection-over-union** (IoU)
when the same initial noise was used: IoU near 1.0 means the
silhouettes are nearly identical, regardless of changed shape words.
A lower IoU indicates different silhouettes but not necessarily a
correct triangle; the preview and label metrics remain essential.
This is a model-sampling experiment, **not training or a guarantee**.

## Triangle geometry failure: same-noise contrastive diagnostic

**Confirmed:** Original v2 pooled model, triangle-heavy fine-tune,
spatial text attention, silhouette loss, and stochastic ancestral
sampling **all failed** to generate any recognizable triangles.
The latest same-initial-noise trial with the 800-step spatial
checkpoint produced 66.67% strict prompt accuracy using
`posterior_mean` but only 41.67% using `ancestral`.
Both had **zero triangle accuracy**. See
`SORA_PIXEL_V2_RESULTS.md`.

A new opt-in research experiment creates paired square/triangle
training targets using the SAME color, location, background, size
and language. Both paired prompts are denoised from the SAME
moderately noisy RGB image, and a counterfactual supervision
term forces a shape-dependent predicted noise correction.
An additional difference-of-silhouettes loss upweights the
triangle cutout corners. This is **only training supervision**.
The image-generation path still uses learned RGB diffusion,
not procedural drawing.

New files:
- `src/pixel_shape_contrast.py`: deterministic paired-data,
  shared-noise targets, differentiable shape loss and read-only probe
- `scripts/pixel_shape_probe.py`: inspect an existing checkpoint's
  response when square/triangle prompts receive identical noisy pixels
- `tests/test_pixel_shape_contrast.py`: mathematical target,
  determinism, EN/AR coverage, gradients, checkpoint preservation,
  and short CLI fine-tuning tests

### First: inspect the current weights, without training

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp

python3 -m unittest discover -s tests -v

CKPT="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$CKPT" || { echo "Spatial checkpoint missing"; false; }

OUT="/tmp/sora-shape-probe-$(date +%s).json"
python3 -m scripts.pixel_shape_probe \
  --checkpoint "$CKPT" \
  --pairs 8 --seed 42 --output "$OUT"
cat "$OUT"
```

A small `prompt_difference_ratio` suggests weak sensitivity to
shape terms under the paired/noisy probe. `shape_difference_alignment`
measures whether the *direction* of the response matches the target
correction. High sensitivity or alignment alone **does not prove**
triangle generation; the 24-image grid remains the real acceptance test.

### Next: bounded 50-step integration experiment (only after tests pass)

```bash
BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"

python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 --conditioning spatial \
  --init-from-v2 "$BASE" \
  --triangle-fraction 0.50 \
  --paired-shape-weight 0.35 --paired-shape-pairs 4 \
  --paired-silhouette-weight 0.50 \
  --silhouette-weight 0.15 \
  --steps 50 --batch-size 12 --base-channels 32 \
  --diffusion-steps 100 --cfg-drop-prob 0.15 \
  --learning-rate 0.00005 --device cpu --cpu-threads 2
```

The standard denoising objective is retained. Logs add
`paired_shape_noise_mse`, `paired_shape_difference_loss`,
`paired_shape_loss`, and total loss. The new checkpoint has
a unique path, records its parent and the training weights,
and does not overwrite or modify the source.

This is **not proven to fix triangles** and shouldn't automatically
be extended to thousands of CPU training steps. First compare the
probe and inspect the same eight successful circle/square classes
for regressions. If the short test works, consider a bounded
300-step trial with explicit evaluation and a maximum runtime budget.
Keep PR #10 in draft. The Sora visual workflow remains
`workflow_dispatch` manual-only to avoid notification flooding.

### Paired-loss 50-step result: validate before any more training

The user passed **62 tests** and completed the opt-in **50-step**
paired square/triangle run from the spatial-v2 800-step checkpoint.
The new local checkpoint is
`storage/artifacts/sora/nano/mvp/pixel-20261103T124924Z-b7835e13/checkpoint.pt`.
Paired shape-difference loss was 0.369545 at step 1 and 0.148050
at step 50, **on different training batches**. We have not yet
measured held-out prompt sensitivity or generated-image quality.

The initial probe reported predicted/expected shape-noise response
0.09417 and cosine alignment −0.0587, but its eight pairs assigned
only one language to each color. The updated probe fixes this by
covering every color in both languages; it labels its new protocol
`matched-v2-multistep-language-balanced`. Compare both checkpoints
using that protocol, not to the unbalanced initial score.

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
NEW="storage/artifacts/sora/nano/mvp/pixel-20261103T124924Z-b7835e13/checkpoint.pt"
test -f "$BASE" && test -f "$NEW" || { echo "Missing checkpoint"; false; }

python3 -m scripts.pixel_shape_probe \
  --checkpoint "$BASE" --pairs 8 --seed 42 \
  --output "/tmp/sora-shape-base-balanced-42.json"
python3 -m scripts.pixel_shape_probe \
  --checkpoint "$NEW" --pairs 8 --seed 42 \
  --output "/tmp/sora-shape-finetuned-balanced-42.json"

OUT="/tmp/sora-paired-50-eval-$(date +%s)"
python3 -m scripts.pixel_metrics \
  --checkpoint "$NEW" --sampler posterior_mean \
  --guidance-scales 5 --seeds 42 \
  --output-dir "$OUT"
open "$OUT/guidance_5_seed_42/preview_grid.png"
python3 -m scripts.pixel_diagnose \
  --evaluation-dir "$OUT" --output "$OUT/shape_audit.json"
```

This uses **no retraining**, no checkpoint modification, and the
same RGB diffusion sampler protocol as the earlier 800-step spatial
benchmark. It will answer whether a short paired run improved the
predicted shape response and, more importantly, whether recognizable
triangles appear in the 24 generated images. Do not extend to 300+
steps without comparing the new images and per-shape breakdown.

## Failed paired shape loss: protect old concepts before any further training

A 50-step `--paired-shape-weight 0.35` run did **not** learn triangles.
Its matched-noise probe strengthened (RMS ratio 0.0893→0.5922,
alignment -0.0096→0.6340), but the **real generated images regressed**:
circles 100%→0%, color 100%→62.5%, strict prompt 66.67%→29.17%.
Triangles remained **0%**. The user's contact sheet confirms this.
Do not use the paired 50-step checkpoint as the best model. Do not
train it longer or replace earlier checkpoints.

### Experimental safeguard: frozen teacher and low-weight pair objective

The optional `--preserve-weight` invokes a **frozen copy** of the
original trained spatial-v2 model as a teacher during fine-tuning.
Both student and teacher predict noise from the **same ordinary
forward-noised RGB images** of circles and squares. The teacher
loss protects both conditional and *null classifier-free guidance*
predictions. A separate paired square/triangle loss is still allowed,
but at a much lower weight of **0.02**, not 0.35.

The objective does NOT guarantee triangle generation, and preserving
the teacher can limit new learning. The teacher checkpoint and all
other existing files stay unchanged. Both teacher and paired losses
default to zero and cannot affect legacy training unless enabled.

### First: run all tests

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
```

### Only after tests pass: optional **20-step** safety experiment

```bash
BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$BASE" || { echo "Proven spatial baseline missing"; false; }

LOG="/tmp/sora-teacher-20-$(date +%s).log"
python3 -m scripts.pixel_diffusion train \
  --dataset-version v2 --conditioning spatial \
  --init-from-v2 "$BASE" \
  --triangle-fraction 0.50 \
  --paired-shape-weight 0.02 --paired-shape-pairs 4 \
  --paired-silhouette-weight 0.50 \
  --preserve-weight 1.0 --preserve-null-weight 0.50 \
  --silhouette-weight 0.15 \
  --steps 20 --log-every 10 --batch-size 12 \
  --base-channels 32 --diffusion-steps 100 \
  --cfg-drop-prob 0.15 --learning-rate 0.00002 \
  --device cpu --cpu-threads 2 | tee "$LOG"

CKPT=$(tail -n 1 "$LOG" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["checkpoint"])')
test -f "$CKPT" || { echo "New checkpoint missing"; false; }
```

### Compare actual generated images with identical prompts/seeds

```bash
REVIEW="/tmp/sora-teacher-gate-$(date +%s)"
python3 -m scripts.pixel_metrics \
  --checkpoint "$BASE" --sampler posterior_mean \
  --guidance-scales 5 --seeds 42 --output-dir "$REVIEW/base"

python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler posterior_mean \
  --guidance-scales 5 --seeds 42 --output-dir "$REVIEW/candidate"

python3 -m scripts.pixel_quality_gate \
  --baseline-report "$REVIEW/base/guidance_5_seed_42/report.json" \
  --candidate-report "$REVIEW/candidate/guidance_5_seed_42/report.json"

open "$REVIEW/candidate/guidance_5_seed_42/preview_grid.png"
```

The quality gate exits with status 2 when a candidate fails. It rejects
an image-quality regression even if its shape-response probe improves.
The gate only uses toy segmentation/label heuristics and **never**
certifies human visual quality. Inspect all 24 images and their
Arabic/English labels before accepting. If triangles remain 0%
or old classes regress, **stop** and investigate the training
distribution/objective instead of increasing the run length.

This model remains an experimental 32×32, 12-class synthetic RGB
diffusion MVP, not open-domain Arabic/English image generation.

### 20-step teacher-preserved result and quality-gate legacy fix

The user finished the 20-step test and supplied the actual CFG-5
generated image sheet: **all circles and squares remained correct,
but all triangle prompts are still blocks**. The new candidate
`pixel-20261103T131058Z-65110681/checkpoint.pt` matched the good
800-step spatial-v2 baseline on the 12-class toy benchmark: colors
100%, circles 100%, squares 100%, triangles **0%**, strict prompt
accuracy 66.67%. This is **not** a triangle-generating model.

The initial quality gate raised
`ValueError: Cannot compare different shared_noise_across_concepts`
because an earlier baseline report omitted the field introduced
later. Legacy reports used independent seed-per-prompt noise, so the
gate now interprets an omitted field as `false`; it still rejects
explicit shared-noise `true` versus `false`, wrong seeds, or
different guidance/sampler settings. Existing saved reports can be
compared directly without regenerating images:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -p 'test_pixel_quality_gate.py' -v

python3 -m scripts.pixel_quality_gate \
  --baseline-report /tmp/sora-spatial-eval-1793689086/guidance_5_seed_42/report.json \
  --candidate-report /tmp/sora-safe20-eval-1793711582/guidance_5_seed_42/report.json
```

**Expected:** gate decision JSON with
`"accepted_by_toy_gate": false` because triangle accuracy
remains zero, and process exit code 2 by design. No more
training is indicated until the triangle data/representation
failure is better understood.

### Read-only ordinary DDPM triangle-conditioning audit — next research gate

The user confirmed **74/74 tests passed**, and the quality gate
correctly rejected the teacher-preserved 20-step candidate: circle,
square, and color were still 100%, but triangle accuracy remained **0%**.
No additional training is justified from those results.

The latest investigation adds
`scripts.pixel_conditioning_audit` to test the denoiser's response
on NORMAL forward diffusion q(x_t|x0), not the specially constructed
midpoint x_t used by the paired auxiliary loss. For actual held-out
square/triangle images, it reuses the same noisy RGB input and epsilon
target, and compares MSE for the **correct** shape word against the
**wrong** shape word in both languages. Timesteps 5, 20, 50, 80 and
seeds 42/123 give 128 comparisons; the script records correct-prompt
win rate, MSE advantage, and prompt-response RMS by shape, language
and timestep.

It also audits the actual target masks and the original optional
50%-triangle curriculum's 120-step language exposure. Triangles are
proper 3-vertex training-image polygons and have equal aggregate
EN/AR representation. The original curriculum nevertheless assigns
circles EN:AR = 2:1 and squares EN:AR = 1:2. The diagnostic does not
change that old training distribution or rewrite model weights.

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
SAFE="storage/artifacts/sora/nano/mvp/pixel-20261103T131058Z-65110681/checkpoint.pt"
test -f "$BASE" && test -f "$SAFE" || { echo "Missing checkpoint"; false; }

python3 -m scripts.pixel_conditioning_audit \
  --checkpoint "$BASE" --seeds 42 123 --timesteps 5 20 50 80 \
  --output /tmp/sora-shape-conditioning-base-v2.json

python3 -m scripts.pixel_conditioning_audit \
  --checkpoint "$SAFE" --seeds 42 123 --timesteps 5 20 50 80 \
  --output /tmp/sora-shape-conditioning-safe20-v2.json
```

This must remain **read-only**. A positive MSE advantage for correct
shape text suggests the model uses that information on a real
forward-noised validation image. It is NOT proof of triangular
generations; a negative advantage or low win rate is useful evidence
of poor conditioning. Compare both saved reports on identical samples.
If the ordinary q(x_t) probe is adequate but sampled triangles
still fail, investigate error accumulation through the reverse
trajectory or the representation/size distribution rather than
blindly adding more training steps.

New tests are committed but have **not been run on the Mac**.
The earlier 800-step spatial checkpoint remains the best toy baseline.

### Reverse denoising trace: where does triangle geometry disappear?

The latest ordinary-forward-q(x_t) audit shows **some correct
square/triangle prompt separation** but worse correctness at high
noise (baseline correct-prompt win rate 90.62% at t=5/20 vs
68.75% at t=50 and 62.5% at t=80, across both shapes combined).
The safe 20-step run does not meaningfully improve this conditioning.
Ground-truth triangle images occupy only 52.2% of their bounding
boxes versus 100% for squares, so training labels do contain the
necessary geometry.

To observe actual generation dynamics, `scripts.pixel_reverse_trace`
runs **real learned posterior-mean RGB reverse diffusion** twice
using identical Gaussian starting pixels and a square or triangle
prompt of the same color/language. At selected timesteps it logs
the prompt-only denoiser epsilon difference on the same input,
RGB trajectory differences, foreground-mask IoU, shape heuristics
and x0 clipping, while exporting the actual learned intermediate
x0 RGB images in `reverse_trace.png`.

This tool does not draw shape templates, write weights, or train.
Its final outputs are checked against the established
`src.pixel_diffusion.generate` sampler in new regression tests.
The code and tests have been committed; **the new Mac test results
are not yet available**.

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp

python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$BASE" || { echo "Baseline checkpoint not found"; false; }
OUT="/tmp/sora-reverse-trace-$(date +%s)"

python3 -m scripts.pixel_reverse_trace \
  --checkpoint "$BASE" --color red --language en \
  --seed 42 --guidance-scale 5 --cpu-threads 2 \
  --output-dir "$OUT"

open "$OUT/reverse_trace.png"
```

The exported `report.json` distinguishes same-input word effects
from cumulative differences between independent reverse paths.
Intermediate `x0` estimates can be badly clipped at high noise,
so heuristic shape labels at early timesteps **do not establish
correctness**. Compare late-stage silhouettes and the final images
to determine if the triangle semantics disappear in reverse sampling.
Run **no further training** pending this analysis.

## High-noise clipping ablation — no retraining

The user ran 82 tests successfully and collected matched-noise,
100-step English and Arabic red square-vs-triangle trajectories.
At timestep 99, **99.2–99.4% of raw x0 RGB values needed clipping**.
At timestep 49, square/triangle mask IoU was **93.2–93.7%** and both
final triangle outputs were classified (and visually looked) like
**squares**. This is direct evidence of an early reverse-trajectory
shape collapse; it does *not* prove clipping caused it.

The inference-only option `--x0-clipping dynamic` uses a per-image
99.5th-percentile dynamic threshold on raw clean-image estimates
instead of fixed [-1,1] clipping. It may preserve relative contrast
when raw x0 is enormous at high-noise steps. The policy is applied
inside the learned RGB reverse-diffusion process; it does not draw
any shape or change the checkpoint. It may also degrade images.

**Default `fixed` preserves existing code behavior**. All training,
weight loading and checkpoint formats remain unchanged.

Run tests first, then compare both policies on the same existing
checkpoint, seed 42, CFG 5, and posterior-mean sampling. This is
only an ablation: *do not promote the dynamic sampler based on
toy metrics without inspecting the 24-image previews.*

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

CKPT="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$CKPT" || { echo "Missing checkpoint"; false; }
OUT="/tmp/sora-x0-ablation-$(date +%s)"

python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler posterior_mean \
  --guidance-scales 5 --seeds 42 \
  --x0-clipping fixed --output-dir "$OUT-fixed"

python3 -m scripts.pixel_metrics \
  --checkpoint "$CKPT" --sampler posterior_mean \
  --guidance-scales 5 --seeds 42 \
  --x0-clipping dynamic --output-dir "$OUT-dynamic"

open "$OUT-fixed/guidance_5_seed_42/preview_grid.png"
open "$OUT-dynamic/guidance_5_seed_42/preview_grid.png"
```

The `scripts.pixel_quality_gate` intentionally refuses to compare
different `x0_clipping` policies as a like-for-like checkpoint
promotion test; the default fixed mode is implicit in older reports.
Use the two separately labeled summaries and visual galleries
to decide whether clipping is contributing. If the dynamic
variant still cannot generate triangles, or sacrifices circles,
squares or color accuracy, keep the original fixed sampler.
No more training is warranted by these diagnostics alone.

**New test code is committed but has not yet run on the Mac.**

## After dynamic clipping: inspect actual triangle recovery instead of sampler tuning

On the user's Mac, **87/87 tests passed** and the same spatial-v2
800-step checkpoint produced these CFG-5 seed-42 results:

| | Fixed | Dynamic thresholding |
|---|---:|---:|
| Color accuracy | 100.00% | 95.83% |
| Shape accuracy | 66.67% | 58.33% |
| Strict prompt accuracy | 66.67% | 54.17% |
| Bilingual both correct | 66.67% | 50.00% |

**Keep fixed clipping.** These aggregate scores do NOT provide
per-triangle accuracy in dynamic mode, so do not falsely claim a
specific triangle hit rate without the saved per-shape diagnostic.

### No-training next step: held-out geometry reconstruction

`scripts.pixel_geometry_audit` compares reconstruction of 32×32
held-out **real triangle/square training-target images** at selected
diffusion timesteps. Given the same ordinarily forward-noised x_t,
the trained model predicts RGB noise twice: once with the correct
English/Arabic shape word and once with that word swapped. It
reconstructs an x0 and compares the predicted foreground
silhouette directly to the actual validation image. Metrics are
mask IoU, foreground area, shape heuristic and how much using the
correct shape word helps. This is NOT a pure-noise generative-image
score and cannot certify triangle generation.

The aim is to distinguish whether the RGB denoiser already fails
to recover triangle geometry from normal q(x_t|x0) inputs, versus
primarily failing when its reverse trajectory begins from Gaussian
noise. It performs NO training, writes no checkpoint and introduces
no hardcoded shapes during generation.

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp

python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$BASE" || { echo "Missing baseline"; false; }

python3 -m scripts.pixel_geometry_audit \
  --checkpoint "$BASE" \
  --guidance-scale 5 --seeds 42 123 \
  --timesteps 5 20 50 80 \
  --output /tmp/sora-base-geometry-audit-20261103.json
```

Also inspect the **already generated** dynamic-clipping images by
shape to discover whether any triangles succeeded:

```bash
python3 -m scripts.pixel_diagnose \
  --evaluation-dir /tmp/sora-x0-ablation-1793713617-dynamic \
  --output /tmp/sora-dynamic-shape-diagnosis-20261103.json
```

The new geometry-audit tests are committed but have **not yet
been executed on the user's Mac**. Do not run another long training
trial until these shape-specific results are available.

## Geometry bottleneck: test the full reverse chain from known noisy shapes

The latest **91/91 Mac tests passed**. The 800-step spatial model's
one-step estimates from held-out validation x_t retained nearly
perfect shape information at t=5/20, but degraded strongly at
t=50 and t=80 (combined mask IoU **0.9528**, then **0.5382**;
combined correct-shape recognition **59.38%**, then **15.63%**).
Triangles were recognizable in much of the low-noise
reconstruction task. Yet pure-noise triangle generation still
fails, and dynamic clipping yielded **0% triangles** and lowered
circles to 75%. Keep the original fixed x0 policy.

The next read-only experiment, `scripts.pixel_partial_reverse_audit`,
starts at ordinary forward-noised **real held-out** square/triangle
images at t=20,50,80,99, then applies the complete learned RGB
posterior-mean reverse process down to t=0. Correct and swapped
English/Arabic shape words receive **the exact same starting x_t**;
square/triangle targets of one color+language+seed also receive the
same Gaussian epsilon. Outputs are learned RGB reconstructions, not
PIL-drawn shapes. This is **conditional reconstruction**, not
pure-noise text-to-image generation.

The script writes `report.json` and a preview grid of actual learned
output RGB pixels; it reports IoU to the known target silhouette,
correct-shape fraction, triangle and square fractions per starting
timestep, and correct-vs-swapped shape-caption advantage.

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git pull --ff-only
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

BASE="storage/artifacts/sora/nano/mvp/pixel-20261103T065647Z-ee054d71/checkpoint.pt"
test -f "$BASE" || { echo "Missing trained checkpoint"; false; }
OUT="/tmp/sora-partial-reverse-$(date +%s)"

python3 -m scripts.pixel_partial_reverse_audit \
  --checkpoint "$BASE" \
  --colors red --seeds 42 \
  --start-timesteps 20 50 80 99 \
  --guidance-scale 5 --cpu-threads 2 \
  --output-dir "$OUT"

open "$OUT/preview_grid.png"
```

New five tests validate all selected reverse steps, reproducibility,
correct-vs-swapped prompting, matched square/triangle q(x_t)
noise, nonmutating model weights and preview export. **Mac tests
not yet run.** No training changes or additional checkpoints.

If final triangles succeed when starting at t20/50 but not 80/99,
the bottleneck likely lies in the **high-noise generative
trajectory**. If they fail even when starting from t20 despite
near-perfect one-step recovery, investigate accumulated reverse
steps and conditioning/guidance mismatch. No further multi-
thousand-step CPU runs are justified before reviewing this result.


## Central BERHOT model storage (default for all Sora Nano tools)

**From this branch onward**, Sora training checkpoints, sample PNGs,
evaluation grids, metrics and read-only audits save by default to:

```text
/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/
```

This is the same central company-wide structure used for BERHOT model-family
artifacts, with each model's own subfolder. It works even when the terminal
is inside `models/sora/variants/nano/mvp`, because paths are anchored to
the repository root, not the current directory. Absolute path in the example
is specific to the current Mac; other clones resolve automatically.

**No new training is required to migrate earlier experiments.** Old Sora
checkpoints remain at:

```text
models/sora/variants/nano/mvp/storage/artifacts/sora/nano/mvp/<run-id>/checkpoint.pt
```

Review the migration before copying:

```bash
python3 -m scripts.migrate_storage
python3 -m scripts.migrate_storage --apply    # explicitly copy + verify
```

The command never moves, deletes or overwrites prior checkpoints. To check
where the next new run would save **without launching training**, use:

```bash
python3 -c 'from src.storage_paths import sora_artifacts,sora_releases; print("artifacts:",sora_artifacts()); print("releases:",sora_releases())'
```

The `--output-root` option is a training-command **flag**, not a standalone
shell command. You no longer need it for normal training. Similarly
`--output-dir` and `--output` can be omitted from Sora CLI evaluation,
sampling and audit commands to use fresh central artifact subdirectories.
Explicit output overrides (`/tmp/...`, for example) still take priority.

Central `storage/releases/sora/nano/<version>/` is reserved for a model
package that has **passed evaluation and been explicitly approved**. The
current experimental Sora model still fails triangle generation; no new
release has been created, and the migration copies **artifacts only**.

For future BERHOT model families, use the root-level helper
`berhot_storage.paths`. See `docs/storage.md` for all family folders and
release rules. Current Nabra and JSM storage is deliberately unchanged.

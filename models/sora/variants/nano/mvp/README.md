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
  --seed 42 --sample-steps 50 \
  --output /tmp/sora-pixel-arabic.png

OUT="/tmp/sora-pixel-eval-$(date +%s)"
python3 -m scripts.pixel_diffusion evaluate \
  --checkpoint "$CKPT" --seed 42 --sample-steps 50 \
  --output-dir "$OUT"
open "$OUT/preview_grid.png"
```

The evaluation exports 12 **real model-sampled PNGs** covering all six concepts
in both languages, along with a JSON report. It does **not** certify visual
quality. Saving into an existing output location is intentionally rejected.

### Known limitations and results

The first independent 850-step CPU training run of an early 16-channel
pixel-U-Net reached noise MSE ~0.07, but **images were still visibly noisy and
not recognizable**. This is a real failure of image fidelity despite correct
end-to-end software operation. It is not appropriate to merge or ship this
as a visually successful diffusion model. The new GitHub CI uses only a
short **smoke training run** and tests checkpoint → inference → PNG export.
The smoke run cannot establish image quality. Longer experiments, stronger
denoising objectives, better spatial/semantic conditioning, additional data,
quantitative quality measures, and visual review remain required.

Do not confuse this image-level diffusion with `layout-ddpm-v1`, which
diffuses only three geometry parameters, or with the shape CVAE and neural
renderer. All baselines are preserved separately.

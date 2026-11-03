# Sora Nano — Shape Field V2 (experimental, not validated)

## Findings from failed V1 pilot

The previous two-stage conditional mask-CVAE + neural RGB model
trained 40 mask steps and 30 appearance steps. Unit tests passed,
but learned RGB images failed 24/24 strict image gates.

On **2026-11-03**, the user's Mac passed **125 tests** and ran a
read-only, stage-isolated diagnosis on the saved V1 checkpoint
`shape-first-20261103T204620Z-c284ab89`.

| Shape | Held-out posterior mask IoU | Gaussian prior shape passes | Color correct with true mask |
|---|---:|---:|---:|
| Circle | 0.46582 | 6/8 | 2/8 |
| Square | 0.51364 | 0/8 | 2/8 |
| Triangle | 0.35879 | 0/8 | 2/8 |

With generated mask, color was also only 2/8 per shape.
Mean generated prior foreground areas clustered tightly at roughly
215–234 pixels regardless of shape. This is consistent with
shape-averaging / weak text shape-conditioning, while RGB failed
even when given the *true* shape. Posterior reconstruction itself
also underperformed. Longer training alone has no evidence
of solving these failures.

Read-only stage report:
`BERHOT/storage/artifacts/sora/nano/mvp/shape-first-diagnose-20261103T205957Z-6636e025/report.json`.

## New experimental approach

**Shape Field V2** is an entirely separate model architecture.
It is **not** a renamed or fine-tuned V1; all V1/V3/original
RGB checkpoints are left untouched.

The first stage is a learned **neural implicit geometry field**.
It receives normalized pixel coordinates, a sampled layout, and
a learned embedding of the specific shape word. A trainable
convolutional per-pixel MLP outputs a probability that each pixel
is foreground. A seeded Gaussian supplies **only position, size
and triangle-orientation variables**, rather than an encoded
mask from a mismatch-prone posterior. Training masks come from
real training RGB samples. The model learns occupancy itself:
**there are no polygon/ellipse/rectangle drawing calls or shape
templates during generation**.

The second stage is a **learned neural colorizer**. A distinct
word embedding selects the requested color word (Arabic or English),
a trainable MLP predicts foreground RGB, and a trainable background
parameter predicts the light backdrop. The resulting RGB is
combined with the *learned mask*. The compositing is a conventional
single-object alpha operation, not a procedural shape renderer.
There is **no hardcoded red/blue/green/yellow RGB lookup in the
inference model**.

**Important scope limitation:** the code identifies the supported
shape and color word IDs explicitly from the tiny v2 vocabulary.
This is a domain-specific multilingual conditioning MVP, NOT a
general-language encoder. There are only 4 colors, 3 shapes,
32×32 RGB, and two supported languages. No claim that this
approach scales directly to open-domain Sora image generation.

Files:
- `src/shape_field_v2.py`: learned mask field, learned colorizer,
  training objectives and noise-to-RGB inference.
- `scripts/shape_field_v2.py`: train, sample, evaluate and
  SHA256-verified checkpoint metadata.
- `tests/test_shape_field_v2.py`: 7 tests for conditioning,
  gradients, tensor geometry, inference, checkpoints, 24-image
  gallery, strict masks and preservation.

Outputs:
```text
BERHOT/storage/artifacts/sora/nano/mvp/
  shape-field-v2-<UTC>-<id>/
    mask_field.pt
    colorizer.pt
    run.json

  shape-field-v2-eval-<UTC>-<id>/
    preview_grid.png
    learned_masks_preview_grid.png
    report.json
    learned_masks/
      <24 mask PNGs>
    <24 RGB PNGs>
```

## Step 1 — Update and test

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git fetch origin feat/image-nano-foundation-mvp-20261102
git merge --ff-only FETCH_HEAD
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
```

Earlier Mac suite **125 passed**. Seven new V2 tests have been
added, so **expect 132 tests**. The seven new tests have **NOT**
yet run on the user's Mac. Stop on any failure; never assume
passing earlier tests proves new code correctness.

## Step 2 — Controlled smoke/pilot, NEW experiment

Only after all tests pass:

```bash
set -o pipefail

python3 -m scripts.shape_field_v2 train \
  --mask-steps 80 \
  --color-steps 60 \
  --batch-size 12 \
  --hidden 64 \
  --word-dim 32 \
  --learning-rate 0.001 \
  --seed 42 \
  --device cpu \
  --cpu-threads 2 \
  --log-every 20
```

This trains both neural stages from scratch. The run is
new and doesn't touch the failed V1 checkpoint, the original
800-step RGB DDPM, or any release.

The command prints its **actual** `run_dir`. Copy it exactly:

```bash
RUN="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-field-v2-REPLACE_WITH_PRINTED_ID"
```

## Step 3 — Evaluate all 24 real learned outputs

```bash
python3 -m scripts.shape_field_v2 evaluate \
  --run-dir "$RUN" --seed 42
```

The printed `preview_grid` is the **learned RGB** gallery.
The printed `learned_masks_preview_grid` is the **mask
generator output**. Both are 24-prompt Arabic/English galleries.
The printed `strict_by_shape` lists separate mask vs RGB
geometry pass counts and color-aware prompt pass counts.

Look at the two grids to locate the responsible stage. A
round mask with wrong color is a colorizer failure; good color
with wrong polygon is a geometry failure. Neither stage is
allowed to substitute procedural shape templates at inference.

## Step 4 — Compare with RGB-DDPM reference

The original RGB-DDPM baseline gallery is unchanged:

```text
BERHOT/storage/artifacts/sora/nano/mvp/
  v3-baseline-20261103T223204/guidance_5_seed_42/
```

Use the **exact printed** new `shape-field-v2-eval` directory:

```bash
BASE="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/v3-baseline-20261103T223204/guidance_5_seed_42"

EVAL="/absolute/path/to/printed/shape-field-v2-eval-id"

python3 -m scripts.shape_first_review \
  --baseline-dir "$BASE" \
  --candidate-dir "$EVAL" \
  --min-triangle-passes 2
```

Cross-architecture seed numbers are not equivalent stochastic
initial states. The gate compares the same 24 color/shape/language
concepts and independent strict geometry/color criteria. Passing
it would ONLY qualify the experiment for further human review,
**not** approval for `storage/releases`.

The original baseline strict prompt accuracy was 0.4583.
The V1 pilot scored zero. Desired outcome is first **nonzero
recognizable triangles**, preservation of at least 4/8 circles
and 7/8 squares, no regression in color, and improved strict
prompt accuracy on all 24. More seeds, positions, and human
review are required before any production claim.

## Next steps depend on measured results

- If the mask field still gives 0/8 triangles but shape loss
  decreases, inspect learned word embeddings and the mask
  probability histograms before longer runs.
- If masks pass but final RGB fails, improve the learned
  colorizer separately; no need to throw away good geometry.
- If both begin to work, evaluate more independent seeds and
  then consider a *separate* new training run with larger
  budgets. Current training CLI does not resume optimizer
  state, so a later `train` starts from scratch.
- Never overwrite the original RGB checkpoint or promote
  an unvalidated image model.

This is a **hypothesis to test**, not a guaranteed fix or a
general-purpose image foundation model. The draft PR remains
unmerged and GitHub Actions stays manual-only.

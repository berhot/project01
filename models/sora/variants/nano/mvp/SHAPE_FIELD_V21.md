# Sora Nano Shape Field V2.1 — Shape-word separation experiment (opt-in)

## Status and evidence: no quality improvement claimed yet

The user's Mac passed **132/132 tests** for V2 and trained:

`storage/artifacts/sora/nano/mvp/shape-field-v2-20261103T211035Z-39d9b863/`

Original V2 strict 24 English/Arabic generated image results:

| Shape | Correct generated-mask geometry | Final strict prompt pass |
| --- | ---: | ---: |
| Circles | **8/8** | **5/8** |
| Squares | **0/8** | **0/8** |
| Triangles | **0/8** | **0/8** |

Color accuracy **0.875**, strict prompt accuracy **0.2083**.
The learned geometry loss fell from 1.390686 to 0.359572.
Despite lower loss, only circles were geometrically recognized;
no squares or triangles passed. Original RGB-DDPM still has
higher overall strict prompt accuracy (0.4583).

**This new recipe is an EXPERIMENT, NOT a proven fix.**

## Diagnosis: possible weak shape-word control

In original V2 geometry training, each image has its own
bounding-box layout and its own class label, but the model is not
explicitly asked to render different shapes at the EXACT SAME
layout. This allows a shape-averaging or geometry-biased solution.

V2.1 tests this specific hypothesis in two ways:

1. **Read-only word audit** on the saved V2: create circle,
   square and triangle prompt tokens while reusing exactly the
   same Gaussian-sampled layout. Measure foreground size,
   strict geometry pass rate, and binary IoU between generated
   masks for swapped shape words. IoU close to 1 means changing
   words barely changes the mask. Audit is not proof of causality
   but detects word insensitivity directly.
2. **Opt-in matched-layout training**: using *training-only*
   synthetic RGB examples with identical color, size, center and
   background, make three real target masks for circle/square/
   triangle. Hold the input layout fixed, including orientation,
   and supply Arabic + English captions. Train the SAME neural
   field to output all three different shapes using caption
   shape-word embeddings. An additional ranking term encourages
   correct shape words to reduce mask BCE relative to swapped
   shape words.

Unlike shape-first V1 and diffusion V3, this training operates on
explicit observed mask occupancy at one shared layout, so it tests
the actual error we measured. No shape templates, polygon
drawing, or fixed shape lookup are used by inference.

**Preservation:** V2.1 can initialize from the exact V2 run
via `--init-from-v2`. It clones the V2 neural-mask weights,
keeps an immutable frozen V2 teacher, optionally distills old
circle predictions at both seen and sampled layouts using
`--circle-preserve-weight`. It clones the working neural RGB
colorizer and supports `--color-steps 0` to keep the **same**
colorizer weights. The source V2 run is read-only. All outputs
are new `shape-field-v21-...` artifact runs.

The neural architectures and checkpoint keys stay the same as V2;
`run.json` records `training_recipe`,
`warm_start`, SHA256 checksums and objective hyperparameters.
No approved releases are written.

## Step 1 — Update source and run all tests

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git fetch origin feat/image-nano-foundation-mvp-20261102
git merge --ff-only FETCH_HEAD

cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
```

**Expect 140 tests**: 132 previously passed, plus 8 new
V2.1 paired-mask, text swapping, gradient, checkpoint,
source-immutability and read-only audit tests. These new tests
have **not yet been executed on user's Mac**.

## Step 2 — Read-only audit of saved V2 before changing training

```bash
BASE="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-field-v2-20261103T211035Z-39d9b863"

python3 -m scripts.shape_field_word_audit \
  --run-dir "$BASE" --seed 42 --device cpu
```

Inspect the returned `shape_word_mask_overlap` for
`circle_vs_square`, `circle_vs_triangle`, and
`square_vs_triangle`. IoU near 1 on the same layout implies
shape-word insensitivity. Also inspect
`by_shape` counts; current V2 evaluation measured 8/8
circles, 0/8 squares and triangles, but this audit may use
a slightly different batch layout implementation. Don't
replace those previously observed results.

## Step 3 — Small safeguarded V2.1 warm-start experiment

Only after the 140 tests pass. Stop if the read-only word audit
reveals an unexpected issue rather than simply low sensitivity.

```bash
BASE="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-field-v2-20261103T211035Z-39d9b863"

set -o pipefail
python3 -m scripts.shape_field_v2 train \
  --shape-v21 \
  --init-from-v2 "$BASE" \
  --circle-preserve-weight 0.2 \
  --pair-weight 1.0 \
  --rank-weight 0.1 \
  --rank-margin 0.03 \
  --paired-layout-count 4 \
  --mask-steps 40 \
  --color-steps 0 \
  --batch-size 12 \
  --hidden 64 \
  --word-dim 32 \
  --learning-rate 0.0003 \
  --seed 42 \
  --device cpu \
  --cpu-threads 2 \
  --log-every 10
```

This fine-tunes **a new copy** of the V2 geometry model.
`--color-steps 0` means the improved V2 RGB colorizer
is copied into the new checkpoint without additional training.
`--circle-preserve-weight 0.2` distills circle masks from a
frozen V2 teacher. A separate run is saved centrally as
`shape-field-v21-<new ID>/`. It does not overwrite the
V2, V1, V3 or RGB-DDPM checkpoints.

The 40 mask steps are a *controlled experiment*, not enough
to assert triangle learning. Don't immediately run longer.

## Step 4 — Measure whether V2.1 actually improves

Copy the actual printed `run_dir` into `NEW`:

```bash
NEW="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-field-v21-REPLACE_WITH_PRINTED_ID"

python3 -m scripts.shape_field_word_audit \
  --run-dir "$NEW" --seed 42 --device cpu

python3 -m scripts.shape_field_v2 evaluate \
  --run-dir "$NEW" --seed 42
```

Open the printed `preview_grid.png` and
`learned_masks_preview_grid.png`. The evaluation
reports `learned_mask_pass_count` separately from
`geometry_pass_count` and `strict_prompt_pass_count`.

Optionally compare against the original RGB-DDPM baseline:

```bash
ORIGINAL="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/v3-baseline-20261103T223204/guidance_5_seed_42"

# Replace with printed directory of new V2.1 evaluation:
EVAL="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-field-v2-eval-REPLACE_WITH_PRINTED_ID"

python3 -m scripts.shape_first_review \
  --baseline-dir "$ORIGINAL" \
  --candidate-dir "$EVAL" \
  --min-triangle-passes 2
```

The script can be reused because checkpoint architectures have
identical tensor shapes and `load_run` verifies the model
and vocabulary by SHA256; the run is explicitly marked with
the V2.1 recipe in metadata. The evaluation still saves
to a `shape-field-v2-eval-` named directory for backward
compatibility. If masks still ignore shape words or triangle
geometry remains 0/8, do not assume longer training fixes it.

## Release rules

- New V2.1 checkpoint is **experimental** until all 24 prompts
  are evaluated and compared with the V2 and RGB-DDPM baselines.
- Shapes are generated by learned weights; evaluation templates
  never enter inference.
- No training or GitHub Actions is automatically triggered.
- PR #10 stays draft, old checkpoints remain intact, no
  `storage/releases/sora/` writes or R2 uploads are done.


## Test fix — Arabic shape-only counterfactual (2026-11-04)

The first Mac run of the new V2.1 code discovered **140 tests,
one failure**: `test_correct_vs_wrong_shape_word_supervision_and_gradient`.
The counterfactual ranking loss rotated **entire caption token
sequences** between circle, square, and triangle. In Arabic, this
also rotated gender-dependent color adjectives: for example,
`دائرة صفراء` (yellow circle) versus `مثلث أصفر` (yellow triangle).
Therefore the supposed "shape-only" test also changed the COLOR
TOKEN in Arabic. This was a valid test failure and a research
confound, not proof of better/worse model generation.

The loss now replaces **only the shape token ID** with the
alternative target-shape word of the same language. All other
tokens, including Arabic gendered adjective, remain unchanged.
The regression test now asserts exactly one changed token per
prompt and identity for the whole non-shape token sequence.

The corrected code is committed but has **not yet passed Mac
testing**. Rerun:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git fetch origin feat/image-nano-foundation-mvp-20261102
git merge --ff-only FETCH_HEAD
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -p 'test_shape_field_v21.py' -v
python3 -m unittest discover -s tests -v
```

Only after all **140 tests pass**, proceed with the read-only
shape-word audit. The V2.1 training pilot remains experimental
and has **not** been launched.

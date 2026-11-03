# Sora Nano — Shape-first V1 (NEW learned architecture, experimental)

## Why another architecture?

The original 800-step RGB DDPM is the stable *research baseline*, NOT a
production release: its 24-prompt benchmark has 100% color, only 4/8
strict-geometry circles, 7/8 squares, **0/8 triangles**, and 45.83%
strict shape+color success. A teacher-preserved 20-step V3 high-noise
prompt-contrast fine-tune made strict circle quality worse (3/8) and
still generated 0/8 triangles. The V3 candidate was **rejected**.

The next hypothesis is **shape-first generation**:

```text
Arabic/English prompt ("yellow triangle" / "مثلث أصفر")
                    |
             learnable word encoder
                    |
 Gaussian z --> trained conditional binary-mask VAE decoder
                    |
           model-generated 32x32 mask
                    |
 text prompt --> trained neural RGB appearance model
                    |
             generated RGB pixels
```

The new architecture is **trained from scratch**, not a warm-start
of the old RGB DDPM. It has separate weight files:

```text
BERHOT/storage/artifacts/sora/nano/mvp/shape-first-<UTC>-<id>/
  mask.pt            # learned conditional mask CVAE
  appearance.pt      # trained text+mask RGB appearance network
  run.json           # both weights' SHA256, config, full provenance
```

`src/shape_first.py` inference uses only **PyTorch learned weights,
Gaussian latent noise, prompt tokens and arithmetic**. It does not
draw triangles, circles or squares with PIL or use a deterministic
hardcoded shape template to produce the mask. Synthetic drawing occurs
**only in the dataset builder during training**, consistent with
the earlier RGB diffusion workflow.

During training, the mask model learns observed segmentation masks
derived from synthetic RGB training images with reconstruction BCE,
Dice, and a KL latent prior. A second loss supplies paired training
square/circle/triangle masks at matched position/size/background:
**the same encoded latent** must reconstruct distinct shapes when
only the shape word changes. Pairs include both Arabic and English.
The trained RGB model learns the requested colors/background from
real training-image pixels and corresponding observed masks.

Inference first samples a Gaussian latent and generates a mask from
learned weights, then generates RGB with the learned appearance
network. The inference path never loads an example target shape.
It is limited to the **12 color/shape concepts in Arabic/English**,
32×32, and is NOT a general-purpose Sora image model. Good training
loss is not proof of good samples. Conditional VAEs can still
exhibit posterior collapse, average shape geometry, or biased priors.

## Step 1 — Update branch and test (does not train models)

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git fetch origin feat/image-nano-foundation-mvp-20261102
git merge --ff-only FETCH_HEAD
cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v
```

The last Mac suite passed 111. The new shape-first feature adds 10
more tests (7 training/inference, 3 cross-architecture review):
**expect 121 tests**, unless later additions change the count.
If tests fail, stop and return the traceback for a code fix.
No one has run this new architecture on the user's Mac yet.

## Step 2 — Short smoke/pilot from scratch, centrally stored

No original checkpoints are loaded, overwritten, renamed or deleted.

```bash
set -o pipefail
LOG="/tmp/sora-shape-first-pilot-$(date +%s).log"
python3 -m scripts.shape_first train \
  --mask-steps 40 \
  --rgb-steps 30 \
  --batch-size 12 \
  --mask-width 32 \
  --rgb-width 32 \
  --text-dim 48 \
  --latent-dim 12 \
  --beta 0.01 \
  --paired-weight 0.5 \
  --paired-layout-count 2 \
  --learning-rate 0.0007 \
  --seed 42 \
  --device cpu \
  --cpu-threads 2 \
  --log-every 10 | tee "$LOG"
```

The script prints its **actual run directory**, e.g.
`BERHOT/storage/artifacts/sora/nano/mvp/shape-first-.../`.
Copy that exact directory into the following variable:

```bash
RUN="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-first-REPLACE_WITH_PRINTED_ID"
```

This is only a **smoke experiment** to validate the pipeline and
inspect the first image outputs. It is much too early to assume
triangles will be recognizable.

## Step 3 — Generate text-to-image (learned mask, learned RGB)

```bash
python3 -m scripts.shape_first sample \
  --run-dir "$RUN" \
  --prompt "yellow triangle" \
  --seed 42

python3 -m scripts.shape_first sample \
  --run-dir "$RUN" \
  --prompt "مثلث أصفر" \
  --seed 42

python3 -m scripts.shape_first sample \
  --run-dir "$RUN" \
  --prompt "blue circle" \
  --seed 42
```

Each command saves two files centrally: the real trained-model RGB
output and the learned binary-mask output. The mask is a debug view
of the learned neural generator, **not** an ellipse/triangle drawn
during inference. If the mask is square for triangle text, the
geometry generator failed. If the mask is a triangle but the RGB is
unrecognizable, the learned appearance stage failed.

## Step 4 — Evaluate ALL 24 bilingual prompts

```bash
python3 -m scripts.shape_first evaluate \
  --run-dir "$RUN" --seed 42
```

The command prints its exact **RGB** `preview_grid.png` and
separate **learned-mask** `learned_masks_preview_grid.png` paths,
plus `report.json`, per-shape strict geometry pass counts,
bilingual consistency and color accuracy. It saves **24 generated
RGB PNGs and 24 corresponding actual learned mask PNGs** centrally.
The mask gallery makes it possible to see whether the failure is
in the trained mask generator or the trained RGB appearance network.

Open the printed `preview_grid.png` with `open "<EXACT_PATH>"`.

The mask model may have low prior-sampling quality even if its
training reconstruction is good. Inspect **all 24 images** and
do not promote an apparent numerical gain without manual review.

## Step 5 — Fair cross-architecture quality review

The previous baseline gallery is preserved:

```text
BERHOT/storage/artifacts/sora/nano/mvp/
  v3-baseline-20261103T223204/guidance_5_seed_42/
```

Take the printed **new** shape-first `evaluate` output directory,
then run the research-only comparison:

```bash
BASE="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/v3-baseline-20261103T223204/guidance_5_seed_42"
NEW_EVAL="/ABSOLUTE/PATH/TO/NEW/shape-first-eval-.../"

python3 -m scripts.shape_first_review \
  --baseline-dir "$BASE" \
  --candidate-dir "$NEW_EVAL" \
  --min-triangle-passes 2
```

The review checks:
- At least **2 of 8 recognizable triangles** and more than baseline.
- At least **4 of 8 circles**, as many as or more than the baseline.
- At least **7 of 8 squares**, as many as or more than the baseline.
- No color-class regression relative to baseline.
- Overall strict prompt accuracy improves.

This gate is intentionally **experimental**: passing it only
qualifies the architecture for further exploration, never a release.
For public-ready shape quality the target should ultimately be
substantially higher (ideally all 24), with independent seeds, more
styles/positions and human visual quality review.

**Cross-architecture seed caveat:** the same numeric seed
does not produce the same underlying noise distribution between
a CVAE and a DDPM. Comparison is by prompt and geometry metrics,
not by pairwise output equality or falsely "same initial noise".

## Step 6 — After the pilot

If 40/30 steps produce recognizable intermediate masks but incomplete
colors or outlines, a **new training run** with e.g. 500 mask / 350
appearance steps can be considered; do not assume this will fix
anything without inspecting the pilot. The script does not currently
resume an optimizer from an existing run; each `train` creates a
new experiment from scratch. Never run a longer experiment until the
short pilot has been reviewed.

The previous RGB DDPM checkpoints and V3 logs remain unchanged.
All output stays within central `BERHOT/storage/artifacts/sora/nano/mvp/`.
No automated training, GitHub Actions, R2 sync, PR merge, or publication
into `storage/releases/` is performed.

## 2026-11-03 pilot verdict: FAIL, diagnose stages before training

The user's Mac completed **121/121 original + shape-first tests** and a
40-step mask + 30-step appearance smoke run from scratch, saved as:

`storage/artifacts/sora/nano/mvp/shape-first-20261103T204620Z-c284ab89/`

The user then evaluated all 24 bilingual prompts successfully:

`storage/artifacts/sora/nano/mvp/shape-first-eval-20261103T204832Z-884e9bc7/`

Outputs include `preview_grid.png` and
`learned_masks_preview_grid.png`.

**Actual generation was poor**: strict circle, square, and triangle
geometry all **0/8**; strict prompt accuracy **0/24**; legacy
shape/prompt accuracy **0/24**; color accuracy **6/24**.
`scripts.shape_first_review` correctly rejected this architecture:
`research_gate_pass=false`, every acceptance check false, compared
against original RGB baseline strict prompt rate **0.4583**.
The mask loss fell from ~1.777 to ~1.198 and weighted RGB appearance
loss from ~1.393 to ~0.645, but falling training loss does not prove
that generation from the prior is good.

This does **not** establish whether the principal failure is:
- reconstruction learning at the mask VAE decoder (posterior);
- mismatch between sampled Gaussian prior and the encoded mask
  posterior or failure to respect text shape words;
- insufficient RGB appearance training with *perfect* masks; or
- distribution shift when the appearance model sees generated masks
  rather than training-time true masks.

Do **not** run a long 500/350-step training recipe until these separate
questions are measured. Do **not** promote this checkpoint or change
any of the original RGB DDPM weights.

### Stage-by-stage diagnosis (read-only, no retraining)

After syncing the latest draft branch and passing the **125 tests**
(121 before this change + four additional diagnosis tests), run:

```bash
cd /Users/farisalsaiari/Desktop/BERHOT
git switch feat/sora-nano
git fetch origin feat/image-nano-foundation-mvp-20261102
git merge --ff-only FETCH_HEAD

cd models/sora/variants/nano/mvp
python3 -m unittest discover -s tests -v

RUN="/Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/sora/nano/mvp/shape-first-20261103T204620Z-c284ab89"

python3 -m scripts.shape_first_diagnose \
  --run-dir "$RUN" --seed 42 --device cpu
```

The new `scripts/shape_first_diagnose.py` tool reads both existing
checkpoint files, validates their SHA-256 hashes, and uses held-out
validation images **only for diagnosing training behavior**.
It reports for each of circle/square/triangle and each color/language:

- **Posterior held-out mask IoU**: how well the mask decoder
  reconstructs an unseen mask supplied to the *encoder*.
- **Random-prior mask strict geometry and foreground size**:
  what the trained decoder makes without a real target mask.
- **Appearance color with true mask**: whether the learned RGB network
  can render the right color if geometry is given correctly.
- **Appearance color with learned mask**: whether the actual end-to-end
  generation behaves differently from clean-mask RGB.

All 24 examples, separate by shape, are preserved in a new JSON
under central `storage/artifacts/sora/nano/mvp/shape-first-diagnose-.../`.

The diagnostic **never** trains/changes model weights, uses no target
mask for the random-prior sampling path, and cannot approve a release.
It is deliberately model-architecture-specific; strict shape scoring
uses reference masks only as **measurement templates** and never
renders from them during inference.

**Interpret the report before deciding how to proceed**:
if held-out posterior reconstruction itself fails, stage-1
underfitting needs attention. If posterior is good but random-prior
shapes are invalid, the CVAE latent prior or conditioning must be
redesigned/trained. If RGB from clean masks is incorrect, stage-2
appearance learning needs attention. If only RGB from generated
masks fails, the model has a mask-distribution mismatch.

The current short pilot is **NOT accepted**. Draft PR #10 stays
unmerged; no R2 sync, automatic CI, or release was triggered.

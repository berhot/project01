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

# BERHOT models

Company model families live under `models/<family>/`. Each family owns its
architecture, variants, training resources, evaluations, and releases.

The current working model is the [JSM Nano MVP](models/jsm/variants/nano/mvp/README.md).

## Company storage

The [storage guide](docs/storage.md) defines shared datasets and artifacts for
all model families, with a local layout ready to map to two R2 buckets.

```text
BERHOT/
├── models/       # Model implementations and configurations
├── data/         # Git-tracked catalog, manifests, splits, and templates
├── pipelines/    # Shared ingestion and data preparation scaffold
├── storage/      # Local payloads; only folder markers are tracked
│   ├── data/     # raw/, processed/, curated/ → proposed berhot-data bucket
│   ├── artifacts/ # Family/variant/run outputs → proposed berhot-artifacts bucket
│   ├── incoming/ # Local imports awaiting validation
│   ├── cache/    # Re-downloadable training data and models
│   └── tmp/      # Scratch space
└── docs/
```

R2 is not configured yet. Existing JSM MVP data and checkpoints remain bundled
with the MVP, so its documented commands still work. Their migration is separate.

## Model families

```text
models/
├── jsm/
│   ├── architecture/          # Family-level architecture space
│   ├── configs/
│   ├── training/
│   ├── inference/
│   ├── tokenizer/
│   ├── post_training/
│   ├── evals/
│   ├── docs/
│   ├── releases/
│   └── variants/
│       ├── nano/
│       │   ├── mvp/           # Current runnable prototype
│       │   ├── 350m/          # Planned size variants
│       │   ├── 550m/
│       │   └── 1b/
│       ├── base/
│       ├── lara/
│       ├── lara-pro/
│       ├── barq/
│       └── wasel/
├── bahr/
├── edrak/
├── fann/
└── nabra/
```

New model families belong alongside `jsm`. Variant-specific implementations and
dependencies belong within their variant. New shared datasets and large artifacts
use the storage layout above. Move code into family-level directories when it is
actually shared between variants.
The existing family and size directories are scaffolding; the MVP is currently
self-contained.

## Run the current MVP

From the repository root on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r models/jsm/variants/nano/mvp/requirements.txt
python -m models.jsm.variants.nano.mvp.scripts.inference --prompt "السعودية" --max-new-tokens 200
```

See the [MVP guide](models/jsm/variants/nano/mvp/README.md) for training,
resuming, and shorter commands from inside the MVP directory.


## Canonical shared model output storage

All **new** model outputs belong under the single repo-root
`storage/artifacts/<family>/<variant>/<stage>/<run-id>/` tree;
validated versions may be published, after review, to
`storage/releases/<family>/<variant>/<version>/`. The shared helper
`berhot_storage.paths` anchors both locations to the repository
regardless of the current working directory. Sora Nano's five trainers,
samples and reports now default to this central layout. Existing JSM/Nabra
MVP paths remain unchanged; see [migration and company-wide rules](docs/storage.md).

Sora's earlier checkpoints were saved under its nested MVP
`storage/artifacts/sora/nano/mvp/`. To migrate them without deletion:

```bash
cd models/sora/variants/nano/mvp
python3 -m scripts.migrate_storage --run-id pixel-20261103T065647Z-ee054d71
python3 -m scripts.migrate_storage --apply --run-id pixel-20261103T065647Z-ee054d71
```

No Sora release has been approved or published; only experimental artifacts
are present on the training machine. Central releases are not produced just
because training finished.


## Sora Nano shape-first architecture experiment

A separate neural **text-to-mask CVAE → mask-to-RGB appearance network**
is available under `models/sora/variants/nano/mvp/`. It uses a
new experimental run format and the same centralized BERHOT artifacts
root. No trained shape-first weights or production release are
committed to GitHub. Follow
[the two-stage training and acceptance guide](models/sora/variants/nano/mvp/SHAPE_FIRST_V1.md)
for Mac testing, a safe small pilot and the actual bilingual 24-image
quality gate. The previous RGB DDPM checkpoint remains untouched.

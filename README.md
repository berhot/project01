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

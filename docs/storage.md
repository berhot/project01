# BERHOT storage guide

## Storage and repository responsibilities

Git holds model and pipeline code, configurations, dataset records, small
approved fixtures, and model cards. Large datasets and training outputs live
under `storage/` locally and will be uploaded to R2 after account setup.

This change creates a local scaffold only. No R2 buckets have been created and
no files have been uploaded. Existing JSM Nano MVP assets and defaults are
unchanged. Directory names do not indicate that datasets or models exist yet.

## Local directory layout

```text
data/                                  # Lightweight records tracked in Git
├── catalog/
├── manifests/
├── splits/
├── samples/
└── templates/

storage/
├── data/                              # Future berhot-data bucket contents
│   ├── raw/
│   │   ├── text/
│   │   ├── audio/
│   │   ├── images/
│   │   ├── video/
│   │   ├── 3d/
│   │   └── multimodal/
│   ├── processed/                     # Same modality groups as raw
│   └── curated/                       # Versioned training dataset releases
├── artifacts/                         # Future berhot-artifacts bucket contents
│   ├── jsm/
│   │   ├── nano/{mvp,350m,550m,1b}/
│   │   ├── base/{1b,3b,7b,14b}/
│   │   ├── lara/{30b,70b}/
│   │   ├── lara-pro/{30b,70b}/
│   │   ├── barq/
│   │   └── wasel/
│   ├── bahr/
│   ├── edrak/
│   ├── fann/
│   └── nabra/
├── incoming/                          # Local only: incomplete/unreviewed imports
├── cache/
│   ├── datasets/                      # Local only: training data cache
│   └── models/                        # Local only: downloaded model cache
└── tmp/                               # Local only: disposable scratch space
```

Braces above abbreviate separate directories. Add new families as peers of
`jsm`; add their variants as they are defined. No modality is assumed from a
family's name. Dataset collections are shared across families.

## Dataset naming and lifecycle

Use lowercase IDs with hyphens, stable sample IDs, and explicit versions:

```text
raw/<modality>/<collection-id>/<source-version>/
processed/<modality>/<collection-id>/<processing-version>/
curated/<dataset-id>/<release>/
```

1. Download or collect into `storage/incoming/`.
2. Verify completeness, origin, and permitted uses; register the collection in
   `data/catalog/` and publish originals into a new raw version.
3. Write transformations, annotations, and normalization outputs into a new
   processed version. Record the source version and processing configuration.
4. Build curated releases with fixed train/validation/test assignments and
   manifests. Each release may draw from several source collections.
5. Train using a specific curated release cached on the training machine.
6. Save outputs under a unique artifact run directory and record their checksums.

Treat published versions as immutable. Corrections produce a new version;
required removals must also update derived releases and catalogs. Keep a separate
backup of irreplaceable originals. A cache is disposable only when its contents
can be retrieved from a verified source.

The modality folders are for discovery. Organize each collection as a unit
inside them; do not split related assets between unrelated collections. For
speech plus transcripts or meshes plus renders, use `multimodal/` when that best
describes the source collection. Pick one canonical location and reference it
from multiple datasets rather than copying it for each model.

Example destinations (examples only; these datasets have not been created):

| Collection | Raw prefix | Potential use |
| --- | --- | --- |
| Arabic documents | `raw/text/arabic-documents/v1/` | Chat and language models |
| Speech recordings | `raw/audio/saudi-speech/v1/` | Audio processing models |
| Licensed design images | `raw/images/design-images/v1/` | Image generation |
| Clips | `raw/video/product-clips/v1/` | Video models |
| Mesh collection | `raw/3d/object-meshes/v1/` | 3D models |
| Meshes, renders, captions | `raw/multimodal/product-objects/v1/` | Text/image-to-3D |

Keep original formats in raw storage. Choose processed formats for the consuming
pipeline; do not force audio, images, or meshes into one common file format.
Text and metadata can use JSONL or Parquet; larger training datasets can use
shards when their loaders support them.

A paired example can look like this:

```text
raw/multimodal/product-objects/v1/object-000001/
├── mesh.glb
├── front.png
├── side.png
├── description.txt
└── metadata.json
```

Link these assets using the same sample ID in the manifest. Store URI, size,
SHA-256, and asset role for each file. Use plain storage URIs, not expiring signed
URLs. A source record should describe licensing and permitted uses; record
consent references where relevant without placing private records in Git.

Curated datasets can have:

```text
curated/<dataset-id>/<release>/
├── train/
├── validation/
├── test/
├── manifest.jsonl
├── splits.jsonl
└── dataset.json
```

The split directories may contain prepared shards. For a reference-only dataset,
use manifests pointing at existing processed assets instead of duplicating files.
Assign related documents, speakers, subjects, and objects to one split to avoid
training/evaluation leakage. Keep large or sensitive manifests in storage and
only their references and checksums in Git.

## Model artifacts

Use this convention:

```text
<family>/<variant>/<optional-size-or-stage>/<run-id>/
├── checkpoints/      # Weights, optimizer state, resumable training state
├── exports/          # Inference-ready weights and deployment packages
├── evaluations/      # Detailed evaluation results
├── samples/          # Generated text, audio, images, meshes, or video
├── logs/             # Training logs
├── config.json       # Exact configuration used
└── run.json          # Code commit, dataset versions, checksums, artifact list
```

For example:
`storage/artifacts/jsm/nano/mvp/<unique-run-id>/checkpoints/model.pt`.
Use a run ID containing a UTC timestamp and a unique suffix or experiment ID.
Do not reuse a completed run's path for a different experiment. Reference exact
run IDs and checksums in model release metadata rather than relying on `latest`.
Small human-readable evaluation summaries and model cards can remain in Git.

The directory scaffold stops at known variants; create run subdirectories only
when a run exists. Copy `data/templates/run.example.json` and fill it in.

## Git behavior

The root `.gitignore` excludes payloads under `storage/data`, `storage/artifacts`,
`storage/incoming`, `storage/cache`, and `storage/tmp`. Their `.gitkeep` markers
remain trackable, so the complete empty layout can be recreated by cloning.
Even JSON and README files inside those payload roots are ignored; place metadata
intended for Git under `data/`, `docs/`, or `models/` instead.

Ignore rules do not remove files already tracked by Git. In particular, the
existing MVP checkpoints and corpus have not been migrated or removed from
history. Future migration must upload and verify the files, update the MVP's
paths or download workflow, and then stop tracking the local payloads.

## Mapping to R2 later

The proposed bucket names are `berhot-data` and `berhot-artifacts`.
Keep them private and grant clients access to the required buckets. R2 exposes
an S3-compatible endpoint; use a dedicated AWS CLI profile for R2 credentials.
See [Cloudflare's AWS CLI guide](https://developers.cloudflare.com/r2/examples/aws/aws-cli/).

Run these commands only when ready to configure the account:

```bash
aws configure --profile r2
# Enter the R2 access key and secret; use region auto and output json.
export BERHOT_R2_ENDPOINT='https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com'

# Create each bucket once, if it does not already exist.
aws s3api create-bucket --bucket berhot-data --profile r2 --region auto --endpoint-url "$BERHOT_R2_ENDPOINT"
aws s3api create-bucket --bucket berhot-artifacts --profile r2 --region auto --endpoint-url "$BERHOT_R2_ENDPOINT"
```

Bucket creation requires an R2 token with bucket-management permission; an
object-only token is sufficient for subsequent uploads to its allowed buckets.
Keep credentials in the AWS profile or environment, outside Git.

From the repository root, preview both uploads:

```bash
aws s3 sync storage/data/ s3://berhot-data/ --profile r2 --region auto --endpoint-url "$BERHOT_R2_ENDPOINT" --no-follow-symlinks --exclude '.DS_Store' --exclude '*/.DS_Store' --dryrun
aws s3 sync storage/artifacts/ s3://berhot-artifacts/ --profile r2 --region auto --endpoint-url "$BERHOT_R2_ENDPOINT" --no-follow-symlinks --exclude '.DS_Store' --exclude '*/.DS_Store' --dryrun
```

Remove `--dryrun` to upload after reviewing the preview. These commands include
`.gitkeep`, so even scaffold-only folders are represented as object prefixes.
To upload only real files, add `--exclude '.gitkeep' --exclude '*/.gitkeep'`;
empty folders will then have no representation in R2.

`sync` transfers new or changed files and can replace objects at the same key.
These commands omit `--delete`, so they do not delete remote-only objects.
Use new version/run paths for published data and artifacts. Git ignore rules
do not filter AWS CLI uploads. See the [AWS sync reference](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html).

Do not sync `incoming`, `cache`, or `tmp`. Download the required dataset release
into the training machine's local cache before repeated training reads.

## DVC option

The human-readable R2 prefixes above are for direct uploads. If DVC is adopted,
give it a separate remote prefix or bucket and let it manage its content-addressed
objects. Do not upload this folder hierarchy into DVC's internal cache namespace
or manipulate that namespace manually. Git tracks the DVC metadata; DVC manages
the payload versions. DVC has not been initialized by this scaffold.

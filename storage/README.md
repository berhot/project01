# Local storage workspace

This directory is the local counterpart of company object storage.
See the [storage guide](../docs/storage.md) for naming, lifecycle, examples,
and terminal commands for R2.

| Local directory | Proposed destination | Purpose |
| --- | --- | --- |
| `data/` | `s3://berhot-data/` | Shared raw, processed, and curated datasets |
| `artifacts/` | `s3://berhot-artifacts/` | Checkpoints and outputs for every family |
| `incoming/` | Local only | Incomplete downloads and unreviewed imports |
| `cache/` | Local only | Re-downloadable datasets and models |
| `tmp/` | Local only | Disposable processing scratch space |

Payload files in these directories are ignored by Git. Only `.gitkeep` files
preserve the directory scaffold. Put documentation and metadata intended for
Git in the repository's `data/`, `docs/`, or model directories instead.
The existing MVP assets remain in their original MVP locations until migration.

The R2 bucket names are proposals. No buckets, credentials, or uploads have been
configured. `r2.example.json` records the planned mapping; it is documentation,
not automatically loaded by training code or sync commands.

Only `storage/data/` and `storage/artifacts/` are upload roots. Do not sync the
whole repository or `storage/` root.

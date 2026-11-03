"""Bridge to the single shared BERHOT storage package.

Sora's historical commands are run from inside the nested MVP path;
they cannot directly import repository-root modules without adding the
root to sys.path. Only this thin bridge performs that bootstrap.
Future model families should import berhot_storage.paths directly
when executed with repository root on PYTHONPATH.
"""
from pathlib import Path
import sys

for candidate in Path(__file__).resolve().parents:
    if (candidate/"docs"/"storage.md").is_file() and (candidate/"models").is_dir():
        _repository_root=candidate
        break
else:
    raise RuntimeError("Cannot locate BERHOT repository root")

if str(_repository_root) not in sys.path:
    sys.path.insert(0,str(_repository_root))

from berhot_storage.paths import artifact_root, release_root, new_artifact_run


def sora_artifacts():
    return artifact_root("sora","nano","mvp")


def sora_releases():
    return release_root("sora","nano")


def sora_run(prefix,output_root=None):
    return new_artifact_run("sora","nano","mvp",prefix,output_root)

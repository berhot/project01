"""Repository-root-anchored storage conventions for EVERY BERHOT model.

All model families share one storage ROOT while using segregated
<family>/<variant>/<stage> subdirectories. This module does not create
directories on import. Runtime outputs are always kept out of Git.

Callers may explicitly override a destination for tests or external
volumes. Never interpret a missing path relative to the current working
directory: that caused nested mvp/storage trees in early Sora trials.
"""
import re
import time
import uuid
from pathlib import Path

_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def repository_root():
    """Directory containing models/, docs/ and this Python package."""
    return Path(__file__).resolve().parents[1]


def storage_root():
    """One canonical project-wide location (not cwd-dependent)."""
    return repository_root() / "storage"


def _components(parts):
    for part in parts:
        if not isinstance(part,str) or not _COMPONENT.fullmatch(part):
            raise ValueError("Invalid storage path component: " + repr(part))
    return parts


def artifact_root(family, variant, stage=None):
    parts = (family,variant) if stage is None else (family,variant,stage)
    return storage_root() / "artifacts" / Path(*_components(parts))


def release_root(family, variant, version=None):
    parts = (family,variant) if version is None else (family,variant,version)
    # version supports semantic versions but must still be path-safe.
    if version is not None:
        if not isinstance(version,str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?",version):
            raise ValueError("Invalid release version")
        _components((family,variant))
        return storage_root() / "releases" / family / variant / version
    return storage_root() / "releases" / Path(*_components(parts))


def new_artifact_run(family,variant,stage,prefix,override_root=None):
    """Unique path for a NEW run; caller explicitly creates it with exist_ok=False.

    An explicit --output-root is kept as supplied (including relative
    locations, for compatibility with existing test fixtures).
    """
    _components((prefix,))
    base=Path(override_root).expanduser() if override_root is not None else artifact_root(family,variant,stage)
    run_id=(prefix+"-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
            +"-"+uuid.uuid4().hex[:8])
    return base / run_id

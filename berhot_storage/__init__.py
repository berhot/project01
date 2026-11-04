"""Shared BERHOT model-family storage paths.

Canonical payload home: <repository>/storage. Nothing is uploaded
or released automatically.
"""
from .paths import (
    repository_root, storage_root, artifact_root, release_root,
    new_artifact_run,
)

__all__ = (
    "repository_root","storage_root","artifact_root",
    "release_root","new_artifact_run",
)

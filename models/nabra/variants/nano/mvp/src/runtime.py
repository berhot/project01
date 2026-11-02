"""Shared device selection and atomic, versioned experiment utilities."""
import hashlib
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

import torch


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError("Device must be cpu, mps, cuda, or auto")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return torch.device(name)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(records):
    # Relocating a dataset does not change its identity. Extra provenance is included.
    portable = [{k: v for k, v in r.items() if k != "audio"} for r in records]
    return hashlib.sha256(json.dumps(portable, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    from importlib.metadata import version, PackageNotFoundError
    dependencies = {}
    for package in ("torch", "numpy", "safetensors", "sounddevice"):
        try: dependencies[package] = version(package)
        except PackageNotFoundError: dependencies[package] = None
    mvp = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(mvp)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ("src", "scripts", "configs") for p in (mvp/folder).rglob("*")
               if p.is_file() and p.suffix in {".py", ".json"}}
    return {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(),
            "code_commit": commit, "dependencies": dependencies, "source_sha256": sources}

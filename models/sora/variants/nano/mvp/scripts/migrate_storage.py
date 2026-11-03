"""Safely COPY older Sora mvp/storage artifact runs into BERHOT/storage.

Dry-run by default. --apply copies each run to a temporary staging
folder, verifies every file's SHA256, then publishes it as a new
run directory under the shared company storage root. Never deletes
local runs, overwrites a destination, or changes checkpoint weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from src.storage_paths import sora_artifacts


def legacy_artifacts():
    return Path(__file__).resolve().parents[1] / "storage" / "artifacts" / "sora" / "nano" / "mvp"


def _hash(path):
    checksum=hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024),b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _listing(root):
    paths={}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in artifacts: {path}")
        if path.is_file():
            paths[path.relative_to(root).as_posix()]=(_hash(path),path.stat().st_size)
    return paths


def migrate(source=None, destination=None, apply=False, run_ids=None):
    src=Path(source) if source is not None else legacy_artifacts()
    dst=Path(destination) if destination is not None else sora_artifacts()
    src,dst=src.expanduser().resolve(),dst.expanduser().resolve()
    if src==dst or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be distinct nonnested directories")
    if not src.exists():
        return {"source":str(src),"destination":str(dst),"mode":"copy" if apply else "dry-run",
                "runs":[],"status":"legacy source not present"}
    if not src.is_dir() or src.is_symlink():
        raise ValueError("Legacy source must be an ordinary directory")
    if run_ids is not None:
        run_ids=set(run_ids)
        if not run_ids or any((not name or "/" in name or "\\\\" in name
                               or name in (".","..") or name.startswith("."))
                              for name in run_ids):
            raise ValueError("run_ids must be safe, nonempty run directory names")
    runs=[]
    for run in sorted(src.iterdir()):
        if run_ids is not None and run.name not in run_ids:
            continue
        if run.is_symlink():
            raise ValueError(f"Symlink run is not permitted: {run}")
        if not run.is_dir():
            continue
        files=_listing(run)
        if not files:
            continue
        target=dst/run.name
        if target.exists() or target.is_symlink():
            if not target.is_dir() or _listing(target)!=files:
                raise FileExistsError(f"Destination collision; refusing overwrite: {target}")
            runs.append({"run":run.name,"files":len(files),"status":"already-identical"})
            continue
        if not apply:
            runs.append({"run":run.name,"files":len(files),"status":"would-copy"})
            continue
        dst.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sora-migration-",dir=dst) as staged:
            stage=Path(staged)/run.name
            shutil.copytree(run,stage,symlinks=False)
            if _listing(stage)!=files:
                raise IOError(f"SHA256 verification failed for run {run.name}")
            # Refuse a destination that appeared during copy rather than
            # ever deleting or overwriting another process's checkpoint.
            if target.exists():
                raise FileExistsError(f"Destination appeared during copy: {target}")
            os.rename(stage,target)
        runs.append({"run":run.name,"files":len(files),"status":"copied-verified"})
    return {"source":str(src),"destination":str(dst),
            "mode":"copy" if apply else "dry-run","runs":runs,
            "status":"completed"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply",action="store_true",
                        help="Copy and verify; omit for dry-run")
    parser.add_argument("--run-id",action="append",default=None,
                        help="Copy only this run ID (repeatable); omit for all")
    args=parser.parse_args()
    print(json.dumps(migrate(apply=args.apply,run_ids=args.run_id),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

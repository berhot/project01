
import hashlib
import json
import math
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from .model import ModelConfig, NabraASR
from .features import FeatureConfig
from .tokenizer import CharTokenizer
from .paths import get_release_dir, get_run_dir


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def package_release(
    checkpoint_path,
    version,
    max_cer,
    approve=False,
):
    if not approve:
        raise ValueError(
            "Explicit release approval required"
        )

    if not math.isfinite(max_cer) or max_cer < 0:
        raise ValueError("Invalid CER threshold")

    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    run_id = checkpoint["run_id"]
    epoch = int(checkpoint["epoch"])

    run_dir = get_run_dir(run_id).resolve()

    if not checkpoint_path.is_relative_to(
        (run_dir / "checkpoints").resolve()
    ):
        raise ValueError(
            "Checkpoint does not belong to run"
        )

    report_path = (
        run_dir / "evaluations"
        / f"epoch_{epoch:04d}.json"
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if report["run_id"] != run_id:
        raise ValueError("Evaluation run mismatch")

    if report["epoch"] != epoch:
        raise ValueError("Evaluation epoch mismatch")

    if report["checkpoint_sha256"] != sha256_file(
        checkpoint_path
    ):
        raise ValueError(
            "Checkpoint changed after evaluation"
        )

    cer = float(report["cer"])

    if not math.isfinite(cer) or cer > max_cer:
        raise ValueError(
            f"CER {cer:.4f} exceeds threshold"
        )

    model_config = ModelConfig(
        **checkpoint["model_config"]
    )

    tokenizer = CharTokenizer.from_dict(
        checkpoint["tokenizer"]
    )

    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("Vocabulary mismatch")

    model = NabraASR(model_config)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    release_dir = get_release_dir(version)

    if release_dir.exists():
        raise FileExistsError(release_dir)

    release_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = release_dir.parent / (
        f".{version}-{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()

    try:
        weights = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }

        weights_path = temporary / "model.safetensors"

        save_file(weights, str(weights_path))

        restored = load_file(
            str(weights_path),
            device="cpu",
        )

        for name, tensor in weights.items():
            if not torch.equal(tensor, restored[name]):
                raise ValueError(
                    f"Weight mismatch: {name}"
                )

        write_json(
            temporary / "config.json",
            {
                "model": asdict(model_config),
                "features": checkpoint["feature_config"],
                "task": "automatic-speech-recognition",
            },
        )

        write_json(
            temporary / "tokenizer.json",
            tokenizer.to_dict(),
        )

        names = (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
        )

        manifest = {
            "schema_version": "1.0.0",
            "model_family": "nabra",
            "variant": "nano",
            "version": version,
            "source_run_id": run_id,
            "epoch": epoch,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "approval": {
                "approved": True,
                "method": "explicit-cli-approval",
                "maximum_cer": max_cer,
            },
            "evaluation": {
                "cer": cer,
                "wer": report["wer"],
                "checkpoint_sha256": (
                    report["checkpoint_sha256"]
                ),
            },
            "files": {
                name: sha256_file(
                    temporary / name
                )
                for name in names
            },
        }

        write_json(
            temporary / "manifest.json",
            manifest,
        )

        if release_dir.exists():
            raise FileExistsError(release_dir)

        temporary.rename(release_dir)

    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return release_dir

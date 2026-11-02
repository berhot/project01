
import json
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file

from .features import (
    FeatureConfig,
    LogMelExtractor,
    read_wav,
)
from .model import ModelConfig, NabraASR
from .tokenizer import CharTokenizer
from .paths import get_release_dir


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model_config = ModelConfig(
        **checkpoint["model_config"]
    )
    feature_config = FeatureConfig(
        **checkpoint["feature_config"]
    )

    tokenizer = CharTokenizer.from_dict(
        checkpoint["tokenizer"]
    )

    model = NabraASR(model_config)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    return (
        model.to(device).eval(),
        tokenizer,
        feature_config,
    )


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_release(version, device):
    directory = get_release_dir(version)

    manifest = json.loads(
        (directory / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest["version"] != version:
        raise ValueError("Release version mismatch")

    if not manifest["approval"]["approved"]:
        raise ValueError("Release not approved")

    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(
                f"Release file hash mismatch: {name}"
            )

    config = json.loads(
        (directory / "config.json").read_text(
            encoding="utf-8"
        )
    )

    model_config = ModelConfig(
        **config["model"]
    )

    feature_config = FeatureConfig(
        **config["features"]
    )

    tokenizer = CharTokenizer.from_dict(
        json.loads(
            (directory / "tokenizer.json").read_text(
                encoding="utf-8"
            )
        )
    )

    model = NabraASR(model_config)

    model.load_state_dict(
        load_file(
            str(directory / "model.safetensors"),
            device="cpu",
        ),
        strict=True,
    )

    return (
        model.to(device).eval(),
        tokenizer,
        feature_config,
    )


@torch.inference_mode()
def transcribe(
    model,
    tokenizer,
    feature_config,
    wav_path,
):
    waveform = read_wav(
        wav_path,
        feature_config.sample_rate,
    )

    extractor = LogMelExtractor(feature_config)

    features = extractor(waveform)

    device = next(model.parameters()).device

    length = torch.tensor(
        [len(features)],
        dtype=torch.long,
    )

    logits, output_lengths = model(
        features.unsqueeze(0).to(device),
        length,
    )

    predicted = logits.argmax(
        dim=-1
    )[0, :int(output_lengths[0])]

    return tokenizer.ctc_decode(
        predicted.cpu().tolist()
    )


import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from safetensors.torch import load_file

from .model import JSMModel, ModelConfig
from .paths import get_release_dir
from .tokenizer import CharTokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_model(config_data, tokenizer, weights, device):
    config = ModelConfig(**config_data)

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Vocabulary size mismatch")

    model = JSMModel(config)

    model.load_state_dict(
        weights,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


def load_checkpoint(path, device):
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("tokenizer_type") != "char-v1":
        raise ValueError("Unsupported tokenizer")

    tokenizer = CharTokenizer(
        checkpoint["tokenizer_chars"]
    )

    model = build_model(
        checkpoint["model_config"],
        tokenizer,
        checkpoint["model_state_dict"],
        device,
    )

    return model, tokenizer, int(checkpoint["step"])


def load_release(version, device):
    directory = get_release_dir(version)

    manifest_path = directory / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("version") != version
        or manifest.get("model_family") != "jsm"
        or manifest.get("variant") != "nano"
    ):
        raise ValueError("Release identity mismatch")

    if not manifest.get("approval", {}).get(
        "approved", False
    ):
        raise ValueError("Release is not approved")

    for filename in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
    ):
        expected = manifest["files"][filename]["sha256"]
        actual = sha256_file(directory / filename)

        if expected != actual:
            raise ValueError(
                f"Release integrity failure: {filename}"
            )

    config_data = json.loads(
        (directory / "config.json").read_text(
            encoding="utf-8"
        )
    )

    tokenizer_data = json.loads(
        (directory / "tokenizer.json").read_text(
            encoding="utf-8"
        )
    )

    tokenizer = CharTokenizer.from_dict(
        tokenizer_data
    )

    weights = load_file(
        str(directory / "model.safetensors"),
        device="cpu",
    )

    model = build_model(
        config_data,
        tokenizer,
        weights,
        device,
    )

    return (
        model,
        tokenizer,
        int(manifest["training_step"]),
    )


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=20,
):
    if not prompt:
        raise ValueError("Prompt cannot be empty")

    if max_new_tokens < 1:
        raise ValueError(
            "max_new_tokens must be positive"
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive"
        )

    if top_k < 1:
        raise ValueError("top_k must be positive")

    device = next(model.parameters()).device

    tokens = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        context = tokens[
            :, -model.config.block_size:
        ]

        logits, _ = model(context)
        scores = logits[:, -1, :] / temperature

        k = min(top_k, scores.size(-1))

        values, _ = torch.topk(scores, k)

        scores = scores.masked_fill(
            scores < values[:, [-1]],
            float("-inf"),
        )

        probabilities = F.softmax(
            scores,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    return tokenizer.decode(
        tokens[0].tolist()
    )


def main():
    parser = argparse.ArgumentParser(
        description="JSM Nano MVP — Inference"
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--checkpoint",
        help="Path to a .pt checkpoint",
    )

    source.add_argument(
        "--release",
        help="Release version, for example 0.1.0",
    )

    parser.add_argument("--prompt", required=True)

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()

    if args.release:
        model, tokenizer, step = load_release(
            args.release,
            device,
        )
        source_name = f"Release {args.release}"
    else:
        model, tokenizer, step = load_checkpoint(
            args.checkpoint,
            device,
        )
        source_name = args.checkpoint

    print("JSM Nano MVP — Inference")
    print("------------------------")
    print("Source:", source_name)
    print("Training step:", step)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
    print("Vocabulary:", tokenizer.vocab_size)
    print("Context length:", model.config.block_size)
    print("Device:", device)
    print("Seed:", args.seed)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n--- Generated Text ---")
    print(output)
    print("----------------------")


if __name__ == "__main__":
    main()

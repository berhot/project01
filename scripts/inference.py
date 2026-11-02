
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.model import JSMModel, ModelConfig
from scripts.tokenizer import CharTokenizer


def get_device():
    """
    Select available hardware:
    NVIDIA CUDA, Apple MPS, or CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """
    Load a trained JSM model from a .pt checkpoint.

    Restore:
    - Model architecture
    - Trained weights
    - Character vocabulary
    """

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("tokenizer_type") != "char-v1":
        raise ValueError(
            "Unsupported tokenizer. Expected char-v1."
        )

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    tokenizer = CharTokenizer(
        checkpoint["tokenizer_chars"]
    )

    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match model."
        )

    # Reconstruct the original Transformer.
    model = JSMModel(config)

    # Restore learned parameters.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)

    # Disable training behavior.
    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("JSM Mini — Inference")
    print("--------------------")
    print(f"Checkpoint: {path}")
    print(f"Training step: {checkpoint['step']}")
    print(f"Parameters: {total_params:,}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Context length: {config.block_size}")
    print(f"Device: {device}")

    return model, tokenizer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=20,
):
    """
    Generate new text autoregressively.

    Each generated character is predicted
    from the previous context.
    """

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if max_new_tokens < 1:
        raise ValueError(
            "max_new_tokens must be positive."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be positive."
        )

    device = next(model.parameters()).device

    # Convert Arabic/English characters to token IDs.
    token_ids = tokenizer.encode(prompt)

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    # Generate one character token at a time.
    for _ in range(max_new_tokens):

        # Respect the model's context limit.
        context = tokens[
            :, -model.config.block_size:
        ]

        # Run the trained Transformer.
        logits, _ = model(context)

        # Get predictions for the next token.
        next_logits = logits[:, -1, :]

        # Adjust probability distribution.
        next_logits = next_logits / temperature

        # Restrict to highest scoring tokens.
        k = min(
            top_k,
            next_logits.size(-1),
        )

        values, _ = torch.topk(
            next_logits,
            k,
        )

        threshold = values[:, [-1]]

        next_logits = next_logits.masked_fill(
            next_logits < threshold,
            float("-inf"),
        )

        # Convert scores to probabilities.
        probabilities = F.softmax(
            next_logits,
            dim=-1,
        )

        # Sample one new token.
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # Append it to the sequence.
        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    # Convert generated token IDs back to text.
    output_tokens = tokens[0].tolist()

    return tokenizer.decode(output_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="JSM Mini — Real Transformer Inference"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/jsm_mini_char.pt",
        help="Path to the trained .pt checkpoint",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input text for generation",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of new character tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Limit sampling to top-k tokens",
    )

    # NEW: Reproducible sampling.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")

    if args.temperature <= 0:
        parser.error("--temperature must be > 0")

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    # Set the random seed before inference.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()

    model, tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")

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

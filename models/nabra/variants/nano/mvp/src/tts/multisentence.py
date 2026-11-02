"""Corpus readiness and deterministic multi-utterance scheduling for Nabra TTS."""
import random
from collections import Counter

from ..corpus_preparation import check_leakage, prompt_key
from .dataset import inspect_sources, speaker_key


def sample_index(step, count, seed):
    """1-based training step -> deterministic shuffled-per-epoch example index.

    Independent of global RNG and safe to recompute after restoring a checkpoint.
    """
    if not isinstance(step, int) or step < 1 or not isinstance(count, int) or count < 1:
        raise ValueError("step/count must be positive integers")
    if not isinstance(seed, int):
        raise ValueError("seed must be integer")
    epoch, position = divmod(step - 1, count)
    permutation = list(range(count))
    random.Random(f"{seed}:{epoch}").shuffle(permutation)
    return permutation[position]


def readiness(train_source, val_source, min_train=20, min_distinct=10):
    """Inspect actual Nabra-compatible WAVs; never claim speech quality."""
    if min_train < 1 or min_distinct < 2:
        raise ValueError("Invalid corpus thresholds")
    records = inspect_sources(train_source, val_source)
    train, val = records["train"], records["val"]
    check_leakage(train, val)
    unique = len({prompt_key(r["transcript"]) for r in train})
    train_speakers = {speaker_key(r) for r in train}
    val_speakers = {speaker_key(r) for r in val}
    if not val_speakers <= train_speakers:
        raise ValueError("Validation contains speakers absent from training; cannot evaluate this TTS speaker embedding")
    counts = Counter(speaker_key(r) for r in train)
    report = {
        "train_recordings": len(train),
        "validation_recordings": len(val),
        "distinct_train_prompts": unique,
        "train_speakers": dict(sorted(counts.items())),
        "train_seconds": sum(r["duration_seconds"] for r in train),
        "val_seconds": sum(r["duration_seconds"] for r in val),
        "readiness": "ready-for-experimental-training" if len(train) >= min_train and unique >= min_distinct else "insufficient-diversity",
        "quality_verified": False,
        "warnings": [],
    }
    if len(train) < min_train:
        report["warnings"].append(f"Fewer than {min_train} training clips")
    if unique < min_distinct:
        report["warnings"].append(f"Fewer than {min_distinct} distinct training prompts")
    return report

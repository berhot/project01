"""Holdout prompt and signal checks for newly synthesized Nabra speech.

This is a *diagnostic*, not an intelligibility or MOS evaluator. No ASR model,
audio downloads, reference voices or training modifications are performed.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from ..ingestion import sha256_file
from ..runtime import atomic_json
from .inference import load_checkpoint, synthesize, write_wav


def normalize(text):
    return " ".join(text.strip().split())


def prompts_from_file(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [normalize(s) for s in lines if s.strip() and not s.lstrip().startswith("#")]
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("Prompts must be nonempty and unique")
    return prompts


def signal_metrics(waveform):
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Waveform must be nonempty, one-dimensional and finite")
    values = waveform.detach().cpu().float()
    rms = float(values.square().mean().sqrt())
    peak = float(values.abs().max())
    clipped = float((values.abs() >= 0.999).float().mean())
    return {"rms": rms, "peak": peak, "clip_fraction": clipped,
            "silent": rms < 0.002, "clipped": clipped > 0.01}


def evaluate_unseen(checkpoint, prompts_file, output_dir, speaker=None, max_frames=400,
                    device="cpu", train_sources=None):
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    texts = prompts_from_file(prompts_file)
    # Caller may provide the train manifest(s) to enforce an exact normalized-text
    # holdout. Without them, we cannot claim that prompts are truly unseen.
    training_texts = set()
    for source in train_sources or []:
        for line in Path(source).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                training_texts.add(normalize(row["transcript"]))
    overlaps = sorted(set(texts) & training_texts)
    if overlaps:
        raise ValueError("Evaluation prompt overlaps training transcript: " + overlaps[0])
    bundle = load_checkpoint(checkpoint, device)
    # Validate all prompts and speaker before writing any output directory.
    tokenizer = bundle[2]
    for t in texts:
        encoded = tokenizer.encode(t)
        if len(encoded) > 512:
            raise ValueError("Evaluation prompt exceeds 512 tokens")
    if speaker is not None and speaker not in bundle[4]:
        raise ValueError("Unknown speaker")
    if speaker is None and len(bundle[4]) != 1:
        raise ValueError("Choose a speaker for multispeaker checkpoints")
    rows = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, prompt in enumerate(texts, 1):
        name = f"unseen_{index:03d}.wav"
        row = {"index": index, "text": prompt, "audio": name}
        try:
            waveform, info = synthesize(bundle, prompt, speaker=speaker, max_frames=max_frames)
            metrics = signal_metrics(waveform)
            write_wav(output_dir / name, waveform.cpu(), info["sample_rate"])
            row.update({"status": "generated", "duration_seconds": info["duration_seconds"],
                        "sha256": sha256_file(output_dir / name), **metrics})
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    n_fail = sum(r["status"] != "generated" or r.get("silent") or r.get("clipped") for r in rows)
    report = {
        "schema_version": 1, "task": "nabra-unseen-tts-diagnostic",
        "checkpoint_sha256": sha256_file(checkpoint),
        "prompts_sha256": sha256_file(prompts_file),
        "train_manifests_checked": len(train_sources or []),
        "holdout_overlap_checked": bool(train_sources),
        "speaker": speaker or bundle[4][0],
        "records": rows, "prompt_count": len(texts),
        "signal_warning_count": n_fail,
        "signal_checks_pass": n_fail == 0,
        "human_intelligibility": "not evaluated",
        "generalization": "not established",
        "release_ready": False,
    }
    atomic_json(output_dir / "report.json", report)
    return report

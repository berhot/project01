"""Run a bounded, non-release evaluation of diagnostic TTS checkpoints.

No ground-truth waveform is assumed for unseen text: waveform/attention metrics
are diagnostics only, never intelligibility scores.
"""
import json
import math
from pathlib import Path

import torch

from ..ingestion import sha256_file
from ..runtime import atomic_json, select_device
from .diagnostics import restore, attention_metrics
from .inference import write_wav
from .tokenizer import TextTokenizer

DEFAULT_PROMPTS = (
    ("seen", "مرحبا أنا أتعلم الذكاء الاصطناعي"),
    ("unseen_short", "صباح الخير"),
    ("unseen_question", "كيف حالك اليوم؟"),
    ("unseen_long", "أريد أن أتعلم اللغة العربية وأتحدث بوضوح."),
    ("unseen_numbers", "لدي ثلاثة كتب وخمسة أقلام."),
    ("unseen_dialect", "وش أخبارك اليوم؟"),
)


def assess_waveform(waveform, sample_rate=16000):
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Expected nonempty mono waveform")
    if not torch.isfinite(waveform).all():
        raise ValueError("Nonfinite synthesized audio")
    signal = waveform.detach().float().cpu()
    return {
        "duration_seconds": signal.numel() / sample_rate,
        "rms": float(signal.square().mean().sqrt()),
        "peak": float(signal.abs().max()),
        "clip_fraction": float((signal.abs() >= 1).float().mean()),
        "near_silence_fraction": float((signal.abs() < 1e-4).float().mean()),
    }


def read_prompts(path):
    if path is None:
        return list(DEFAULT_PROMPTS)
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        identifier = item.get("id", f"prompt_{index+1:03d}")
        text = item["text"]
        if not isinstance(identifier, str) or not identifier.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Prompt id must be alphanumeric, _ or -")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Prompt text cannot be empty")
        rows.append((identifier, text))
    if len(rows) < 2 or len({k for k, _ in rows}) != len(rows):
        raise ValueError("Need two or more prompts with unique IDs")
    return rows


@torch.inference_mode()
def evaluate(acoustic_checkpoint, vocoder_checkpoint, output, prompts=None, device_name="cpu", max_frames=800):
    if not 1 <= max_frames <= 4000:
        raise ValueError("max_frames outside 1..4000")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation")
    prompts_list = read_prompts(prompts)
    device = select_device(device_name)
    acoustic, ac = restore(acoustic_checkpoint, device)
    vocoder, vc = restore(vocoder_checkpoint, device)
    if ac["stage"] != "acoustic" or vc["stage"] != "vocoder":
        raise ValueError("Wrong checkpoint stages")
    if ac["mel_config"] != vc["mel_config"]:
        raise ValueError("Acoustic/vocoder Mel configurations differ")
    if ac["record"]["audio_sha256"] != vc["record"]["audio_sha256"]:
        raise ValueError("Checkpoints learned different reference recordings")
    tokenizer = TextTokenizer.from_dict(ac["tokenizer"])
    acoustic.eval()
    vocoder.eval()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for identifier, text in prompts_list:
            ids = tokenizer.encode(text)
            mel, generation = acoustic.generate(
                torch.tensor([ids], device=device),
                max_frames=max_frames,
                return_attention=True,
            )
            attention = generation.pop("attention")
            waveform = vocoder(mel)[0].cpu()
            metrics = assess_waveform(waveform)
            metrics["attention"] = attention_metrics(attention)
            metrics["stop_predicted"] = bool(generation["stop_predicted"])
            metrics["frame_limit_reached"] = bool(generation["frame_limit_reached"])
            metrics["mel_frames"] = int(mel.shape[1])
            output_file = destination / (identifier + ".wav")
            write_wav(output_file, waveform)
            rows.append({
                "id": identifier,
                "text": text,
                "wav_file": output_file.name,
                "sha256": sha256_file(output_file),
                "metrics": metrics,
                "listener_review": {
                    "intelligible": None,
                    "pronunciation": None,
                    "hiss": None,
                    "notes": None,
                },
            })
        report = {
            "schema_version": 1,
            "evaluation_type": "diagnostic-unseen-text",
            "production_ready": False,
            "quality_status": "pending-human-listening-review",
            "acoustic_checkpoint_sha256": sha256_file(acoustic_checkpoint),
            "vocoder_checkpoint_sha256": sha256_file(vocoder_checkpoint),
            "reference_training_text": ac["record"]["transcript"],
            "note": "Spectral statistics and successful WAV creation do not establish intelligibility or generalization.",
            "prompts": rows,
        }
        atomic_json(destination / "evaluation.json", report)
        (destination / "LISTENING_CHECKLIST.md").write_text(
            "# Listening checklist\n\nFor each generated WAV, record recognizable words, "
            "pronunciation errors, hiss, intelligibility and whether the text was seen in training. "
            "Do not assess unseen intelligibility from spectral metrics alone.\n",
            encoding="utf-8",
        )
        return report
    except BaseException:
        (destination / "INCOMPLETE").write_text("Evaluation failed; do not treat as complete.\n")
        raise

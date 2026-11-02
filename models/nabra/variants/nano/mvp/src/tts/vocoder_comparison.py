"""Compare baseline and recovered vocoders on exactly the same source audio.

Read-only checkpoint/source use; produces a new directory of listening WAVs
and a report. No audio-quality claims without human listening.
"""
from pathlib import Path

import torch

from ..features import read_wav
from ..ingestion import sha256_file
from ..runtime import atomic_json
from .dataset import MelExtractor
from .diagnostics import restore, wave_metrics
from .inference import load_checkpoint, write_wav


@torch.inference_mode()
def compare_vocoders(standard_checkpoint, recovery_checkpoint, output, record_index=0, split="train",
                     device="cpu"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(record_index, int) or record_index < 0 or split not in ("train", "val"):
        raise ValueError("Invalid split or recording index")

    standard_data = torch.load(standard_checkpoint, map_location="cpu", weights_only=True)
    if standard_data.get("schema_version") != 1 or standard_data.get("task") != "tts":
        raise ValueError("Baseline checkpoint must be a standard TTS checkpoint")
    records = standard_data["source_manifest"][split]
    if record_index >= len(records):
        raise ValueError("Recording index out of range")
    record = records[record_index]
    source = Path(record["audio"])
    if sha256_file(source) != record["audio_sha256"]:
        raise ValueError("Source recording hash mismatch")

    baseline = load_checkpoint(standard_checkpoint, device)
    recovered, metadata = restore(recovery_checkpoint, device)
    if metadata["stage"] != "vocoder":
        raise ValueError("Recovery checkpoint must be a vocoder")
    if metadata["record"]["audio_sha256"] != record["audio_sha256"]:
        raise ValueError("Vocoder recovery was trained on a different recording")
    if metadata["mel_config"] != standard_data["config"]["features"]:
        raise ValueError("Baseline and recovered Mel extraction configurations differ")

    source_audio = read_wav(source).cpu()
    mel = MelExtractor(baseline[3])(source_audio).unsqueeze(0).to(device)
    baseline_wave = baseline[1](mel)[0].cpu()
    recovered_wave = recovered(mel)[0].cpu()
    # Reference comparison uses the *identical* waveform and Mel condition.
    if min(len(source_audio), len(baseline_wave), len(recovered_wave)) <= 0:
        raise ValueError("Empty reconstruction")
    outputs = {
        "standard": baseline_wave,
        "recovered": recovered_wave,
    }
    metrics = {}
    for name, audio in outputs.items():
        n = min(len(audio), len(source_audio))
        metrics[name] = wave_metrics(audio[:n], source_audio[:n])
        metrics[name]["length_mismatch_samples"] = int(len(audio) - len(source_audio))
        metrics[name]["vocoder_parameters"] = int(sum(p.numel() for p in
            (baseline[1] if name == "standard" else recovered).parameters()))

    output.mkdir(parents=True, exist_ok=False)
    write_wav(output / "reference.wav", source_audio)
    for name, audio in outputs.items():
        write_wav(output / f"{name}.wav", audio)
    report = {
        "schema_version": 1,
        "task": "nabra-vocoder-controlled-comparison",
        "source_audio_sha256": sha256_file(source),
        "baseline_checkpoint_sha256": sha256_file(standard_checkpoint),
        "recovery_checkpoint_sha256": sha256_file(recovery_checkpoint),
        "source_transcript": record["transcript"],
        "record_index": record_index,
        "split": split,
        "recovery_head": metadata["model_config"]["head"],
        "baseline_head": baseline[1].config.head,
        "results": metrics,
        "output_files": ["reference.wav", "standard.wav", "recovered.wav"],
        "human_intelligibility": "not evaluated",
        "generalization": "not established; recovery checkpoint may be single-clip overfit",
        "release_ready": False,
    }
    atomic_json(output / "report.json", report)
    return report

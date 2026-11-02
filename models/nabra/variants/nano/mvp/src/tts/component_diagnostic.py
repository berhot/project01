"""Read-only teacher-forcing, autoregressive and vocoder component diagnostics.

Neither a signal metric nor teacher-forced reconstruction proves intelligibility.
"""
import json
from pathlib import Path

import torch

from ..ingestion import sha256_file
from ..runtime import atomic_json
from .dataset import TTSDataset, collate_tts
from .inference import load_checkpoint, write_wav


def attention_summary(attention):
    if attention.ndim != 3 or attention.shape[0] != 1:
        raise ValueError("Expected attention [1, decoder_steps, text_tokens]")
    a = attention[0].detach().float().cpu()
    positions = a.argmax(dim=-1)
    return {
        "steps": int(len(positions)),
        "unique_peak_positions": int(positions.unique().numel()),
        "first_peak": int(positions[0]),
        "last_peak": int(positions[-1]),
        "mean_peak_probability": float(a.max(dim=-1).values.mean()),
    }


@torch.inference_mode()
def stop_summary(logits, valid_steps):
    p = logits[0, :valid_steps].sigmoid().detach().cpu()
    return {
        "valid_steps": int(valid_steps),
        "first_probability": float(p[0]),
        "last_probability": float(p[-1]),
        "max_probability": float(p.max()),
        "last_step_probability": float(p[-1]),
    }


@torch.inference_mode()
def inspect_components(checkpoint, output, count=3, split="train", max_frames=400, device="cpu",
                       save_audio=True):
    checkpoint = Path(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if count < 1 or max_frames < 1 or max_frames > 4000 or split not in {"train", "val"}:
        raise ValueError("Invalid diagnostic arguments")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data.get("task") != "tts" or data.get("schema_version") != 1:
        raise ValueError("Expected a standard Nabra TTS checkpoint")
    records = data["source_manifest"][split][:count]
    if not records:
        raise ValueError("No recordings in split")
    for record in records:
        if sha256_file(record["audio"]) != record["audio_sha256"]:
            raise ValueError("Source audio hash changed")
    bundle = load_checkpoint(checkpoint, device)
    acoustic, vocoder, tokenizer, features, speakers = bundle
    dataset = TTSDataset(records, tokenizer, features, speakers)
    # All input validation happens before the destination directory is created.
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (record, item) in enumerate(zip(records, dataset), 1):
        batch = collate_tts([item])
        tokens = batch["tokens"].to(device)
        target_mel = batch["mel"].to(device)
        forced, forced_stop, forced_attn = acoustic(
            tokens, batch["text_lengths"], batch["speaker"].to(device), target_mel
        )
        # Self-fed generation: no reference Mel frames at inference.
        generated, gen_info = acoustic.generate(
            tokens, speaker=item["speaker"], max_frames=max_frames,
            min_frames=min(20, max_frames), return_attention=True
        )
        free_attn = gen_info.pop("attention")
        # Teacher-forced stop is compared at the *true* last decoder step.
        valid_steps = (len(item["mel"]) + acoustic.config.reduction - 1) // acoustic.config.reduction
        # Vocoder's output length can be shorter or longer depending on its head.
        reconstructed = vocoder(target_mel)[0].detach().cpu()
        reference = item["waveform"].detach().cpu()
        n = min(len(reconstructed), len(reference))
        if n == 0:
            raise ValueError("Vocoder produced empty audio")
        aligned = reconstructed[:n]
        truth = reference[:n]
        error = (aligned - truth).square().mean()
        snr = 10 * torch.log10(truth.square().mean().clamp_min(1e-12) / error.clamp_min(1e-12))
        compare_frames = min(len(generated[0]), len(item["mel"]))
        row = {
            "index": index, "text": record["transcript"], "source_audio_sha256": record["audio_sha256"],
            "target_mel_frames": int(len(item["mel"])),
            "teacher_forced_mel_l1": float((forced[0] - target_mel[0]).abs().mean()),
            "free_running_overlap_mel_l1": float(
                (generated[0, :compare_frames] - target_mel[0, :compare_frames]).abs().mean()),
            "free_running_frames": int(len(generated[0])),
            "free_running_stop_predicted": bool(gen_info["stop_predicted"]),
            "free_running_frame_limit_reached": bool(gen_info["frame_limit_reached"]),
            "teacher_forced_stop": stop_summary(forced_stop, valid_steps),
            "teacher_forced_attention": attention_summary(forced_attn),
            "free_running_attention": attention_summary(free_attn),
            "vocoder_reference_snr_db": float(snr),
            "vocoder_reference_samples": int(n),
        }
        if save_audio:
            name = f"reference_vocoder_{index:03d}.wav"
            write_wav(output / name, reconstructed, features.sample_rate)
            row["vocoder_audio"] = name
            row["vocoder_audio_sha256"] = sha256_file(output / name)
        rows.append(row)
    report = {
        "schema_version": 1, "task": "nabra-tts-components-diagnostic",
        "checkpoint_sha256": sha256_file(checkpoint), "split": split,
        "count": len(rows), "records": rows,
        "human_intelligibility": "not evaluated",
        "conclusion": "Diagnostic only; listen to reference-vocoder WAVs against original source files",
        "release_ready": False,
    }
    atomic_json(output / "report.json", report)
    return report

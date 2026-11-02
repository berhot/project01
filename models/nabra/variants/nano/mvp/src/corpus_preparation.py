"""Immutable speech standardization, provenance and leakage-safe splits."""
import hashlib
import json
import os
import random
import subprocess
import tempfile
import unicodedata
import wave
from pathlib import Path

from .ingestion import sha256_file, read_records, validate_splits


def prompt_key(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty transcript")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(c for c in normalized if not (0x064B <= ord(c) <= 0x065F or ord(c) == 0x0670))
    normalized = " ".join(normalized.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_groups(rows, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("val fraction must be between 0 and 1")
    groups = {}
    for row in rows:
        groups.setdefault(prompt_key(row["transcript"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two distinct prompts required for validation")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    val_keys = set(keys[:max(1, min(len(keys)-1, round(len(keys)*fraction)))])
    train = [r for key in keys if key not in val_keys for r in groups[key]]
    val = [r for key in keys if key in val_keys for r in groups[key]]
    return train, val


def check_leakage(train, val, speaker_disjoint=False):
    validate_splits(train, val)
    overlap = {prompt_key(x["transcript"]) for x in train} & {prompt_key(x["transcript"]) for x in val}
    if overlap:
        raise ValueError("Repeated normalized prompt across train and validation")
    if speaker_disjoint:
        a = {x.get("speaker_id") for x in train}
        b = {x.get("speaker_id") for x in val}
        if None in a | b or a & b:
            raise ValueError("Missing or shared speaker identities across splits")
    return True


def _jsonl(path, records):
    with path.open("x", encoding="utf-8") as stream:
        for item in records:
            stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def prepare(index, output, fraction=0.2, seed=42, ffmpeg="ffmpeg"):
    """Input index JSONL: audio path relative to index, transcript and provenance fields.

    Writes a NEW output directory; originals remain byte-identical.
    Requires FFmpeg for transcoding, including when input is WAV.
    """
    index = Path(index).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite corpus directory: " + str(output))
    sources = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        original = Path(item["audio"])
        if not original.is_absolute():
            original = index.parent / original
        original = original.resolve(strict=True)
        if original.is_relative_to(output):
            raise ValueError("Source audio cannot be inside new output")
        digest = sha256_file(original)
        if "audio_sha256" in item and item["audio_sha256"] != digest:
            raise ValueError("Source hash mismatch: " + str(original))
        prompt_key(item["transcript"])
        sources.append((item, original, digest))
    if not sources:
        raise ValueError("Empty corpus index")
    if len({s[2] for s in sources}) != len(sources):
        raise ValueError("Duplicate original audio files in corpus index")
    # Validate and group BEFORE creating output.
    dummy = [{"transcript": s[0]["transcript"]} for s in sources]
    split_groups(dummy, fraction, seed)
    output.mkdir(parents=True, exist_ok=False)
    try:
        converted = []
        for item, original, digest in sources:
            # Stable names and no source overwrite, even for different originals sharing filenames.
            destination = output / "audio" / (digest + ".wav")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                # Atomic destination, no source mutation; FFmpeg must exit cleanly.
                fd, temporary = tempfile.mkstemp(prefix=".audio-", suffix=".wav", dir=destination.parent)
                os.close(fd)
                try:
                    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                                    "-y", "-i", str(original), "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", temporary], check=True, capture_output=True)
                    with wave.open(temporary, "rb") as wav:
                        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
                            raise ValueError("Conversion output is not 16kHz mono PCM16")
                        duration = wav.getnframes() / 16000
                    if not 0.25 <= duration <= 20.0:
                        raise ValueError("Audio duration outside Nabra limits")
                    os.replace(temporary, destination)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if sha256_file(original) != digest:
                raise ValueError("Source changed during preparation")
            with wave.open(str(destination), "rb") as wav:
                duration = wav.getnframes() / 16000
            converted.append({
                "audio": "audio/" + destination.name,
                "audio_sha256": sha256_file(destination),
                "transcript": item["transcript"],
                "sample_rate": 16000, "duration_seconds": round(duration, 4),
                "prompt_sha256": prompt_key(item["transcript"]),
                "speaker_id": item.get("speaker_id"),
                "provenance": {
                    **(item.get("provenance") or {}),
                    "original_audio": str(original),
                    "original_sha256": digest,
                    "source_url": item.get("source_url"),
                    "license": item.get("license"),
                    "consent": item.get("consent"),
                    "conversion": "ffmpeg 16000 Hz mono PCM16",
                },
            })
        train, val = split_groups(converted, fraction, seed)
        check_leakage(train, val)
        _jsonl(output / "train.jsonl", train)
        _jsonl(output / "val.jsonl", val)
        report = {"schema_version": 1, "train_clips": len(train), "val_clips": len(val),
                  "train_seconds": sum(r["duration_seconds"] for r in train),
                  "val_seconds": sum(r["duration_seconds"] for r in val),
                  "seed": seed, "val_fraction": fraction, "source_index": str(index)}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    except BaseException:
        # Avoid silently treating a partial output as a valid complete corpus.
        (output / "INCOMPLETE").write_text("Preparation failed; do not train from this directory.\n")
        raise


def preflight(train_source, val_source, speaker_disjoint=False):
    train, val = read_records(train_source), read_records(val_source)
    check_leakage(train, val, speaker_disjoint)
    return {"train": len(train), "val": len(val),
            "train_hours": sum(r["duration_seconds"] for r in train)/3600,
            "val_hours": sum(r["duration_seconds"] for r in val)/3600,
            "prompt_groups": len({prompt_key(r["transcript"]) for r in train+val})}

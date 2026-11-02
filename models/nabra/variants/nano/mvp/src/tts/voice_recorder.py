"""Local, consent-based, resumable voice collection adapted from jsm-voice.

Record source audio separately from the 16 kHz Nabra training corpus.
Never upload captured speech or overwrite an accepted take.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import wave

import numpy as np

from ..paths import BERHOT_ROOT

SAMPLE_RATE = 22050
CORPUS_ROOT = BERHOT_ROOT / "storage/voice_corpus/nabra"


def prompts_from_file(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("Prompt list must be nonempty, with unique sentences")
    return prompts


def check_take(samples):
    samples = np.asarray(samples)
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError("Expected mono int16 microphone samples")
    if len(samples) < SAMPLE_RATE // 2:
        return "too short"
    if len(samples) > SAMPLE_RATE * 20:
        return "too long (Nabra maximum is 20 seconds)"
    peak = int(np.abs(samples.astype(np.int32)).max())
    if peak >= 32700:
        return "clipped"
    if peak < 2000:
        return "too quiet"
    return None


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_speaker(speaker):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", speaker):
        raise ValueError("Speaker ID must contain 1-64 letters, digits, _ or -")
    return speaker


def completed_rows(folder):
    metadata = folder / "metadata.jsonl"
    result = []
    if not metadata.exists():
        return result
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relative = Path(row["audio"])
        path = (folder / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError("Invalid recorded WAV path")
        if _digest(path) != row["audio_sha256"]:
            raise ValueError("Previously recorded WAV has changed: " + str(path))
        result.append(row)
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate recording identifiers")
    return result


def save_take(folder, number, prompt, samples):
    reason = check_take(samples)
    if reason:
        raise ValueError(reason)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wavs").mkdir(exist_ok=True)
    identifier = f"{number:05d}"
    if any(row["id"] == identifier for row in completed_rows(folder)):
        raise FileExistsError("Already accepted recording " + identifier)
    wav = folder / "wavs" / (identifier + ".wav")
    with wav.open("xb") as stream:
        with wave.open(stream, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    row = {"id": identifier, "text": prompt, "audio": "wavs/" + identifier + ".wav",
           "audio_sha256": _digest(wav), "sample_rate": SAMPLE_RATE,
           "created_at": datetime.now(timezone.utc).isoformat()}
    with (folder / "metadata.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def export_index(folder, output, speaker):
    """Index originals for Nabra's immutable FFmpeg prepare_corpus command."""
    folder = Path(folder).resolve()
    output = Path(output).resolve()
    rows = completed_rows(folder)
    if not rows:
        raise ValueError("Nothing recorded yet")
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((folder / "speaker.json").read_text(encoding="utf-8"))
    if not receipt.get("recording_consent", {}).get("granted"):
        raise ValueError("No consent receipt")
    if receipt.get("speaker") != speaker:
        raise ValueError("Speaker receipt mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps({
                "audio": str((folder / row["audio"]).resolve()),
                "audio_sha256": row["audio_sha256"],
                "transcript": row["text"], "speaker_id": speaker,
                "consent": receipt["recording_consent"],
                "license": "recorded_by_consenting_speaker",
                "provenance": {"source": "local_microphone", "recorded_at": row["created_at"],
                               "source_sample_rate": SAMPLE_RATE, "source_id": row["id"]},
            }, ensure_ascii=False) + "\n")
    return len(rows)


def start_session(speaker, prompts_path, consent):
    validate_speaker(speaker)
    if not consent:
        raise ValueError("Explicit --consent required")
    prompts_path = Path(prompts_path).resolve(strict=True)
    prompts = prompts_from_file(prompts_path)
    folder = CORPUS_ROOT / speaker
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "speaker.json"
    prompt_hash = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("speaker") != speaker or receipt.get("prompt_sha256") != prompt_hash or not receipt.get("recording_consent", {}).get("granted"):
            raise ValueError("Existing session speaker/consent/prompt list differs; use another speaker ID or restore original prompts")
    else:
        receipt = {"schema_version": 1, "speaker": speaker, "sample_rate_hz": SAMPLE_RATE,
                   "prompt_sha256": prompt_hash,
                   "recording_consent": {"granted": True, "asserted_at": datetime.now(timezone.utc).isoformat()},
                   "purpose": "Train Berhot Nabra's speech synthesis model",
                   "provenance": {"source_type": "local_microphone", "prompts": str(prompts_path)}}
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
    existing = completed_rows(folder)
    by_id = {r["id"]: r for r in existing}
    for number, text in enumerate(prompts, 1):
        if f"{number:05d}" in by_id and by_id[f"{number:05d}"]["text"] != text:
            raise ValueError("Completed prompt differs from current prompt list")
    return folder, prompts, by_id


def microphone_session(device=None):
    """Keep stream open across takes: avoids repeated macOS CoreAudio startup."""
    import sounddevice as sd
    class Session:
        def __enter__(self):
            self.blocks = None
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device, callback=self.callback)
            self.stream.start()
            return self
        def callback(self, data, frames, times, status):
            if self.blocks is not None:
                self.blocks.append(data[:, 0].copy())
        def take(self):
            self.blocks = []
            input("  Recording... press Enter when finished: ")
            blocks, self.blocks = self.blocks, None
            return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int16)
        def __exit__(self, *args):
            # Preserve older macOS workaround for a blocking CoreAudio close.
            import threading
            thread = threading.Thread(target=self.stream.close, daemon=True)
            thread.start()
            thread.join(2)
            if thread.is_alive():
                print("Warning: microphone stream close timed out; recordings were saved.")
    return Session()

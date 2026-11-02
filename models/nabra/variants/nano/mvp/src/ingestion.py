
import hashlib
import json
import wave

from pathlib import Path


SAMPLE_RATE = 16000
MIN_DURATION = 0.25
MAX_DURATION = 20.0


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def inspect_recording(wav_path, txt_path):
    wav_path = Path(wav_path).resolve()
    txt_path = Path(txt_path).resolve()

    if not txt_path.is_file():
        raise FileNotFoundError(
            f"Missing transcript: {txt_path}"
        )

    transcript = txt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not transcript:
        raise ValueError(
            f"Empty transcript: {txt_path}"
        )

    with wave.open(str(wav_path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frames = audio.getnframes()
        compression = audio.getcomptype()

    if channels != 1:
        raise ValueError(
            f"Expected mono audio: {wav_path}"
        )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16000 Hz: {wav_path}"
        )

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM: {wav_path}"
        )

    if compression != "NONE":
        raise ValueError(
            f"Unsupported WAV compression: {wav_path}"
        )

    duration = frames / sample_rate

    if not MIN_DURATION <= duration <= MAX_DURATION:
        raise ValueError(
            f"Invalid duration {duration:.2f}s: {wav_path}"
        )

    return {
        "audio": str(wav_path),
        "transcript": transcript,
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 4),
        "audio_sha256": sha256_file(wav_path),
        "transcript_sha256": sha256_file(txt_path),
    }


def inspect_split(directory):
    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise FileNotFoundError(directory)

    recordings = sorted(
        directory.rglob("*.wav")
    )

    if not recordings:
        raise ValueError(
            f"No WAV recordings found: {directory}"
        )

    records = []
    seen = set()

    for wav_path in recordings:
        txt_path = wav_path.with_suffix(".txt")

        record = inspect_recording(
            wav_path,
            txt_path,
        )

        audio_hash = record["audio_sha256"]

        if audio_hash in seen:
            raise ValueError(
                f"Duplicate audio recording: {wav_path}"
            )

        seen.add(audio_hash)
        records.append(record)

    transcript_files = set(
        directory.rglob("*.txt")
    )

    expected_files = {
        Path(record["audio"]).with_suffix(".txt")
        for record in records
    }

    unmatched = transcript_files - expected_files

    if unmatched:
        raise ValueError(
            f"Unmatched transcripts: {sorted(unmatched)}"
        )

    return records


def write_manifest(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n"
            )

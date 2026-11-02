import unittest
import json
import math
import tempfile
import wave
from pathlib import Path

from src.tts.multisentence import readiness
from src.tts.multisentence import sample_index


class MultiSentenceTests(unittest.TestCase):
    def test_epoch_covers_each_recording_once(self):
        for count in (1, 2, 7):
            for epoch in range(3):
                ids = [sample_index(epoch * count + position + 1, count, 42)
                       for position in range(count)]
                self.assertEqual(sorted(ids), list(range(count)))

    def test_repeatable_after_resume(self):
        first = [sample_index(step, 9, 2026) for step in range(1, 31)]
        later = [sample_index(step, 9, 2026) for step in range(11, 31)]
        self.assertEqual(first[10:], later)

    def test_invalid_args(self):
        for step, count in ((0, 3), (1, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                sample_index(step, count, 42)


    def test_readiness_with_real_wav_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, val = [], []
            for idx, text in enumerate(("مرحبا", "كيف حالك", "صباح النور"), start=1):
                wav = root / f"clip_{idx}.wav"
                # Distinct audible PCM16 waveforms, not silent/duplicate fixtures.
                samples = bytearray()
                for n in range(8000):
                    sample = int(2500 * math.sin(2 * math.pi * (180 + idx * 40) * n / 16000))
                    samples.extend(sample.to_bytes(2, "little", signed=True))
                with wave.open(str(wav), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(samples)
                (train if idx < 3 else val).append({
                    "audio": wav.name, "transcript": text, "speaker_id": "speaker-1"
                })
            for name, rows in (("train", train), ("val", val)):
                with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for record in rows:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            report = readiness(root / "train.jsonl", root / "val.jsonl",
                               min_train=2, min_distinct=2)
            self.assertEqual(report["readiness"], "ready-for-experimental-training")
            self.assertEqual(report["distinct_train_prompts"], 2)
            self.assertEqual(report["validation_recordings"], 1)
            self.assertFalse(report["quality_verified"])
            # Unseen validation speaker is incompatible with a learned embedding.
            val[0]["speaker_id"] = "other-speaker"
            (root / "val.jsonl").write_text(
                json.dumps(val[0], ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "speakers absent"):
                readiness(root / "train.jsonl", root / "val.jsonl", min_train=2, min_distinct=2)

if __name__ == "__main__":
    unittest.main()

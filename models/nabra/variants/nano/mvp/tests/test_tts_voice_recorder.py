import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.tts.voice_recorder import check_take, completed_rows, export_index, prompts_from_file, save_take


class VoiceRecorderTests(unittest.TestCase):
    def test_prompt_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.txt"
            path.write_text("# comment\nمرحبا\n\nصباح الخير\n", encoding="utf8")
            self.assertEqual(prompts_from_file(path), ["مرحبا", "صباح الخير"])

    def test_quality_and_preserved_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(check_take(np.zeros(22050, dtype=np.int16)), "too quiet")
            samples = np.full(22050, 3000, dtype=np.int16)
            row = save_take(folder, 1, "مرحبا", samples)
            self.assertEqual(completed_rows(folder)[0]["audio_sha256"], row["audio_sha256"])
            with self.assertRaises(FileExistsError):
                save_take(folder, 1, "مرحبا", samples)
            self.assertTrue((folder / "wavs/00001.wav").exists())

    def test_export_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_take(folder, 1, "مرحبا", np.full(22050, 3000, dtype=np.int16))
            (folder / "speaker.json").write_text(json.dumps({
                "speaker": "fahad", "recording_consent": {"granted": True},
            }), encoding="utf8")
            output = folder / "index.jsonl"
            self.assertEqual(export_index(folder, output, "fahad"), 1)
            row = json.loads(output.read_text(encoding="utf8").strip())
            self.assertEqual(row["speaker_id"], "fahad")
            self.assertEqual(row["transcript"], "مرحبا")
            with self.assertRaises(FileExistsError):
                export_index(folder, output, "fahad")


if __name__ == "__main__":
    unittest.main()

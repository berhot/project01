import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.tts.unseen_evaluation import assess_waveform, read_prompts


class UnseenEvaluationTests(unittest.TestCase):
    def test_waveform_checks(self):
        values = assess_waveform(torch.tensor([0.0, 0.5, -1.0]))
        self.assertEqual(values["clip_fraction"], 1 / 3)
        self.assertEqual(values["duration_seconds"], 3 / 16000)
        with self.assertRaises(ValueError):
            assess_waveform(torch.tensor([float("nan")]))

    def test_prompt_validation(self):
        self.assertGreaterEqual(len(read_prompts(None)), 5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompts.jsonl"
            path.write_text(json.dumps({"id": "one", "text": "مرحبا"}) + "\n" +
                            json.dumps({"id": "two", "text": "كيف حالك"}) + "\n", encoding="utf8")
            self.assertEqual(len(read_prompts(path)), 2)
            path.write_text(json.dumps({"id": "one", "text": "مرحبا"}) + "\n" +
                            json.dumps({"id": "one", "text": "كيف حالك"}) + "\n", encoding="utf8")
            with self.assertRaises(ValueError):
                read_prompts(path)


if __name__ == "__main__":
    unittest.main()

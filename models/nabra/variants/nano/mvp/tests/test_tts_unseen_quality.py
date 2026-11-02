import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.tts.unseen_quality import prompts_from_file, signal_metrics, evaluate_unseen


class UnseenQualityTests(unittest.TestCase):
    def test_prompt_parser_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "prompts.txt"
            path.write_text("# comment\nأهلا   بك\n\nمرحبا\n", encoding="utf-8")
            self.assertEqual(prompts_from_file(path), ["أهلا بك", "مرحبا"])
            path.write_text("أهلا بك\n  أهلا   بك \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prompts_from_file(path)

    def test_signal_metrics(self):
        metrics = signal_metrics(torch.zeros(100))
        self.assertTrue(metrics["silent"])
        self.assertFalse(metrics["clipped"])
        self.assertTrue(signal_metrics(torch.ones(100))["clipped"])
        with self.assertRaises(ValueError):
            signal_metrics(torch.tensor([float("nan")]))

    def test_holdout_overlap_rejected_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            prompts = p / "prompts.txt"
            prompts.write_text("صباح الخير\n", encoding="utf-8")
            manifest = p / "train.jsonl"
            manifest.write_text(json.dumps({"transcript": "صباح  الخير"}, ensure_ascii=False) + "\n",
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlaps training"):
                evaluate_unseen(p / "nonexistent.pt", prompts, p / "new",
                                train_sources=[manifest])
            self.assertFalse((p / "new").exists())

    def test_refuse_existing_output_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            prompts = p / "prompts.txt"
            prompts.write_text("مرحبا\n", encoding="utf-8")
            out = p / "existing"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                evaluate_unseen(p / "missing.pt", prompts, out)

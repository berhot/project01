import tempfile
import unittest
from pathlib import Path

import torch

from src.tts.component_diagnostic import attention_summary, stop_summary, inspect_components


class ComponentDiagnosticTests(unittest.TestCase):
    def test_attention_summary(self):
        attn = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        result = attention_summary(attn)
        self.assertEqual(result["unique_peak_positions"], 3)
        self.assertEqual(result["last_peak"], 2)

    def test_stop_summary(self):
        result = stop_summary(torch.tensor([[-4., 0., 4.]]), 3)
        self.assertGreater(result["last_probability"], 0.98)
        self.assertLess(result["first_probability"], 0.02)

    def test_rejects_invalid_arguments_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "report"
            with self.assertRaises(ValueError):
                inspect_components("missing.pt", out, count=0)
            self.assertFalse(out.exists())

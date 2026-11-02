import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tts.vocoder_comparison import compare_vocoders


class VocoderComparisonTests(unittest.TestCase):
    def test_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                compare_vocoders("missing.pt", "missing.pt", directory)

    def test_rejects_invalid_index_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            with self.assertRaises(ValueError):
                compare_vocoders("missing.pt", "missing.pt", out, record_index=-1)
            self.assertFalse(out.exists())

    def test_rejects_nonstandard_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            with patch("src.tts.vocoder_comparison.torch.load", return_value={"task": "other"}):
                with self.assertRaisesRegex(ValueError, "standard TTS"):
                    compare_vocoders("fake.pt", "fake.pt", out)
            self.assertFalse(out.exists())

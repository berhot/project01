import tempfile
import unittest
from pathlib import Path

import torch

from src.model import CAPTIONS, Config, Denoiser, caption_id, synthetic_image, training_loss, sample, save_png, schedule


class ImageNanoTests(unittest.TestCase):
    def test_english_arabic_caption_alias(self):
        self.assertEqual(caption_id("RED   CIRCLE"), 0)
        self.assertEqual(caption_id("دائرة حمراء"), 0)
        self.assertEqual(caption_id("مربع أزرق"), 5)
        with self.assertRaises(ValueError):
            caption_id("realistic Saudi cafe")

    def test_synthetic_dataset_determinism(self):
        x = synthetic_image(0, 7)
        self.assertEqual(tuple(x.shape), (3, 32, 32))
        self.assertTrue(torch.equal(x, synthetic_image(0, 7)))
        self.assertFalse(torch.equal(x, synthetic_image(1, 7)))

    def test_train_forward_backward(self):
        torch.manual_seed(1)
        model = Denoiser(Config(channels=8, diffusion_steps=4))
        images = torch.stack([synthetic_image(0, 5), synthetic_image(5, 6)])
        labels = torch.tensor([0, 5])
        loss = training_loss(model, images, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_schedule_reaches_noise(self):
        _, _, cumulative = schedule(Config(diffusion_steps=50), "cpu")
        self.assertLess(float(cumulative[-1]), 0.001)
        self.assertGreater(float(cumulative[0]), 0.9)

    def test_checkpoint_roundtrip_and_infer(self):
        torch.manual_seed(1)
        model = Denoiser(Config(channels=8, diffusion_steps=3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            loaded = Denoiser(Config(channels=8, diffusion_steps=3))
            loaded.load_state_dict(torch.load(path, weights_only=True))
            a = sample(model, 2, seed=7)
            b = sample(loaded, 2, seed=7)
            self.assertTrue(torch.equal(a, b))
            out = Path(directory) / "sample.png"
            save_png(a, out)
            self.assertTrue(out.is_file())
            with self.assertRaises(FileExistsError):
                save_png(a, out)

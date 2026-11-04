import tempfile
import unittest
from pathlib import Path

import torch

from src.cvae import CVAEConfig, ShapeCVAE, batch_from_synthetic, cvae_loss, generate
from scripts.cvae import evaluate, restore


class ShapeCVAETests(unittest.TestCase):
    def test_training_batch_and_gradient(self):
        torch.manual_seed(3)
        labels = torch.tensor([0, 3, 5])
        images = batch_from_synthetic(labels, step=2)
        self.assertEqual(tuple(images.shape), (3, 3, 32, 32))
        self.assertGreaterEqual(float(images.min()), 0)
        self.assertLessEqual(float(images.max()), 1)
        model = ShapeCVAE(CVAEConfig(channels=8, latent_dim=4))
        prediction, mean, logvar = model(images, labels)
        loss, recon, kl = cvae_loss(prediction, images, mean, logvar)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(recon), 0)
        self.assertGreaterEqual(float(kl), 0)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_checkpoint_roundtrip_and_six_image_export(self):
        torch.manual_seed(11)
        config = CVAEConfig(channels=8, latent_dim=4)
        model = ShapeCVAE(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cvae.pt"
            torch.save({
                "schema_version": 1, "architecture": "shape-cvae-v1",
                "config": vars(config), "captions": [
                    "red circle", "green circle", "blue circle",
                    "red square", "green square", "blue square"
                ], "model": model.state_dict(),
            }, checkpoint)
            loaded = restore(checkpoint)
            self.assertTrue(torch.equal(generate(model, 2, 42), generate(loaded, 2, 42)))
            report_dir = Path(directory) / "preview"
            output = evaluate(loaded, report_dir, seed=42, scale=1)
            self.assertTrue(output.exists())
            self.assertEqual(len(list(report_dir.glob("*.png"))), 7)
            self.assertTrue((report_dir / "report.json").exists())
            with self.assertRaises(FileExistsError):
                evaluate(loaded, report_dir)

    def test_old_ddpm_checkpoint_cannot_load_as_cvae(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"schema_version": 1, "task": "caption-conditioned-ddpm-demo"}, path)
            with self.assertRaises(ValueError):
                restore(path)

    def test_bad_config_and_label(self):
        with self.assertRaises(ValueError):
            CVAEConfig(beta=0)
        with self.assertRaises(ValueError):
            generate(ShapeCVAE(CVAEConfig(channels=8)), -1)

import tempfile
import unittest
from pathlib import Path

import torch

from src.spatial_field import (
    SpatialConfig, SpatialField, layout_for_seed, synthetic_batch,
    spatial_loss, render, save_field_png,
)
from scripts.spatial_field import restore, evaluate


class SpatialFieldTests(unittest.TestCase):
    def test_shapes_and_gradients(self):
        torch.manual_seed(4)
        model = SpatialField(SpatialConfig(hidden=8, embedding_dim=4))
        labels = torch.tensor([0, 3, 5])
        pixels, masks, layouts = synthetic_batch(labels, step=1)
        self.assertEqual(tuple(pixels.shape), (3, 32, 32, 3))
        self.assertEqual(tuple(masks.shape), (3, 32, 32))
        logit, rgb = model(labels, layouts)
        loss, seg, col = spatial_loss(logit, rgb, pixels, masks)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(seg.detach()), 0)
        self.assertGreater(float(col.detach()), 0)
        loss.backward()
        self.assertIsNotNone(model.field[0].weight.grad)
        self.assertGreater(float(model.field[0].weight.grad.abs().sum()), 0)

    def test_layout_sampling_is_reproducible(self):
        self.assertEqual(layout_for_seed(123), layout_for_seed(123))
        self.assertNotEqual(layout_for_seed(123), layout_for_seed(124))

    def test_checkpoint_and_six_images(self):
        config = SpatialConfig(hidden=8, embedding_dim=4)
        model = SpatialField(config)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save({
                "schema_version": 1, "architecture": "spatial-field-v1",
                "config": vars(config), "captions": [
                    "red circle", "green circle", "blue circle",
                    "red square", "green square", "blue square",
                ], "model": model.state_dict(),
            }, path)
            restored = restore(path)
            self.assertTrue(torch.equal(render(model, 0, 42)[0], render(restored, 0, 42)[0]))
            out = evaluate(restored, Path(d) / "eval", seed=42)
            self.assertTrue(out.exists())
            self.assertEqual(len(list((Path(d) / "eval").glob("*.png"))), 7)
            with self.assertRaises(FileExistsError):
                evaluate(restored, Path(d) / "eval", seed=42)

    def test_reject_wrong_checkpoint_and_inputs(self):
        model = SpatialField(SpatialConfig(hidden=8))
        with self.assertRaises(ValueError):
            render(model, 6)
        with self.assertRaises(ValueError):
            model(torch.tensor([0]), torch.tensor([[1., 2., 0.]]))
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.pt"
            torch.save({"schema_version": 1, "architecture": "shape-cvae-v1"}, bad)
            with self.assertRaises(ValueError):
                restore(bad)

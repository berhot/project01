import random
import tempfile
import unittest
from pathlib import Path

import torch

from src.layout_diffusion import (
    LayoutConfig, LayoutDenoiser, encode_layout, decode_layout,
    training_loss, sample_layout,
)
from src.spatial_field import SpatialConfig, SpatialField
from scripts.layout_diffusion import (
    CAPTIONS, restore_prior, restore_renderer, training_layouts, evaluate,
)


class LayoutDiffusionTests(unittest.TestCase):
    def test_reversible_geometry_encoding(self):
        layouts = training_layouts(16, random.Random(42))
        self.assertTrue(torch.allclose(
            layouts, decode_layout(encode_layout(layouts)), atol=1e-5
        ))
        with self.assertRaises(ValueError):
            encode_layout(torch.tensor([[100., 0., 5.]]))

    def test_training_gradient_and_seeded_layout_sampling(self):
        torch.manual_seed(11)
        model = LayoutDenoiser(LayoutConfig(steps=4, hidden=16))
        labels = torch.tensor([0, 1, 4, 5])
        layouts = training_layouts(len(labels), random.Random(3))
        loss = training_loss(model, labels, layouts)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        a = sample_layout(model, labels, seed=42)
        self.assertTrue(torch.equal(a, sample_layout(model, labels, seed=42)))
        self.assertFalse(torch.equal(a, sample_layout(model, labels, seed=123)))
        self.assertTrue(torch.all(a[:, 2] >= 6))
        self.assertTrue(torch.all(a[:, 2] <= 10.5))

    def test_two_checkpoint_roundtrip_and_preview(self):
        prior = LayoutDenoiser(LayoutConfig(steps=4, hidden=16))
        renderer = SpatialField(SpatialConfig(hidden=8, embedding_dim=4))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            p, s = base / "prior.pt", base / "renderer.pt"
            torch.save({
                "schema_version": 1, "architecture": "layout-ddpm-v1",
                "config": vars(prior.config), "captions": list(CAPTIONS),
                "model": prior.state_dict(),
            }, p)
            torch.save({
                "schema_version": 1, "architecture": "spatial-field-v1",
                "config": vars(renderer.config), "captions": list(CAPTIONS),
                "model": renderer.state_dict(),
            }, s)
            loaded_prior = restore_prior(p)
            loaded_renderer = restore_renderer(s)
            labels = torch.arange(6)
            self.assertTrue(torch.equal(
                sample_layout(loaded_prior, labels, 42),
                sample_layout(prior, labels, 42)
            ))
            export = base / "eval"
            grid = evaluate(loaded_prior, loaded_renderer, export, 42, scale=1)
            self.assertTrue(grid.is_file())
            self.assertEqual(len(list(export.glob("*.png"))), 7)
            self.assertTrue((export / "report.json").is_file())
            with self.assertRaises(FileExistsError):
                evaluate(loaded_prior, loaded_renderer, export, seed=123)

    def test_reject_incompatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "wrong.pt"
            torch.save({
                "schema_version": 1, "architecture": "shape-cvae-v1"
            }, bad)
            with self.assertRaises(ValueError):
                restore_prior(bad)
            with self.assertRaises(ValueError):
                restore_renderer(bad)

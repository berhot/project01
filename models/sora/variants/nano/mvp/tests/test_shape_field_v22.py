"""V2.2 opt-in loss and immutable-checkpoint regression coverage."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.shape_field_v2 import NeuralMaskField, ShapeFieldConfig
from src.shape_field_v21 import matched_layout_shape_batch
from src.shape_field_v22 import boundary_centroid_objective
from scripts.shape_field_v2 import train as train_original, load_run
from scripts.shape_field_v22 import train as train_v22


class ShapeFieldV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_edge_centroid_loss_is_differentiable_and_circle_excluded(self):
        model = NeuralMaskField(ShapeFieldConfig(hidden=16, word_dim=8))
        layouts, targets, tokens, _ = matched_layout_shape_batch(1, 1)
        logits = model(layouts, tokens)
        result = boundary_centroid_objective(logits, targets, tokens)
        self.assertEqual(result["supervised_noncircles"], 4)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreaterEqual(float(result["edge_loss"]), 0)
        self.assertGreaterEqual(float(result["centroid_loss"]), 0)
        result["loss"].backward()
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()), 0)
        circle_only = boundary_centroid_objective(
            logits[:2], targets[:2], tokens[:2])
        self.assertEqual(circle_only["supervised_noncircles"], 0)
        self.assertEqual(float(circle_only["loss"]), 0)

    def test_rejects_invalid_training_loss_inputs(self):
        layout, targets, tokens, _ = matched_layout_shape_batch(1, 1)
        logits = torch.zeros_like(targets)
        with self.assertRaises(ValueError):
            boundary_centroid_objective(logits, targets, tokens,
                                        boundary_weight=float("nan"))
        with self.assertRaises(ValueError):
            boundary_centroid_objective(logits, targets[:, :, :31], tokens)

    def test_v22_warmstarts_and_never_mutates_v21(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = SimpleNamespace(
                mask_steps=2, color_steps=2, batch_size=12,
                hidden=16, word_dim=8, learning_rate=0.0008,
                seed=42, device="cpu", cpu_threads=1,
                log_every=2, output_root=str(root/"v21"),
                shape_v21=True, pair_weight=1.0,
                rank_weight=0.1, rank_margin=0.03,
                paired_layout_count=1)
            previous = train_original(recipe)
            source_files = [previous/x for x in
                            ("mask_field.pt", "colorizer.pt", "run.json")]
            before = {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in source_files
            }
            trial = SimpleNamespace(
                init_from_v21=str(previous), steps=2, batch_size=12,
                paired_layout_count=1, learning_rate=0.0003,
                boundary_weight=0.5, centroid_weight=0.05,
                circle_preserve_weight=0.5, seed=42, device="cpu",
                cpu_threads=1, log_every=2, output_root=str(root/"v22"))
            output = train_v22(trial)
            self.assertTrue(output.name.startswith("shape-field-v22-"))
            manifest = json.loads((output/"run.json").read_text())
            self.assertEqual(manifest["training_recipe"],
                             "shape-boundary-centroid-v2.2")
            self.assertEqual(manifest["release_status"], "not-approved")
            self.assertEqual(manifest["warm_start"]["source_run_dir"], str(previous))
            mask_model, color_model, _ = load_run(output)
            self.assertIsNotNone(mask_model)
            self.assertIsNotNone(color_model)
            self.assertEqual((previous/"colorizer.pt").read_bytes(),
                             (output/"colorizer.pt").read_bytes())
            self.assertEqual(before, {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in source_files
            })


if __name__ == "__main__":
    unittest.main()

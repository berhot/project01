"""Opt-in learned radius field contract; no changes to earlier models."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.shape_field_v31 import (
    RadiusMaskField, RadiusConfig, training_mask, training_batch,
    radius_objective
)
from scripts.shape_field_v31 import train, load, evaluate


class ShapeFieldV31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_synthetic_radius_targets_are_distinct(self):
        target_sharp, layout = training_mask(71, 0)
        target_round, other_layout = training_mask(71, 1)
        self.assertTrue(torch.equal(layout, other_layout))
        self.assertFalse(torch.equal(target_sharp, target_round))
        self.assertEqual(target_sharp.shape, (1,32,32))

    def test_learned_radius_condition_and_backward(self):
        torch.manual_seed(42)
        model=RadiusMaskField(RadiusConfig(hidden=16))
        layouts,radii,truth=training_batch(1,10)
        out=model(layouts,radii)
        self.assertEqual(tuple(out.shape),(10,1,32,32))
        objective=radius_objective(model,layouts,radii,truth)
        self.assertTrue(torch.isfinite(objective["loss"]))
        objective["loss"].backward()
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()),0)
        different=model(layouts,torch.ones_like(radii))
        self.assertFalse(torch.equal(out,different))

    def test_rejects_bad_radius_or_layout(self):
        model=RadiusMaskField(RadiusConfig(hidden=16))
        with self.assertRaises(ValueError):
            training_mask(1,1.5)
        with self.assertRaises(ValueError):
            model(torch.zeros(1,4),torch.tensor([[0.5]]))
        with self.assertRaises(ValueError):
            model(torch.ones(1,4),torch.tensor([[1.5]]))

    def test_separate_experimental_checkpoint_and_heldout_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(steps=2,batch_size=10,hidden=16,
                learning_rate=0.001,seed=42,log_every=1,device="cpu",
                cpu_threads=1,output_root=str(root/"runs"))
            run=train(args)
            self.assertTrue(run.name.startswith("shape-field-v31-radius-"))
            files=[run/"run.json",run/"radius_field.pt"]
            before=[hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
            model=load(run)
            self.assertIsNotNone(model)
            result=evaluate(run,output_dir=str(root/"eval"))
            self.assertEqual(len(result["items"]),40)
            self.assertFalse(result["release_approved"])
            self.assertEqual(before,[hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in files])


if __name__ == "__main__":
    unittest.main()

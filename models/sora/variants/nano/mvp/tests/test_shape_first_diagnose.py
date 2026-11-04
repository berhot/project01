"""Tests for stage-specific research diagnosis on already trained models."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.pixel_dataset_v2 import synthetic_example_v2
from src.shape_first import mask_from_rgb
from scripts.shape_first_diagnose import score_binary_mask,diagnose
from scripts.shape_first import train


class ShapeFirstDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_geometry_scorer_accepts_real_heldout_shapes(self):
        for shape in ("circle","square","triangle"):
            image,_,_,_=synthetic_example_v2(
                64,color="red",shape=shape,
                language="ar",split="validation",variant=0
            )
            mask=mask_from_rgb(image[None])[0]
            scored=score_binary_mask(mask,shape)
            self.assertTrue(scored["strict_geometry_pass"],scored)
            self.assertGreater(scored["expected_mask_iou"],0.95)

    def test_scorer_rejects_squares_as_triangles_and_empty_masks(self):
        square,_,_,_=synthetic_example_v2(
            53,color="yellow",shape="square",split="validation",variant=0)
        mask=mask_from_rgb(square[None])[0]
        wrong=score_binary_mask(mask,"triangle")
        self.assertFalse(wrong["strict_geometry_pass"])
        self.assertLess(wrong["expected_mask_iou"],0.80)
        empty=score_binary_mask(torch.zeros(1,32,32),"circle")
        self.assertEqual(empty["foreground_pixels"],0)
        self.assertFalse(empty["strict_geometry_pass"])
        with self.assertRaises(ValueError):
            score_binary_mask(torch.zeros(3,32,32),"square")
        with self.assertRaises(ValueError):
            score_binary_mask(torch.zeros(1,32,32),"octagon")

    def test_existing_checkpoint_readonly_diagnostic_breakdown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=SimpleNamespace(
                mask_steps=2,rgb_steps=2,batch_size=12,
                mask_width=16,rgb_width=16,text_dim=24,latent_dim=4,
                learning_rate=0.0005,beta=0.01,paired_weight=0.5,
                paired_layout_count=1,seed=42,cpu_threads=1,log_every=1,
                device="cpu",output_root=str(root/"runs"),
            )
            run=train(args)
            original={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (run/"mask.pt",run/"appearance.pt",run/"run.json")}
            result=diagnose(run,seed=42)
            self.assertEqual(result["task"],
                             "sora-shape-first-isolated-stage-diagnosis")
            self.assertEqual(len(result["items"]),24)
            self.assertTrue(result["no_retraining"])
            self.assertTrue(result["no_release"])
            for shape in ("circle","square","triangle"):
                metrics=result["summary_by_shape"][shape]
                self.assertEqual(metrics["count"],8)
                self.assertGreaterEqual(
                    metrics["mean_posterior_iou_vs_heldout_gt"],0)
                self.assertLessEqual(
                    metrics["mean_posterior_iou_vs_heldout_gt"],1)
                self.assertGreaterEqual(metrics["prior_strict_shape_passes"],0)
                self.assertLessEqual(metrics["prior_strict_shape_passes"],8)
                self.assertGreaterEqual(metrics["clean_mask_rgb_color_passes"],0)
                self.assertLessEqual(metrics["clean_mask_rgb_color_passes"],8)
            after={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (run/"mask.pt",run/"appearance.pt",run/"run.json")}
            self.assertEqual(original,after)

    def test_bad_model_run_fails_without_fabricating_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                diagnose(Path(temp)/"does_not_exist")

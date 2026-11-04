"""Read-only held-out-vs-sampled shape geometry diagnostics for saved V2 runs."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.shape_field_v2 import train
from scripts.shape_field_geometry_diagnose import _mask_iou,diagnose,compare


class ShapeFieldGeometryDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def arguments(destination):
        return SimpleNamespace(
            mask_steps=2,color_steps=2,batch_size=12,
            hidden=16,word_dim=8,learning_rate=0.0008,
            seed=42,device="cpu",cpu_threads=1,
            log_every=1,output_root=str(destination),
        )

    def test_iou_ground_truth_and_empty_cases(self):
        blank=torch.zeros(1,1,32,32)
        filled=blank.clone()
        filled[:,:,:,4:12]=1
        self.assertEqual(_mask_iou(blank,blank),1.0)
        self.assertEqual(_mask_iou(blank,filled),0.0)
        self.assertEqual(_mask_iou(filled,filled),1.0)

    def test_diagnosis_reports_both_prior_and_heldout_without_model_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=train(self.arguments(root/"runs"))
            originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (run/"mask_field.pt",
                                 run/"colorizer.pt",run/"run.json")}
            report=diagnose(run,seed=42)
            self.assertEqual(report["diagnostic"],
                             "shape-field-heldout-vs-prior-layout")
            self.assertEqual(len(report["items"]),24)
            self.assertTrue(report["no_training"])
            self.assertFalse(report["release_approved"])
            self.assertEqual(set(report["by_shape"]),
                             {"circle","square","triangle"})
            for shape in ("circle","square","triangle"):
                row=report["by_shape"][shape]
                self.assertEqual(row["samples"],8)
                self.assertEqual(set(row["thresholds"]),
                                 {"0.4","0.5","0.6"})
                for score in row["thresholds"].values():
                    self.assertGreaterEqual(
                        score["heldout_iou_vs_truth_mean"],0)
                    self.assertLessEqual(
                        score["heldout_iou_vs_truth_mean"],1)
                    self.assertGreaterEqual(
                        score["sampled_strict_shape_passes"],0)
                    self.assertLessEqual(
                        score["sampled_strict_shape_passes"],8)
            self.assertEqual(originals,{
                p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (run/"mask_field.pt",run/"colorizer.pt",run/"run.json")
            })

    def test_matched_baseline_self_comparison_is_exactly_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=train(self.arguments(Path(tmp)/"runs"))
            old=diagnose(run,seed=42)
            new=diagnose(run,seed=42)
            delta=compare(old,new)
            self.assertFalse(delta["release_approved"])
            for result in delta["comparison_by_shape"].values():
                self.assertEqual(result["heldout_ground_truth_iou_change"],0)
                self.assertEqual(result["heldout_strict_pass_change"],0)
                self.assertEqual(result["sampled_strict_pass_change"],0)
            with self.assertRaisesRegex(ValueError,"matched diagnostic and seeds"):
                compare(old,diagnose(run,seed=43))
            new["items"][0]["validation_seed"]=9999
            with self.assertRaisesRegex(ValueError,"Unmatched held-out"):
                compare(old,new)

    def test_nonexistent_checkpoint_does_not_fake_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                diagnose(Path(tmp)/"missing")

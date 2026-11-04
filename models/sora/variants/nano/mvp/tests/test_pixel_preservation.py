"""Tests for non-destructive old-concept teacher preservation during v2 trials."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.pixel_diffusion import PixelConfig, PixelDenoiser, vocabulary_for
from src.pixel_dataset_v2 import balanced_training_batch
from src.pixel_preservation import old_shape_mask, preservation_objective
from scripts.pixel_diffusion import load_model, train


class PixelPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def new_model(self):
        return PixelDenoiser(PixelConfig(
            channels=16, diffusion_steps=8, dataset_version="v2",
            architecture="rgb-unet-spatial-text-v2",
        ))

    def save_source(self, root):
        model=self.new_model()
        path=Path(root)/"source-spatial-v2.pt"
        torch.save({
            "schema_version":1,"architecture":model.config.architecture,
            "config":vars(model.config),
            "vocabulary":list(vocabulary_for("v2")),
            "cfg_drop_prob":0.15,"training_steps":800,
            "model":model.state_dict(),
        },path)
        return path, model

    def test_selects_old_shapes_in_both_languages(self):
        images,tokens=balanced_training_batch(
            step=1,batch_size=12,seed=42,triangle_fraction=0.5
        )
        selected=old_shape_mask(tokens)
        self.assertEqual(int(selected.sum()),6)
        self.assertEqual(images.shape,(12,3,32,32))
        with self.assertRaises(ValueError):
            old_shape_mask(torch.zeros(5,10,dtype=torch.long))

    def test_preservation_zero_for_same_model_and_gradient_for_changed_student(self):
        torch.manual_seed(24)
        teacher=self.new_model()
        student=self.new_model()
        student.load_state_dict(teacher.state_dict())
        teacher.requires_grad_(False)
        images,tokens=balanced_training_batch(1,12,seed=42,
                                             triangle_fraction=0.5)
        exact=preservation_objective(student,teacher,images,tokens)
        self.assertEqual(exact["protected_examples"],6)
        self.assertLess(float(exact["total_loss"]),1e-10)
        with torch.no_grad():
            student.input.weight.add_(0.05)
        changed=preservation_objective(student,teacher,images,tokens)
        self.assertGreater(float(changed["total_loss"]),0)
        changed["total_loss"].backward()
        self.assertGreater(float(student.input.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        with self.assertRaises(ValueError):
            preservation_objective(student,teacher,images,tokens,
                                   null_weight=float("nan"))

    def test_teacher_config_mismatch_and_no_old_shape_replay_rejected(self):
        teacher=self.new_model()
        student=PixelDenoiser(PixelConfig(
            channels=32,diffusion_steps=8,dataset_version="v2",
            architecture="rgb-unet-spatial-text-v2"))
        images,tokens=balanced_training_batch(1,12,seed=42,
                                             triangle_fraction=0.5)
        with self.assertRaises(ValueError):
            preservation_objective(student,teacher,images,tokens)
        student=self.new_model()
        _,all_triangles=balanced_training_batch(1,12,seed=42,
                                                triangle_fraction=1.0)
        with self.assertRaises(ValueError):
            preservation_objective(student,teacher,images,all_triangles)

    def test_short_training_records_teacher_and_does_not_mutate_source(self):
        torch.manual_seed(21)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,_=self.save_source(root)
            old_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            args=SimpleNamespace(
                steps=2,batch_size=12,learning_rate=0.00002,seed=42,
                channels=16,cfg_drop_prob=0.15,
                dataset_version="v2",diffusion_steps=8,
                device="cpu",cpu_threads=1,log_every=1,
                output_root=str(root/"safe-experiment"),
                init_from_v1=None,init_from_v2=str(source),
                triangle_fraction=0.5,conditioning="spatial",
                silhouette_weight=0.15,
                paired_shape_weight=0.02,paired_shape_pairs=4,
                paired_silhouette_weight=0.5,
                preserve_weight=1.0,preserve_null_weight=0.5,
            )
            checkpoint=train(args)
            self.assertNotEqual(checkpoint,source)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             old_hash)
            record=torch.load(checkpoint,map_location="cpu",weights_only=True)
            self.assertEqual(record["preserve_teacher_checkpoint"],str(source))
            self.assertEqual(record["preserve_weight"],1.0)
            self.assertEqual(record["preserve_null_weight"],0.5)
            self.assertEqual(record["paired_shape_weight"],0.02)
            self.assertEqual(load_model(checkpoint).config.dataset_version,"v2")

    def test_refuses_teacher_preservation_without_matching_v2_source(self):
        args=SimpleNamespace(
            steps=1,batch_size=12,learning_rate=0.00002,seed=42,
            channels=16,cfg_drop_prob=0.15,
            dataset_version="v2",diffusion_steps=8,
            device="cpu",cpu_threads=1,log_every=1,
            output_root="/tmp/not-used",
            init_from_v1=None,init_from_v2=None,
            triangle_fraction=0.5,conditioning="spatial",
            silhouette_weight=0.15,
            paired_shape_weight=0.02,paired_shape_pairs=2,
            paired_silhouette_weight=0.5,
            preserve_weight=1.0,preserve_null_weight=0.5,
        )
        with self.assertRaises(ValueError):
            train(args)
        args.init_from_v2="unused-due-to-early-validation"
        args.triangle_fraction=1.0
        with self.assertRaises(ValueError):
            train(args)

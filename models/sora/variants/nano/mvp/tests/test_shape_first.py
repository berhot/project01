"""End-to-end tests for the NEW learned mask-prior + RGB appearance model."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import src.shape_first as shape_core
from src.shape_first import (
    ShapeFirstConfig,MaskVAE,RGBAppearance,
    mask_from_rgb,mask_training_objective,
    appearance_training_objective,generate_pipeline,
)
from src.pixel_dataset_v2 import balanced_training_batch
from src.pixel_diffusion import tokenize
from scripts.shape_first import (
    same_layout_training_pairs,train,load_pipeline,
    sample,evaluate,
)
from scripts.pixel_shape_quality import audit_gallery


class ShapeFirstModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return ShapeFirstConfig(mask_width=16,rgb_width=16,
                                text_dim=24,latent_dim=4)

    def args(self,root):
        return SimpleNamespace(
            mask_steps=2,rgb_steps=2,batch_size=12,
            mask_width=16,rgb_width=16,text_dim=24,latent_dim=4,
            learning_rate=0.0005,beta=0.01,paired_weight=0.5,
            paired_layout_count=1,seed=42,cpu_threads=1,log_every=1,
            device="cpu",output_root=str(root)
        )

    def test_mask_from_actual_synthetic_rgb_has_distinct_shapes(self):
        images,_=balanced_training_batch(1,12,42)
        masks=mask_from_rgb(images)
        self.assertEqual(masks.shape,(12,1,32,32))
        self.assertTrue(torch.isfinite(masks).all())
        self.assertGreater(int(masks.sum()),100)
        self.assertLess(int(masks.sum()),12*1024)
        with self.assertRaises(ValueError):
            mask_from_rgb(torch.zeros(12,1,32,32))

    def test_same_layout_pairs_share_source_but_have_different_shape_targets(self):
        sources,targets,tokens=same_layout_training_pairs(2,1,42)
        self.assertEqual(sources.shape,(6,1,32,32))
        self.assertEqual(targets.shape,(6,1,32,32))
        self.assertEqual(tokens.shape,(6,12))
        self.assertTrue(torch.equal(sources[0],sources[1]))
        self.assertTrue(torch.equal(sources[0],sources[2]))
        self.assertTrue(torch.equal(targets[0],targets[1]))
        self.assertFalse(torch.equal(targets[0],targets[2]))
        self.assertFalse(torch.equal(targets[2],targets[4]))
        self.assertFalse(torch.equal(tokens[0],tokens[1]))
        with self.assertRaises(ValueError):
            same_layout_training_pairs(0,1,42)

    def test_mask_vae_finite_gradients_and_same_latent_word_supervision(self):
        torch.manual_seed(41)
        cfg=self.config()
        model=MaskVAE(cfg)
        images,tokens=balanced_training_batch(1,12,42)
        paired=same_layout_training_pairs(2,1,42)
        losses=mask_training_objective(
            model,images,tokens,paired=paired,contrast_weight=0.5)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["counterfactual_mask_loss"]),0)
        losses["loss"].backward()
        self.assertGreater(float(model.decoder[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.text.embedding.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            mask_training_objective(model,images,tokens,beta=-1)
        with self.assertRaises(ValueError):
            mask_training_objective(model,images,tokens,
                                    paired=(paired[0],paired[1][:3],paired[2]))

    def test_appearance_network_learns_rgb_from_mask_and_bilingual_text(self):
        torch.manual_seed(12)
        model=RGBAppearance(self.config())
        images,tokens=balanced_training_batch(2,12,42)
        mask=mask_from_rgb(images)
        out=model(mask,tokens)
        self.assertEqual(out.shape,(12,3,32,32))
        self.assertTrue(torch.isfinite(out).all())
        losses=appearance_training_objective(model,images,tokens)
        self.assertGreater(float(losses["loss"]),0)
        losses["loss"].backward()
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            model(mask,torch.zeros(12,11,dtype=torch.long))

    def test_true_inference_is_deterministic_and_never_calls_shape_drawer(self):
        torch.manual_seed(19)
        cfg=self.config()
        mask_model=MaskVAE(cfg)
        rgb_model=RGBAppearance(cfg)
        with patch("src.pixel_dataset_v2.synthetic_example_v2",
                   side_effect=AssertionError("training drawing called at inference")):
            first,hard,soft=generate_pipeline(
                mask_model,rgb_model,"yellow triangle",seed=42)
            same,hard2,soft2=generate_pipeline(
                mask_model,rgb_model,"yellow triangle",seed=42)
            arabic,_,_=generate_pipeline(
                mask_model,rgb_model,"مثلث أصفر",seed=42)
        self.assertTrue(torch.equal(first,same))
        self.assertTrue(torch.equal(hard,hard2))
        self.assertTrue(torch.equal(soft,soft2))
        self.assertEqual(first.shape,(3,32,32))
        self.assertEqual(hard.shape,(1,32,32))
        self.assertTrue(torch.isfinite(arabic).all())
        self.assertFalse(hasattr(shape_core,"ImageDraw"))
        self.assertFalse(hasattr(shape_core,"Image"))
        self.assertFalse(torch.equal(
            tokenize("yellow triangle",dataset_version="v2"),
            tokenize("مثلث أصفر",dataset_version="v2")))
        with self.assertRaises(ValueError):
            generate_pipeline(mask_model,rgb_model,
                              "photorealistic golden cow",seed=42)

    def test_run_roundtrip_24_actual_generations_and_checkpoint_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=train(self.args(root/"new-runs"))
            self.assertTrue((run/"mask.pt").is_file())
            self.assertTrue((run/"appearance.pt").is_file())
            self.assertTrue((run/"run.json").is_file())
            manifest=json.loads((run/"run.json").read_text())
            self.assertEqual(manifest["release_status"],"not-approved")
            self.assertFalse(manifest["procedural_shapes_at_inference"])
            mask_model,rgb_model,restored=load_pipeline(run)
            self.assertEqual(restored["mask_steps"],2)
            self.assertEqual(restored["rgb_steps"],2)
            a=sample(run,"yellow triangle",seed=42,
                     output=root/"samples"/"sample.png")
            b=sample(run,"yellow triangle",seed=42,
                     output=root/"samples"/"again.png")
            self.assertTrue(Path(a["image"]).is_file())
            self.assertTrue(Path(a["learned_mask"]).is_file())
            self.assertEqual(Path(a["image"]).read_bytes(),
                             Path(b["image"]).read_bytes())
            with self.assertRaises(FileExistsError):
                sample(run,"yellow triangle",seed=42,
                       output=root/"samples"/"sample.png")
            report=evaluate(run,output_dir=root/"eval",seed=42)
            self.assertEqual(report["images"],24)
            self.assertEqual(len(report["items"]),24)
            self.assertTrue((root/"eval"/"preview_grid.png").exists())
            self.assertTrue((root/"eval"/"learned_masks_preview_grid.png").exists())
            self.assertEqual(len(list((root/"eval"/"learned_masks").glob("*.png"))),24)
            self.assertTrue(all(
                (root/"eval"/item["learned_mask"]).is_file()
                for item in report["items"]
            ))
            quality=audit_gallery(root/"eval")
            self.assertEqual(quality["sample_count"],24)
            self.assertEqual(sum(x["samples"] for x in quality["by_shape"].values()),24)
            # Reject accidentally mixed models, with no model inference fallback.
            path=run/"mask.pt"
            original=path.read_bytes()
            path.write_bytes(original+b"corrupt")
            with self.assertRaisesRegex(ValueError,"Corrupt"):
                load_pipeline(run)
            path.write_bytes(original)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest["sha256"]["mask.pt"])

    def test_validation_rejects_bad_config_before_creating_artifact_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new-runs"
            for key,value in (
                ("mask_steps",0),("rgb_steps",0),("batch_size",13),
                ("paired_layout_count",0),("learning_rate",float("nan")),
                ("mask_width",1),("beta",-1),("paired_weight",5),
            ):
                args=self.args(root)
                setattr(args,key,value)
                with self.assertRaises(ValueError):
                    train(args)
            self.assertFalse(root.exists())

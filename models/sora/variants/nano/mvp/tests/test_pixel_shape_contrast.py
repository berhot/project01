"""Regression tests for matched shape pairs and opt-in counterfactual training."""
import hashlib
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from src.pixel_diffusion import (
    PixelConfig,PixelDenoiser,vocabulary_for,
)
from src.pixel_shape_contrast import (
    matched_shape_batch,make_counterfactual_noise,
    matched_shape_objective,paired_prompt_probe,
)
from scripts.pixel_diffusion import load_model,train


class ShapeContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self):
        return PixelDenoiser(PixelConfig(
            channels=16,diffusion_steps=8,dataset_version="v2",
            architecture="rgb-unet-spatial-text-v2",
        ))

    def test_matched_examples_share_layout_background_and_differ_in_shape(self):
        a,tokens,meta=matched_shape_batch(6,8,seed=42,split="validation")
        b,twice,_=matched_shape_batch(6,8,seed=42,split="validation")
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(tokens,twice))
        self.assertEqual(a.shape,(8,2,3,32,32))
        self.assertEqual(tokens.shape,(8,2,12))
        shape_words={vocabulary_for("v2").index(x) for x in ("square","triangle","مربع","مثلث")}
        for i in range(8):
            square,triangle=a[i]
            self.assertTrue(torch.equal(square[:,0,0],triangle[:,0,0]))
            # The square has the same color/background/box as the triangle,
            # but the triangle must leave some square corner pixels empty.
            masks=(a[i].amax(dim=1)-a[i].amin(dim=1)>0.45)
            self.assertGreater(int(masks[0].sum()),int(masks[1].sum()))
            self.assertGreater(int((masks[0]!=masks[1]).sum()),12)
            self.assertTrue(any(int(x) in shape_words for x in tokens[i,0]))
            self.assertTrue(any(int(x) in shape_words for x in tokens[i,1]))
        languages=defaultdict(set)
        for step in range(1,9):
            _,_,details=matched_shape_batch(step,4,seed=42)
            for detail in details:
                languages[detail["color"]].add(detail["language"])
        self.assertEqual(set(languages),{"red","green","blue","yellow"})
        self.assertTrue(all(x=={"ar","en"} for x in languages.values()))

    def test_shared_xt_has_two_distinct_mathematically_valid_noise_targets(self):
        images,_,_=matched_shape_batch(3,3,seed=42)
        model=self.make_model()
        g=torch.Generator().manual_seed(17)
        xt,times,eps_target=make_counterfactual_noise(
            model.config,images,generator=g
        )
        self.assertEqual(xt.shape,(3,3,32,32))
        self.assertEqual(eps_target.shape,(3,2,3,32,32))
        self.assertGreater(float((eps_target[:,0]-eps_target[:,1]).abs().mean()),0)
        from src.pixel_diffusion import noise_schedule
        _,_,cum=noise_schedule(model.config,"cpu")
        a=cum[times][:,None,None,None,None]
        reconstruction=(
            xt[:,None]-(1-a).sqrt()*eps_target
        )/a.sqrt()
        self.assertTrue(torch.allclose(reconstruction,images,atol=1e-5))

    def test_shape_pair_loss_backprop_and_probe_is_deterministic(self):
        torch.manual_seed(9)
        model=self.make_model()
        images,tokens,_=matched_shape_batch(2,3,42)
        result=matched_shape_objective(
            model,images,tokens,silhouette_weight=0.5,
            generator=torch.Generator().manual_seed(21)
        )
        self.assertTrue(torch.isfinite(result["total_loss"]))
        self.assertGreater(float(result["pair_shape_loss"]),0)
        result["total_loss"].backward()
        self.assertGreater(float(model.text.token.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.output[-1].weight.grad.abs().sum()),0)
        probe_a=paired_prompt_probe(model,seed=42,pairs=4)
        probe_b=paired_prompt_probe(model,seed=42,pairs=4)
        self.assertEqual(probe_a,probe_b)
        self.assertGreater(probe_a["expected_epsilon_difference_rms"],0)
        self.assertGreaterEqual(probe_a["predicted_epsilon_difference_rms"],0)
        self.assertTrue(math.isfinite(probe_a["shape_difference_alignment"]))
        with self.assertRaises(ValueError):
            matched_shape_objective(model,images,tokens,silhouette_weight=-1)

    def test_new_run_from_spatial_checkpoint_preserves_original(self):
        torch.manual_seed(7)
        model=self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"original-spatial.pt"
            torch.save({
                "schema_version":1,"architecture":model.config.architecture,
                "config":vars(model.config),
                "vocabulary":list(vocabulary_for("v2")),
                "cfg_drop_prob":0.15,"training_steps":800,
                "model":model.state_dict(),
            },source)
            old_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            args=SimpleNamespace(
                steps=2,batch_size=12,learning_rate=0.00005,seed=42,
                channels=16,cfg_drop_prob=0.15,
                dataset_version="v2",diffusion_steps=8,
                device="cpu",cpu_threads=1,log_every=1,
                output_root=str(root/"new-finetune"),
                init_from_v1=None,init_from_v2=str(source),
                triangle_fraction=0.5,conditioning="spatial",
                silhouette_weight=0.15,
                paired_shape_weight=0.35,paired_shape_pairs=4,
                paired_silhouette_weight=0.5,
            )
            checkpoint=train(args)
            self.assertTrue(checkpoint.exists())
            record=torch.load(checkpoint,map_location="cpu",weights_only=True)
            self.assertEqual(record["paired_shape_weight"],0.35)
            self.assertEqual(record["paired_shape_pairs"],4)
            self.assertEqual(record["paired_silhouette_weight"],0.5)
            self.assertEqual(record["warm_start"]["source_checkpoint"],str(source))
            self.assertEqual(record["architecture"],"rgb-unet-spatial-text-v2")
            self.assertEqual(
                old_hash,hashlib.sha256(source.read_bytes()).hexdigest()
            )
            loaded=load_model(checkpoint)
            self.assertEqual(loaded.config.dataset_version,"v2")

    def test_reject_unsupported_pair_training_on_old_architecture(self):
        args=SimpleNamespace(
            steps=1,batch_size=12,learning_rate=0.0001,seed=42,
            channels=16,cfg_drop_prob=0.15,
            dataset_version="v2",diffusion_steps=8,
            device="cpu",cpu_threads=1,log_every=1,
            output_root="/tmp/unused-pixel-contrast-test",
            init_from_v1=None,init_from_v2=None,
            triangle_fraction=0.5,conditioning="pooled",
            silhouette_weight=0.15,
            paired_shape_weight=0.5,paired_shape_pairs=2,
            paired_silhouette_weight=0.5,
        )
        with self.assertRaises(ValueError):
            train(args)

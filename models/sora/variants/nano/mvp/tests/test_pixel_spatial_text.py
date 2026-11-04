"""Experimental trainable spatial text attention and RGB silhouette objective."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, guided_noise,
    shape_aware_training_objective, tokenize, vocabulary_for, generate,
)
from src.pixel_dataset_v2 import synthetic_example_v2
from scripts.pixel_diffusion import (
    initialize_v2_from_v2, load_model, train,
)


class PixelSpatialTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create_pooled_v2_checkpoint(self, root):
        config=PixelConfig(
            dataset_version="v2",architecture="rgb-unet-text-v1",
            channels=16,diffusion_steps=8
        )
        model=PixelDenoiser(config)
        checkpoint=Path(root)/"base-v2.pt"
        torch.save({
            "schema_version":1,"architecture":config.architecture,
            "config":vars(config),"vocabulary":list(vocabulary_for("v2")),
            "cfg_drop_prob":0.15,"training_steps":2500,
            "model":model.state_dict(),
        },checkpoint)
        return checkpoint,model

    def test_spatial_architecture_is_versioned_and_no_v1_modification(self):
        with self.assertRaises(ValueError):
            PixelConfig(dataset_version="v1",architecture="rgb-unet-spatial-text-v2")
        new=PixelDenoiser(PixelConfig(
            dataset_version="v2",architecture="rgb-unet-spatial-text-v2",
            channels=16,diffusion_steps=8))
        old=PixelDenoiser(PixelConfig(
            dataset_version="v2",architecture="rgb-unet-text-v1",
            channels=16,diffusion_steps=8))
        self.assertTrue(any(k.startswith("spatial32.") for k in new.state_dict()))
        self.assertFalse(any(k.startswith("spatial32.") for k in old.state_dict()))

    def test_old_weights_preserved_and_old_forward_is_initially_identical(self):
        with tempfile.TemporaryDirectory() as d:
            source_path,original=self.create_pooled_v2_checkpoint(d)
            new=PixelDenoiser(PixelConfig(
                dataset_version="v2",architecture="rgb-unet-spatial-text-v2",
                channels=16,diffusion_steps=8))
            transfer=initialize_v2_from_v2(new,source_path)
            self.assertEqual(transfer["source_architecture"],"rgb-unet-text-v1")
            self.assertTrue(torch.equal(new.input.weight,original.input.weight))
            original.eval()
            new.eval()
            noisy=torch.randn(2,3,32,32)
            t=torch.tensor([1,6])
            words=torch.stack([
                tokenize("red square",dataset_version="v2"),
                tokenize("مثلث أصفر",dataset_version="v2"),
            ])
            with torch.no_grad():
                old_prediction=original(noisy,t,words)
                new_prediction=new(noisy,t,words)
            self.assertTrue(torch.allclose(old_prediction,new_prediction,atol=1e-6))
            self.assertEqual(new_prediction.shape,(2,3,32,32))

    def test_null_prompts_are_zero_spatial_residual_even_after_bias_training(self):
        config=PixelConfig(
            dataset_version="v2",architecture="rgb-unet-spatial-text-v2",
            channels=16,diffusion_steps=8)
        net=PixelDenoiser(config)
        with torch.no_grad():
            net.spatial32.output.weight.fill_(0.01)
            net.spatial32.output.bias.fill_(0.1)
        feature=torch.randn(2,16,32,32)
        blank=torch.zeros(2,12,dtype=torch.long)
        unchanged=net.spatial32(feature,blank,net.text)
        self.assertTrue(torch.equal(unchanged,feature))
        nonempty=blank.clone()
        nonempty[0,0]=vocabulary_for("v2").index("triangle")
        changed=net.spatial32(feature,nonempty,net.text)
        self.assertFalse(torch.equal(changed[0],feature[0]))
        self.assertTrue(torch.equal(changed[1],feature[1]))

    def test_spatial_gradients_and_shape_objective(self):
        torch.manual_seed(7)
        config=PixelConfig(
            dataset_version="v2",architecture="rgb-unet-spatial-text-v2",
            channels=16,diffusion_steps=8)
        net=PixelDenoiser(config)
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
        samples=[
            synthetic_example_v2(42,"red","triangle","en"),
            synthetic_example_v2(43,"blue","circle","ar"),
            synthetic_example_v2(44,"yellow","square","en"),
        ]
        rgb=torch.stack([x[0] for x in samples])
        tokens=torch.stack([x[1] for x in samples])
        for i in range(2):
            optimizer.zero_grad(set_to_none=True)
            parts=shape_aware_training_objective(
                net,rgb,tokens,cfg_drop_prob=0.0,silhouette_weight=0.15)
            self.assertTrue(torch.isfinite(parts["total_loss"]))
            self.assertTrue(torch.isfinite(parts["silhouette_loss"]))
            self.assertGreaterEqual(float(parts["silhouette_loss"]),0)
            parts["total_loss"].backward()
            self.assertGreater(float(net.spatial32.output.weight.grad.abs().sum()),0)
            optimizer.step()
        # Output projection starts at zero; after training, attention
        # query/key/value gradients can flow to learn prompt structure.
        net.zero_grad(set_to_none=True)
        parts=shape_aware_training_objective(
            net,rgb,tokens,cfg_drop_prob=0.0,silhouette_weight=0.15)
        parts["total_loss"].backward()
        self.assertGreater(float(net.spatial32.query.weight.grad.abs().sum()),0)
        self.assertGreater(float(net.spatial16.key.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            shape_aware_training_objective(net,rgb,tokens,silhouette_weight=-1)

    def test_spatial_sampling_and_checkpoint_finetune_preserve_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source_path,_=self.create_pooled_v2_checkpoint(root)
            original_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
            args=SimpleNamespace(
                steps=2,batch_size=12,learning_rate=0.0001,seed=42,
                channels=16,cfg_drop_prob=0.15,
                dataset_version="v2",diffusion_steps=8,
                device="cpu",cpu_threads=1,log_every=1,
                output_root=str(root/"new-models"),
                init_from_v1=None,init_from_v2=str(source_path),
                triangle_fraction=0.5,conditioning="spatial",
                silhouette_weight=0.15,
            )
            new_path=train(args)
            self.assertNotEqual(new_path,source_path)
            self.assertEqual(
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
                original_hash
            )
            model=load_model(new_path)
            self.assertEqual(model.config.architecture,"rgb-unet-spatial-text-v2")
            report=torch.load(new_path,map_location="cpu",weights_only=True)
            self.assertEqual(report["silhouette_weight"],0.15)
            self.assertEqual(report["warm_start"]["source_checkpoint"],str(source_path))
            sample=generate(
                model,"مثلث أصفر",seed=11,device="cpu",
                sampler="ddim",steps=4,guidance_scale=3
            )
            self.assertEqual(sample.shape,(3,32,32))
            self.assertTrue(torch.isfinite(sample).all())

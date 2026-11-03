"""DDPM ancestral sampler smoke tests; no image-quality claim."""
import tempfile
import unittest
from pathlib import Path

import torch

from src.pixel_diffusion import PixelConfig, PixelDenoiser, generate
from scripts.pixel_diffusion import evaluate


class AncestralSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ancestral_seed_determinism_and_real_stochastic_variance(self):
        torch.manual_seed(17)
        model=PixelDenoiser(PixelConfig(
            channels=16,diffusion_steps=8,dataset_version="v2"
        ))
        a=generate(model,"مثلث أصفر",seed=42,device="cpu",
                   sampler="ancestral",guidance_scale=3.0)
        b=generate(model,"مثلث أصفر",seed=42,device="cpu",
                   sampler="ancestral",guidance_scale=3.0)
        c=generate(model,"مثلث أصفر",seed=123,device="cpu",
                   sampler="ancestral",guidance_scale=3.0)
        deterministic=generate(model,"مثلث أصفر",seed=42,device="cpu",
                               sampler="posterior_mean",guidance_scale=3.0)
        self.assertEqual(a.shape,(3,32,32))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(a,c))
        self.assertFalse(torch.equal(a,deterministic))

    def test_posterior_variance_requires_complete_timestep_schedule(self):
        model=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with self.assertRaises(ValueError):
            generate(model,"red circle",seed=42,
                     sampler="ancestral",steps=4)
        with self.assertRaises(ValueError):
            generate(model,"red circle",seed=42,sampler="notreal")

    def test_ancestral_evaluation_does_not_rewrite_original_checkpoint(self):
        torch.manual_seed(19)
        cfg=PixelConfig(channels=16,diffusion_steps=8,dataset_version="v2")
        net=PixelDenoiser(cfg)
        net.cfg_drop_prob=0.15
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"stochastic_samples"
            metrics=evaluate(net,output,seed=42,sample_steps=8,
                             scale=1,sampler="ancestral",guidance_scale=3.0)
            self.assertEqual(metrics["total_images"],24)
            self.assertEqual(len(list(output.glob("*.png"))),25)
            self.assertEqual(metrics["dataset_version"],"v2")
            with self.assertRaises(FileExistsError):
                evaluate(net,output,seed=42,sample_steps=8,
                         scale=1,sampler="ancestral",guidance_scale=3.0)

"""DDPM ancestral sampler smoke tests; no image-quality claim."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, generate, guided_noise, noise_schedule,
)
from scripts.pixel_diffusion import evaluate


class AncestralSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ancestral_seed_determinism_and_real_stochastic_variance(self):
        torch.manual_seed(17)
        config=PixelConfig(channels=16,diffusion_steps=8,dataset_version="v2")
        model=PixelDenoiser(config)
        original_randn=torch.randn
        original_guided=guided_noise
        expected_timesteps=list(range(config.diffusion_steps-1,-1,-1))

        with (
            patch("src.pixel_diffusion.torch.randn", wraps=original_randn) as rng,
            patch("src.pixel_diffusion.guided_noise", wraps=original_guided) as predict,
        ):
            a=generate(model,"مثلث أصفر",seed=42,device="cpu",
                       sampler="ancestral",guidance_scale=3.0)
            # One x_T draw plus exactly T-1 Gaussian posterior draws.
            self.assertEqual(rng.call_count,config.diffusion_steps)
            self.assertEqual(predict.call_count,config.diffusion_steps)
            self.assertEqual(
                [int(call.args[2].item()) for call in predict.call_args_list],
                expected_timesteps
            )
            self.assertTrue(all(
                call.kwargs.get("generator") is not None
                for call in rng.call_args_list
            ))
        b=generate(model,"مثلث أصفر",seed=42,device="cpu",
                   sampler="ancestral",guidance_scale=3.0)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(a.shape,(3,32,32))
        self.assertTrue(torch.isfinite(a).all())

        with (
            patch("src.pixel_diffusion.torch.randn", wraps=original_randn) as rng,
            patch("src.pixel_diffusion.guided_noise", wraps=original_guided) as predict,
        ):
            deterministic=generate(model,"مثلث أصفر",seed=42,device="cpu",
                                   sampler="posterior_mean",guidance_scale=3.0)
            self.assertEqual(rng.call_count,1)
            self.assertEqual(predict.call_count,config.diffusion_steps)
            self.assertEqual(
                [int(call.args[2].item()) for call in predict.call_args_list],
                expected_timesteps
            )
        self.assertTrue(torch.isfinite(deterministic).all())
        # We deliberately do NOT assert different final RGB tensors:
        # clipping can make outputs identical even with different noise.
        beta,alpha,cumulative=noise_schedule(config,"cpu")
        timestep=3
        previous=cumulative[timestep-1]
        variance=beta[timestep]*(1-previous)/(1-cumulative[timestep])
        self.assertGreater(float(variance),0.0)

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

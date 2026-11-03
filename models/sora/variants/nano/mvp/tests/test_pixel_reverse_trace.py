"""Regression tests for read-only learned RGB reverse diffusion tracing."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, generate, guided_noise,
)
from scripts.pixel_reverse_trace import trace_pair,export_trace


class PixelReverseTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self):
        model=PixelDenoiser(PixelConfig(
            dataset_version="v2",architecture="rgb-unet-spatial-text-v2",
            channels=16,diffusion_steps=8,
        ))
        model.cfg_drop_prob=0.15
        return model

    def test_trace_calls_every_timestep_and_prompt_only_response(self):
        torch.manual_seed(17)
        model=self.make_model()
        original=guided_noise
        with patch("scripts.pixel_reverse_trace.guided_noise",
                   wraps=original) as denoiser:
            report,snapshots,final=trace_pair(
                model,color="red",language="en",
                seed=42,guidance_scale=5.0
            )
        self.assertEqual(denoiser.call_count,8)
        self.assertEqual(
            [int(x.args[2][0]) for x in denoiser.call_args_list],
            list(range(7,-1,-1)),
        )
        self.assertEqual(report["all_reverse_timesteps_executed"],8)
        self.assertTrue(report["initial_noise_shared_between_prompts"])
        self.assertEqual(report["prompts"]["square"],"red square")
        self.assertEqual(report["prompts"]["triangle"],"red triangle")
        self.assertEqual(report["snapshots"][0]["timestep"],7)
        self.assertEqual(report["snapshots"][-1]["timestep"],0)
        self.assertEqual(report["snapshots"][0]["evolved_noisy_pixel_mae"],0)
        self.assertEqual(set(final),{"square","triangle"})
        self.assertEqual(set(snapshots),set(x["timestep"] for x in report["snapshots"]))
        for image in final.values():
            self.assertEqual(image.shape,(3,32,32))
            self.assertTrue(torch.isfinite(image).all())

    def test_trace_agrees_with_production_posterior_mean_generate(self):
        torch.manual_seed(19)
        model=self.make_model()
        report,snapshots,final=trace_pair(
            model,color="blue",language="ar",
            seed=42,guidance_scale=3.0
        )
        for shape,prompt in report["prompts"].items():
            actual=generate(
                model,prompt,seed=42,device="cpu",
                sampler="posterior_mean",guidance_scale=3.0
            )
            self.assertTrue(
                torch.allclose(actual,final[shape],atol=1e-4,rtol=1e-4),
                f"Trace diverged from real sampler for {shape}",
            )

    def test_outputs_are_model_snapshots_and_reproducible(self):
        torch.manual_seed(21)
        model=self.make_model()
        first,snapshots,final=trace_pair(model,seed=123,guidance_scale=1.0)
        second,snapshots2,final2=trace_pair(model,seed=123,guidance_scale=1.0)
        self.assertEqual(first,second)
        self.assertTrue(torch.equal(final["square"],final2["square"]))
        self.assertTrue(torch.equal(final["triangle"],final2["triangle"]))
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/"trace"
            source_hash=hashlib.sha256(
                model.input.weight.detach().numpy().tobytes()
            ).hexdigest()
            written=export_trace(first,snapshots,final,directory,scale=2)
            self.assertTrue((directory/"reverse_trace.png").exists())
            self.assertTrue((directory/"report.json").exists())
            self.assertTrue((directory/"square_final.png").exists())
            self.assertTrue((directory/"triangle_final.png").exists())
            self.assertEqual(written["all_reverse_timesteps_executed"],8)
            self.assertEqual(source_hash,hashlib.sha256(
                model.input.weight.detach().numpy().tobytes()
            ).hexdigest())
            with self.assertRaises(FileExistsError):
                export_trace(first,snapshots,final,directory,scale=2)

    def test_invalid_and_unsupported_inputs_are_rejected(self):
        model=self.make_model()
        for color,language,scale in (
            ("pink","en",1.0),("red","fr",1.0),("red","en",-1.0),
        ):
            with self.assertRaises(ValueError):
                trace_pair(model,color=color,language=language,
                           guidance_scale=scale)
        v1=PixelDenoiser(PixelConfig(
            dataset_version="v1",channels=16,diffusion_steps=8
        ))
        with self.assertRaises(ValueError):
            trace_pair(v1,guidance_scale=1.0)

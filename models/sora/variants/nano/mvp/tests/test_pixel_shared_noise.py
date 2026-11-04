"""Test identical-noise comparisons independently of triangle fidelity."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.pixel_diffusion import PixelConfig, PixelDenoiser
from scripts.pixel_diffusion import evaluate
from scripts.pixel_metrics import compare


class SharedNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_prompt_images_use_same_seed_when_requested(self):
        model=PixelDenoiser(PixelConfig(
            channels=16,dataset_version="v2",diffusion_steps=8
        ))
        model.cfg_drop_prob=0.15
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"shared"
            result=evaluate(
                model,folder,seed=42,sampler="ddim",sample_steps=4,
                guidance_scale=3.0,shared_noise=True,scale=1
            )
            self.assertEqual(result["total_images"],24)
            report=json.loads((folder/"report.json").read_text())
            self.assertTrue(report["shared_noise_across_concepts"])
            self.assertEqual({entry["seed"] for entry in report["items"]},{42})
            self.assertEqual(len({entry["prompt"] for entry in report["items"]}),24)

    def test_default_evaluation_retains_distinct_concept_seeds(self):
        model=PixelDenoiser(PixelConfig(
            channels=16,dataset_version="v2",diffusion_steps=8
        ))
        model.cfg_drop_prob=0.15
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"default"
            evaluate(model,directory,seed=42,sampler="ddim",
                     sample_steps=4,guidance_scale=3.0,scale=1)
            report=json.loads((directory/"report.json").read_text())
            self.assertFalse(report["shared_noise_across_concepts"])
            self.assertEqual(len({entry["seed"] for entry in report["items"]}),12)

    def test_multiscale_shared_noise_report_remains_auditable(self):
        model=PixelDenoiser(PixelConfig(
            channels=16,dataset_version="v2",diffusion_steps=8
        ))
        model.cfg_drop_prob=0.15
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"compare"
            summary=compare(
                model,out,scales=(5.0,),seeds=(42,),
                sampler="ancestral",sample_steps=8,shared_noise=True
            )
            self.assertTrue(summary["shared_noise_across_concepts"])
            self.assertEqual(summary["sampler"],"ancestral")
            self.assertEqual(summary["results"][0]["metrics"]["total_images"],24)
            self.assertTrue(Path(summary["results"][0]["preview_grid"]).exists())

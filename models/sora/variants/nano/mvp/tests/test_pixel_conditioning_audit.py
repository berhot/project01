"""Read-only v2 shape conditioning audit on ordinary q(x_t|x0) examples."""
import tempfile
import unittest
from pathlib import Path

import torch

from src.pixel_diffusion import PixelConfig, PixelDenoiser
from scripts.pixel_conditioning_audit import (
    validation_items, conditioning_audit, curriculum_exposure,
)


class PixelConditioningAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_targets_are_real_triangles_with_bilingual_matched_prompts(self):
        items=validation_items((42,123))
        self.assertEqual(len(items),32)
        expected={(color,shape,language)
                  for color in ("red","green","blue","yellow")
                  for shape in ("square","triangle")
                  for language in ("en","ar")}
        for seed in (42,123):
            got={(x["color"],x["shape"],x["language"])
                 for x in items if x["seed"]==seed}
            self.assertEqual(got,expected)
        for item in items:
            self.assertFalse(torch.equal(
                item["correct_tokens"],item["wrong_tokens"]))
            if item["shape"]=="triangle":
                self.assertLess(item["foreground_fill"],0.70)
                self.assertGreater(item["foreground_fill"],0.35)
            else:
                self.assertGreater(item["foreground_fill"],0.88)
        for seed in (42,123):
            for color in ("red","green","blue","yellow"):
                for language in ("en","ar"):
                    square=next(x for x in items if x["seed"]==seed
                        and x["color"]==color and x["language"]==language
                        and x["shape"]=="square")
                    triangle=next(x for x in items if x["seed"]==seed
                        and x["color"]==color and x["language"]==language
                        and x["shape"]=="triangle")
                    self.assertFalse(torch.equal(square["image"],triangle["image"]))

    def test_audit_is_deterministic_and_reports_correct_shape_coverage(self):
        torch.manual_seed(7)
        cfg=PixelConfig(dataset_version="v2",channels=16,diffusion_steps=8,
                        architecture="rgb-unet-spatial-text-v2")
        model=PixelDenoiser(cfg)
        first=conditioning_audit(model,seeds=(42,),timesteps=(1,3,6))
        second=conditioning_audit(model,seeds=(42,),timesteps=(1,3,6))
        self.assertEqual(first,second)
        self.assertEqual(first["sample_count"],16)
        self.assertEqual(first["evaluation_count"],48)
        self.assertEqual(len(first["items"]),48)
        self.assertEqual(
            set(first["by_shape_language"]),
            {"square_en","square_ar","triangle_en","triangle_ar"}
        )
        for group in first["by_shape_language"].values():
            self.assertEqual(group["count"],12)
            self.assertGreaterEqual(group["correct_prompt_win_rate"],0)
            self.assertLessEqual(group["correct_prompt_win_rate"],1)
        self.assertGreater(
            first["validation_foreground_fill"]["square"],
            first["validation_foreground_fill"]["triangle"],
        )

    def test_triangle_heavy_curriculum_exposure_and_language_imbalance(self):
        coverage=curriculum_exposure(steps=120,batch_size=12)
        self.assertEqual(coverage,{
            "circle_ar":120,"circle_en":240,
            "square_ar":240,"square_en":120,
            "triangle_ar":360,"triangle_en":360,
        })
        self.assertEqual(sum(coverage.values()),1440)

    def test_refuses_invalid_timestep_and_non_v2_checkpoint(self):
        v1=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with self.assertRaises(ValueError):
            conditioning_audit(v1,seeds=(42,),timesteps=(1,))
        v2=PixelDenoiser(PixelConfig(
            dataset_version="v2",channels=16,diffusion_steps=8))
        for steps in ((),(1,1),(-1,),(8,)):
            with self.assertRaises(ValueError):
                conditioning_audit(v2,seeds=(42,),timesteps=steps)

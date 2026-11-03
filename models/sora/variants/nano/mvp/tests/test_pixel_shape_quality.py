"""Geometry-quality scoring is stricter than legacy 32x32 fill heuristics."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.pixel_dataset_v2 import synthetic_example_v2
from src.pixel_shape_quality import shape_fidelity,score_model_sample
from scripts.pixel_shape_quality import audit_gallery,compare_galleries


class PixelShapeQualityTests(unittest.TestCase):
    def test_real_training_shapes_have_correct_ideal_mask_fidelity(self):
        for shape in ("circle","square","triangle"):
            rgb,_,_,_=synthetic_example_v2(
                52,color="blue",shape=shape,
                language="en",split="validation",variant=0,
            )
            result=score_model_sample(rgb,"blue",shape)
            self.assertGreater(result["expected_mask_iou"],0.95)
            self.assertTrue(result["qualified_geometry"],result)
            self.assertTrue(result["strict_prompt_pass"],result)

    def test_square_does_not_pass_as_triangle_or_circle(self):
        img,_,_,_=synthetic_example_v2(
            67,color="yellow",shape="square",split="validation",variant=0)
        square=shape_fidelity(img,"square")
        triangle=shape_fidelity(img,"triangle")
        circle=shape_fidelity(img,"circle")
        self.assertTrue(square["qualified_geometry"])
        self.assertFalse(triangle["qualified_geometry"])
        self.assertFalse(circle["qualified_geometry"])
        self.assertLess(triangle["expected_mask_iou"],0.8)
        self.assertLess(circle["expected_mask_iou"],0.90)

    def test_empty_or_edge_filled_images_cannot_claim_high_quality(self):
        empty=torch.ones(3,32,32)
        result=shape_fidelity(empty,"circle")
        self.assertFalse(result["qualified_geometry"])
        self.assertFalse(result["touches_image_border"] is False)
        saturated=torch.ones(3,32,32)*-1
        self.assertFalse(shape_fidelity(saturated,"square")["qualified_geometry"])
        with self.assertRaises(ValueError):
            shape_fidelity(torch.zeros(1,3,32,32),"circle")
        with self.assertRaises(ValueError):
            shape_fidelity(torch.zeros(3,32,32),"hexagon")

    def make_gallery(self,root,swap_triangle=False):
        directory=Path(root)
        directory.mkdir(parents=True)
        rows=[]
        for shape in ("circle","square","triangle"):
            for color in ("red","green","blue","yellow"):
                for language in ("en","ar"):
                    target="square" if swap_triangle and shape=="triangle" else shape
                    image,_,prompt,_=synthetic_example_v2(
                        79,color=color,shape=target,
                        language=language,split="validation",variant=0)
                    pixels=((image.permute(1,2,0).clamp(-1,1).numpy()+1)*127.5)
                    filename=f"{color}_{shape}_{language}.png"
                    Image.fromarray(pixels.round().astype("uint8")).save(
                        directory/filename
                    )
                    rows.append({"image":filename,"language":language,
                                 "seed":42,"prompt":prompt})
        (directory/"report.json").write_text(json.dumps({
            "dataset_version":"v2","sampler":"posterior_mean",
            "guidance_scale":5.0,"x0_clipping":"fixed",
            "shared_noise_across_concepts":False,"items":rows
        }))
        return directory

    def test_gallery_reports_failures_and_rejects_fake_triangle_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            base=self.make_gallery(Path(temp)/"bad",swap_triangle=True)
            valid=self.make_gallery(Path(temp)/"valid",swap_triangle=False)
            failed=audit_gallery(base)
            good=audit_gallery(valid)
            self.assertEqual(failed["sample_count"],24)
            self.assertEqual(failed["by_shape"]["triangle"]["geometry_pass_count"],0)
            self.assertEqual(good["by_shape"]["triangle"]["geometry_pass_count"],8)
            self.assertEqual(good["by_shape"]["circle"]["geometry_pass_count"],8)
            self.assertEqual(good["by_shape"]["square"]["geometry_pass_count"],8)
            accepted=compare_galleries(failed,good)
            self.assertTrue(accepted["eligible_for_human_review"],accepted)
            rejected=compare_galleries(good,failed)
            self.assertFalse(rejected["eligible_for_human_review"])
            self.assertFalse(rejected["checks"]["at_least_one_recognizable_triangle"])
            (base/"blue_circle_en.png").unlink()
            with self.assertRaises(FileNotFoundError):
                audit_gallery(base)

    def test_gallery_rejects_unmatched_seeds_or_sampler(self):
        with tempfile.TemporaryDirectory() as temp:
            before=self.make_gallery(Path(temp)/"before")
            after=self.make_gallery(Path(temp)/"after")
            original=audit_gallery(before)
            updated=audit_gallery(after)
            self.assertFalse(compare_galleries(original,updated)[
                "eligible_for_human_review"])
            updated["items"][0]["seed"]=123
            with self.assertRaisesRegex(ValueError,"seeds"):
                compare_galleries(original,updated)

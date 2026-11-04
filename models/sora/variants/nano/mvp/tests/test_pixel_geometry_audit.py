"""Read-only geometry reconstruction audits: fixed x0 from actual q(x_t|x0)."""
import unittest
import torch

from src.pixel_diffusion import PixelConfig,PixelDenoiser
from scripts.pixel_geometry_audit import (
    foreground,mask_iou,geometry_audit,
)


class PixelGeometryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self):
        torch.manual_seed(19)
        model=PixelDenoiser(PixelConfig(
            image_size=32,channels=16,diffusion_steps=8,
            dataset_version="v2",architecture="rgb-unet-spatial-text-v2",
        ))
        model.cfg_drop_prob=0.15
        return model

    def test_foreground_and_iou_math(self):
        image=torch.ones((2,3,32,32))*0.2
        image[:,0,5:17,3:15]=1
        mask=foreground(image)
        self.assertEqual(mask.shape,(2,32,32))
        self.assertEqual(int(mask.sum()),2*12*12)
        self.assertTrue(torch.equal(mask_iou(mask,mask),torch.ones(2)))
        self.assertTrue(torch.equal(
            mask_iou(mask,torch.zeros_like(mask)),torch.zeros(2)))
        with self.assertRaises(ValueError):
            foreground(torch.ones(3,32,32))
        with self.assertRaises(ValueError):
            mask_iou(mask,torch.ones(2,10,10,dtype=torch.bool))

    def test_bilingual_shape_geometry_groups_and_fixed_seed_repeatability(self):
        model=self.make_model()
        first=geometry_audit(model,seeds=(42,),timesteps=(1,4,6),
                             guidance_scale=5.0)
        second=geometry_audit(model,seeds=(42,),timesteps=(1,4,6),
                              guidance_scale=5.0)
        self.assertEqual(first,second)
        self.assertEqual(first["evaluation_count"],48)
        self.assertEqual(len(first["items"]),48)
        self.assertEqual(
            set(first["by_shape_language"]),
            {"square_en","square_ar","triangle_en","triangle_ar"})
        for key,group in first["by_shape_language"].items():
            self.assertEqual(group["count"],12)
            for value in (
                "correct_prompt_mean_mask_iou",
                "wrong_prompt_mean_mask_iou",
                "correct_prompt_iou_win_rate",
                "correct_prompt_predicted_shape_rate",
            ):
                self.assertGreaterEqual(group[value],0)
                self.assertLessEqual(group[value],1)
        for row in first["items"]:
            self.assertIn(row["correct_prompt_predicted_shape"],
                          ("circle","square","triangle","unknown"))
            self.assertGreater(row["target_foreground_pixels"],0)
            self.assertGreaterEqual(row["correct_prompt_mask_iou"],0)
            self.assertLessEqual(row["correct_prompt_mask_iou"],1)
        self.assertEqual(first["scope"],
                         "held-out synthetic image x_t recovery, not pure-noise image generation")

    def test_audit_does_not_update_model_weights(self):
        model=self.make_model()
        first_weights={name:t.clone() for name,t in model.state_dict().items()}
        geometry_audit(model,seeds=(42,),timesteps=(3,),guidance_scale=1.0)
        self.assertTrue(all(
            torch.equal(t,model.state_dict()[name])
            for name,t in first_weights.items()
        ))

    def test_invalid_checkpoint_schedule_and_guidance(self):
        model=self.make_model()
        for steps in ((),(1,1),(-1,),(8,)):
            with self.assertRaises(ValueError):
                geometry_audit(model,seeds=(42,),timesteps=steps)
        with self.assertRaises(ValueError):
            geometry_audit(model,seeds=(42,42),timesteps=(1,))
        with self.assertRaises(ValueError):
            geometry_audit(model,seeds=(42,),timesteps=(1,),
                           guidance_scale=float("nan"))
        v1=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with self.assertRaises(ValueError):
            geometry_audit(v1,seeds=(42,),timesteps=(1,))

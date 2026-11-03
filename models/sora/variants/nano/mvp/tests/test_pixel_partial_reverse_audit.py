"""Tests for real partial-DDPM reverse processes from held-out known x0."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.pixel_partial_reverse_audit import (
    partial_reverse_audit, save_audit,
)
from scripts.pixel_conditioning_audit import validation_items
from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, guided_noise, noise_schedule,
)


class PartialReverseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(8)
        model=PixelDenoiser(PixelConfig(
            channels=16,diffusion_steps=8,dataset_version="v2",
            architecture="rgb-unet-spatial-text-v2",
        ))
        model.cfg_drop_prob=0.15
        return model

    def test_executes_all_reverse_steps_and_is_deterministic(self):
        model=self.model()
        original=guided_noise
        with patch("scripts.pixel_partial_reverse_audit.guided_noise",
                   wraps=original) as called:
            first,images=partial_reverse_audit(
                model,seeds=(42,),colors=("red",),
                start_timesteps=(0,3,6),guidance_scale=1.0
            )
            self.assertEqual(called.call_count,1+4+7)
        second,images_second=partial_reverse_audit(
            model,seeds=(42,),colors=("red",),
            start_timesteps=(0,3,6),guidance_scale=1.0
        )
        self.assertEqual(first,second)
        self.assertEqual(first["evaluation_count"],12)
        self.assertEqual(first["start_timesteps"],[0,3,6])
        self.assertEqual(len(images),12)
        self.assertTrue(all(torch.equal(image,images_second[key])
                            for key,image in images.items()))
        for timestep in (0,3,6):
            group=first["by_start_timestep"][str(timestep)]
            self.assertEqual(group["cases"],4)
            self.assertGreaterEqual(group["mean_target_mask_iou"],0)
            self.assertLessEqual(group["mean_target_mask_iou"],1)
        for key,image in images.items():
            self.assertEqual(image.shape,(3,32,32))
            self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(
            first["scope"],
            "conditional reconstruction given a noisy real target, NOT text-to-image generation"
        )

    def test_same_initial_noise_on_swapped_words_and_matched_shapes(self):
        model=self.model()
        captured=[]
        original=guided_noise
        def spy(net,noisy,times,tokens,scale):
            if not captured:
                captured.append((noisy.detach().clone(),times.detach().clone(),
                                 tokens.detach().clone()))
            return original(net,noisy,times,tokens,scale)
        with patch("scripts.pixel_partial_reverse_audit.guided_noise",
                   side_effect=spy):
            partial_reverse_audit(
                model,seeds=(42,),colors=("red",),
                start_timesteps=(5,),guidance_scale=1.0
            )
        noisy,times,tokens=captured[0]
        self.assertEqual(noisy.shape,(8,3,32,32))
        self.assertTrue(torch.equal(noisy[:4],noisy[4:]))
        self.assertTrue((times==5).all())
        self.assertFalse(torch.equal(tokens[:4],tokens[4:]))
        # Correct shapes share Gaussian eps within color/language/seed,
        # but they have DISTINCT actual source images.
        examples=[e for e in validation_items((42,)) if e["color"]=="red"]
        original_images=torch.stack([e["image"] for e in examples])
        _,_,cum=noise_schedule(model.config,"cpu")
        a=cum[5]
        eps=(noisy[:4]-a.sqrt()*original_images)/(1-a).sqrt()
        self.assertTrue(torch.allclose(eps[0],eps[2],atol=1e-5))
        self.assertTrue(torch.allclose(eps[1],eps[3],atol=1e-5))
        self.assertFalse(torch.equal(noisy[0],noisy[2]))

    def test_near_clean_start_preserves_reference_geometry(self):
        model=self.model()
        audit,_=partial_reverse_audit(
            model,seeds=(42,),colors=("red",),
            start_timesteps=(0,),guidance_scale=1.0
        )
        self.assertGreater(audit["by_start_timestep"]["0"]["mean_target_mask_iou"],0.95)
        self.assertEqual(audit["by_start_timestep"]["0"]["correct_shape_rate"],1.0)

    def test_read_only_and_model_generated_grid_exports(self):
        model=self.model()
        source={k:v.clone() for k,v in model.state_dict().items()}
        report,images=partial_reverse_audit(
            model,seeds=(42,),colors=("red",),
            start_timesteps=(2,5),guidance_scale=1.0
        )
        self.assertTrue(all(
            torch.equal(v,model.state_dict()[k]) for k,v in source.items()
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"actual-model-outputs"
            preview=save_audit(report,images,path,scale=2)
            self.assertTrue(preview.is_file())
            self.assertTrue((path/"report.json").is_file())
            with self.assertRaises(FileExistsError):
                save_audit(report,images,path,scale=2)

    def test_rejects_invalid_timesteps_and_inputs(self):
        model=self.model()
        base=dict(seeds=(42,),colors=("red",),start_timesteps=(5,))
        for updates in (
            {"seeds":()},{"seeds":(42,42)},
            {"colors":()},{"colors":("pink",)},
            {"start_timesteps":()},{"start_timesteps":(1,1)},
            {"start_timesteps":(-1,)},{"start_timesteps":(8,)},
            {"guidance_scale":float("nan")},
        ):
            params=dict(base)
            params.update(updates)
            with self.assertRaises(ValueError):
                partial_reverse_audit(model,**params)
        model_v1=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with self.assertRaises(ValueError):
            partial_reverse_audit(model_v1,seeds=(42,),colors=("red",),
                                  start_timesteps=(5,),guidance_scale=1.0)

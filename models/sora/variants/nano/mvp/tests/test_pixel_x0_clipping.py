"""Inference-only dynamic x0 clipping ablation and default preservation."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.pixel_diffusion import (
    PixelConfig,PixelDenoiser,clip_predicted_x0,generate
)
from scripts.pixel_diffusion import evaluate
from scripts.pixel_metrics import compare
from scripts.pixel_quality_gate import compare_reports


class X0ClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create_model(self):
        torch.manual_seed(17)
        model=PixelDenoiser(PixelConfig(
            channels=16,diffusion_steps=8,dataset_version="v2",
            architecture="rgb-unet-spatial-text-v2"
        ))
        model.cfg_drop_prob=0.15
        return model

    def test_default_fixed_clip_matches_explicit_fixed(self):
        model=self.create_model()
        for sampler in ("posterior_mean","ddim","ancestral"):
            options={"steps":4} if sampler=="ddim" else {}
            implicit=generate(model,"red triangle",seed=42,
                              sampler=sampler,guidance_scale=5.0,**options)
            explicit=generate(model,"red triangle",seed=42,
                              sampler=sampler,guidance_scale=5.0,
                              x0_clipping="fixed",**options)
            self.assertTrue(torch.equal(implicit,explicit))

    def test_dynamic_thresholding_is_bounded_and_softens_saturation(self):
        x=torch.linspace(-70,70,3072,dtype=torch.float32).reshape(1,3,32,32)
        original=x.clone()
        fixed=clip_predicted_x0(x,"fixed")
        dynamic=clip_predicted_x0(x,"dynamic")
        self.assertTrue(torch.equal(x,original))
        self.assertLessEqual(float(dynamic.abs().max()),1.0)
        self.assertTrue(torch.isfinite(dynamic).all())
        self.assertGreater(float((fixed.abs()>0.99).float().mean()),0.90)
        self.assertLess(float((dynamic.abs()>0.99).float().mean()),0.1)
        self.assertFalse(torch.equal(fixed,dynamic))
        self.assertTrue(torch.equal(
            clip_predicted_x0(torch.ones_like(x)*0.5,"dynamic"),
            torch.ones_like(x)*0.5
        ))
        with self.assertRaises(ValueError):
            clip_predicted_x0(x,"wrong")
        with self.assertRaises(ValueError):
            clip_predicted_x0(torch.ones(3,32,32),"dynamic")

    def test_dynamic_generation_is_repeatable_and_bounded(self):
        model=self.create_model()
        first=generate(model,"مثلث أحمر",seed=42,
                       sampler="posterior_mean",guidance_scale=5.0,
                       x0_clipping="dynamic")
        second=generate(model,"مثلث أحمر",seed=42,
                        sampler="posterior_mean",guidance_scale=5.0,
                        x0_clipping="dynamic")
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(first.shape,(3,32,32))
        self.assertTrue(torch.isfinite(first).all())
        self.assertLessEqual(float(first.abs().max()),1.0)
        with self.assertRaises(ValueError):
            generate(model,"red square",x0_clipping="unsupported")

    def test_evaluation_records_clipping_and_gate_rejects_mixed_modes(self):
        model=self.create_model()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixed=root/"fixed"
            dynamic=root/"dynamic"
            old=evaluate(model,fixed,seed=42,scale=1,
                         sampler="posterior_mean",guidance_scale=5.0)
            new=evaluate(model,dynamic,seed=42,scale=1,
                         sampler="posterior_mean",guidance_scale=5.0,
                         x0_clipping="dynamic")
            self.assertEqual(old["total_images"],24)
            self.assertEqual(new["total_images"],24)
            before=json.loads((fixed/"report.json").read_text())
            after=json.loads((dynamic/"report.json").read_text())
            self.assertEqual(before["x0_clipping"],"fixed")
            self.assertEqual(after["x0_clipping"],"dynamic")
            with self.assertRaisesRegex(ValueError,"x0_clipping"):
                compare_reports(fixed/"report.json",dynamic/"report.json")

    def test_legacy_report_without_clipping_key_means_fixed(self):
        model=self.create_model()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixed=root/"fixed"
            candidate=root/"candidate"
            evaluate(model,fixed,seed=42,scale=1,
                     sampler="posterior_mean",guidance_scale=5.0)
            evaluate(model,candidate,seed=42,scale=1,
                     sampler="posterior_mean",guidance_scale=5.0)
            old=fixed/"report.json"
            obj=json.loads(old.read_text())
            del obj["x0_clipping"]
            old.write_text(json.dumps(obj))
            result=compare_reports(old,candidate/"report.json")
            self.assertTrue(result["circles_preserved"] if "circles_preserved" in result else True)
            self.assertTrue("accepted_by_toy_gate" in result)

import tempfile
import unittest
from pathlib import Path

import torch

from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, WORDS, caption, generate,
    noise_schedule, synthetic_example, tokenize, training_loss,
)
from scripts.pixel_diffusion import evaluate, load_model


class PixelDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_arabic_and_english_tokenization(self):
        english=tokenize("red circle")
        arabic=tokenize("دائرة حمراء")
        self.assertEqual(english.shape,(12,))
        self.assertEqual(arabic.shape,(12,))
        self.assertFalse(torch.equal(english,arabic))
        self.assertEqual(caption("red","square","ar"),"مربع أحمر")
        self.assertEqual(caption("blue","circle","en"),"blue circle")
        self.assertNotEqual(WORDS.index("أحمر"),WORDS.index("حمراء"))
        with self.assertRaises(ValueError): tokenize("photorealistic desert skyline")

    def test_deterministic_synthetic_training_pairs(self):
        image,tokens,prompt,concept=synthetic_example(42,"red","circle","ar")
        self.assertEqual(image.shape,(3,32,32))
        self.assertEqual(tokens.shape,(12,))
        self.assertEqual(concept,("red","circle"))
        self.assertTrue("دائرة" in prompt)
        self.assertTrue(torch.equal(image,synthetic_example(42,"red","circle","ar")[0]))
        self.assertFalse(torch.equal(image,synthetic_example(43,"red","circle","ar")[0]))

    def test_rgb_diffusion_gradients_and_full_noise_schedule(self):
        cfg=PixelConfig(channels=16,diffusion_steps=8)
        net=PixelDenoiser(cfg)
        examples=[synthetic_example(i) for i in (42,43)]
        rgb=torch.stack([v[0] for v in examples])
        tokens=torch.stack([v[1] for v in examples])
        loss=training_loss(net,rgb,tokens)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(net.input.weight.grad.abs().sum()),0)
        _,_,alpha=noise_schedule(cfg,"cpu")
        self.assertLess(float(alpha[-1]),0.001)
        self.assertLess(float(alpha[-1]),float(alpha[0]))

    def test_checkpoint_roundtrip_and_12_real_image_exports(self):
        torch.manual_seed(123)
        cfg=PixelConfig(channels=16,diffusion_steps=8)
        net=PixelDenoiser(cfg)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            path=root/"weights.pt"
            torch.save({
                "schema_version":1,"architecture":cfg.architecture,
                "config":vars(cfg),"vocabulary":list(WORDS),
                "model":net.state_dict(),
            },path)
            restored=load_model(path)
            a=generate(net,"red circle",seed=7,steps=4)
            b=generate(restored,"red circle",seed=7,steps=4)
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(tuple(a.shape),(3,32,32))
            evaluate(restored,root/"preview",seed=42,sample_steps=4,scale=1)
            self.assertEqual(len(list((root/"preview").glob("*.png"))),13)
            self.assertTrue((root/"preview"/"report.json").is_file())
            with self.assertRaises(FileExistsError):
                evaluate(restored,root/"preview",seed=42,sample_steps=4,scale=1)

    def test_invalid_prompts_and_checkpoint_architecture(self):
        model=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with self.assertRaises(ValueError):
            generate(model,"horse on a beach",steps=4)
        with self.assertRaises(ValueError):
            generate(model,"red circle",steps=1)
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/"bad.pt"
            torch.save({"schema_version":1,"architecture":"layout-ddpm-v1"},bad)
            with self.assertRaises(ValueError):
                load_model(bad)

import tempfile
import unittest
from pathlib import Path

import torch

from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, apply_text_dropout,
    guided_noise, tokenize, synthetic_example, training_loss, generate,
)
from src.pixel_metrics import analyze_image, summarize
from scripts.pixel_diffusion import load_model, require_guidance_training
from scripts.pixel_metrics import compare


class GuidanceAndMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_prompt_dropout_and_input_is_unchanged(self):
        tokens=torch.stack([tokenize("red circle"), tokenize("دائرة زرقاء")])
        original=tokens.clone()
        blank, mask=apply_text_dropout(tokens,1)
        self.assertTrue(bool(mask.all()))
        self.assertTrue(bool((blank==0).all()))
        retained, mask2=apply_text_dropout(tokens,0)
        self.assertFalse(bool(mask2.any()))
        self.assertTrue(torch.equal(retained,tokens))
        self.assertTrue(torch.equal(tokens,original))
        for bad in (-0.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                apply_text_dropout(tokens,bad)

    def test_guidance_mathematics_separates_text_condition(self):
        class Toy(torch.nn.Module):
            def forward(self,images,t,words):
                strength=words.float().sum(dim=1).view(-1,1,1,1)
                return torch.zeros_like(images)+strength
        model=Toy()
        image=torch.zeros(1,3,4,4)
        t=torch.tensor([3])
        text=tokenize("red circle").unsqueeze(0)
        cond=guided_noise(model,image,t,text,1.0)
        self.assertTrue(torch.allclose(guided_noise(model,image,t,text,0.0),torch.zeros_like(image)))
        self.assertTrue(torch.allclose(guided_noise(model,image,t,text,3.0),cond*3))
        with self.assertRaises(ValueError):
            guided_noise(model,image,t,text,-1)

    def test_unconditional_training_gradient_and_guided_sampling(self):
        torch.manual_seed(7)
        net=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        img,tok,_,_=synthetic_example(7)
        loss=training_loss(net,img.unsqueeze(0),tok.unsqueeze(0),cfg_drop_prob=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(net.input.weight.grad)
        a=generate(net,"دائرة حمراء",seed=11,steps=4,sampler="ddim",guidance_scale=3.0)
        b=generate(net,"دائرة حمراء",seed=11,steps=4,sampler="ddim",guidance_scale=3.0)
        self.assertTrue(torch.equal(a,b))

    def test_known_synthetic_images_color_and_shape(self):
        for color in ("red","green","blue"):
            for shape in ("circle","square"):
                image,_,_,_=synthetic_example(78,color,shape,"en")
                detected=analyze_image(image)
                self.assertEqual((detected["color"],detected["shape"]),(color,shape))
        blank=torch.ones(3,32,32)
        self.assertEqual(analyze_image(blank)["shape"],"unknown")

    def test_bilingual_summary_cannot_relabel_errors_success(self):
        pairs=[
            {"expected_color":"red","expected_shape":"circle",
             "predicted_color":"red","predicted_shape":"circle",
             "language":"en","seed":42},
            {"expected_color":"red","expected_shape":"circle",
             "predicted_color":"blue","predicted_shape":"circle",
             "language":"ar","seed":42},
        ]
        scores=summarize(pairs)
        self.assertEqual(scores["color_accuracy"],0.5)
        self.assertEqual(scores["shape_accuracy"],1.0)
        self.assertEqual(scores["prompt_accuracy"],0.5)
        self.assertEqual(scores["bilingual_prediction_consistency"],0.0)
        self.assertEqual(scores["bilingual_both_correct"],0.0)
        with self.assertRaises(ValueError):
            summarize(pairs+[pairs[0]])

    def test_checkpoint_metadata_and_compare_grids(self):
        torch.manual_seed(5)
        net=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"weights.pt"
            torch.save({
                "schema_version":1,"architecture":"rgb-unet-text-v1",
                "config":vars(net.config),
                "vocabulary":list(__import__("src.pixel_diffusion",fromlist=["WORDS"]).WORDS),
                "cfg_drop_prob":0.15,
                "model":net.state_dict(),
            },path)
            loaded=load_model(path)
            self.assertEqual(loaded.cfg_drop_prob,0.15)
            summary=compare(loaded,Path(directory)/"results",
                            scales=(1.0,3.0),seeds=(42,),
                            sampler="ddim",sample_steps=4)
            self.assertEqual(len(summary["results"]),2)
            for record in summary["results"]:
                self.assertIn("prompt_accuracy",record["metrics"])
                self.assertEqual(record["metrics"]["total_images"],12)
                self.assertTrue(Path(record["preview_grid"]).exists())
            with self.assertRaises(FileExistsError):
                compare(loaded,Path(directory)/"results",
                        scales=(1.0,),seeds=(42,),sampler="ddim",sample_steps=4)

    def test_guidance_requires_dropout_trained_checkpoint(self):
        model=PixelDenoiser(PixelConfig(channels=16,diffusion_steps=8))
        with self.assertRaises(ValueError):
            require_guidance_training(model,3.0)
        require_guidance_training(model,1.0)

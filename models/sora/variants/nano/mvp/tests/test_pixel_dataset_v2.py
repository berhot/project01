import tempfile
import unittest
from pathlib import Path

import torch

from src.pixel_diffusion import (
    WORDS, PixelConfig, PixelDenoiser, generate, tokenize, vocabulary_for,
)
from src.pixel_dataset_v2 import (
    WORDS_V2, COLORS, SHAPES, CONCEPTS, caption_v2, tokenize_v2,
    synthetic_example_v2, balanced_training_batch,
)
from src.pixel_metrics import analyze_image, summarize
from scripts.pixel_diffusion import load_model, evaluate
from scripts.dataset_v2 import preview


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_v2_grammar_and_index_preservation(self):
        self.assertEqual(tuple(WORDS_V2[:len(WORDS)]),WORDS)
        self.assertEqual(vocabulary_for("v1"), WORDS)
        self.assertEqual(vocabulary_for("v2"), WORDS_V2)
        self.assertEqual(len(CONCEPTS),12)
        self.assertEqual(len(COLORS),4)
        self.assertEqual(len(SHAPES),3)
        self.assertEqual(caption_v2("yellow","circle","ar"),"دائرة صفراء")
        self.assertEqual(caption_v2("yellow","triangle","ar"),"مثلث أصفر")
        self.assertEqual(caption_v2("green","square","ar"),"مربع أخضر")
        self.assertTrue(torch.equal(
            tokenize("red circle"),tokenize_v2("red circle")
        ))
        self.assertNotEqual(int(tokenize_v2("yellow triangle")[0]),0)
        self.assertNotEqual(int(tokenize_v2("مثلث أصفر")[0]),0)
        with self.assertRaises(ValueError):
            tokenize("yellow triangle")
        with self.assertRaises(ValueError):
            caption_v2("purple","triangle","en")

    def test_all_twelve_concepts_heldout_both_languages(self):
        for color,shape in CONCEPTS:
            for language in ("en","ar"):
                image,tokens,prompt,concept=synthetic_example_v2(
                    782,color,shape,language,split="validation"
                )
                self.assertEqual(concept,(color,shape))
                self.assertEqual(image.shape,(3,32,32))
                self.assertEqual(tokens.shape,(12,))
                self.assertTrue(prompt)
                detected=analyze_image(image,dataset_version="v2")
                self.assertEqual(
                    (detected["color"],detected["shape"]),(color,shape),
                    f"{color} {shape}: {detected}"
                )
        with self.assertRaises(ValueError):
            synthetic_example_v2(1,split="holdout")

    def test_determinism_and_disjoint_seed_spaces(self):
        a=synthetic_example_v2(123,"blue","triangle","ar",split="train")
        b=synthetic_example_v2(123,"blue","triangle","ar",split="train")
        c=synthetic_example_v2(123,"blue","triangle","ar",split="validation")
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertEqual(a[2],b[2])
        self.assertFalse(torch.equal(a[0],c[0]))
        imgs,tokens=balanced_training_batch(1,12,42)
        self.assertEqual(imgs.shape,(12,3,32,32))
        self.assertEqual(tokens.shape,(12,12))
        x,y=balanced_training_batch(1,12,42)
        self.assertTrue(torch.equal(imgs,x))
        self.assertTrue(torch.equal(tokens,y))

    def test_v2_checkpoint_tokenizer_sampling_and_24_images(self):
        cfg=PixelConfig(channels=16,diffusion_steps=8,dataset_version="v2")
        model=PixelDenoiser(cfg)
        self.assertEqual(model.text.token.num_embeddings,len(WORDS_V2))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ckpt=root/"weights.pt"
            torch.save({
                "schema_version":1,"architecture":"rgb-unet-text-v1",
                "config":vars(cfg),"vocabulary":list(WORDS_V2),
                "cfg_drop_prob":0.15,
                "model":model.state_dict(),
            },ckpt)
            restored=load_model(ckpt)
            image=generate(restored,"مثلث أصفر",seed=42,steps=4,
                           sampler="ddim",guidance_scale=3)
            self.assertEqual(image.shape,(3,32,32))
            metrics=evaluate(
                restored,root/"generated",seed=42,
                sample_steps=4,scale=1,sampler="ddim",guidance_scale=3
            )
            self.assertEqual(metrics["dataset_version"],"v2")
            self.assertEqual(metrics["total_images"],24)
            self.assertEqual(metrics["bilingual_pairs"],12)
            self.assertEqual(len(list((root/"generated").glob("*.png"))),25)
            old_record=torch.load(ckpt,weights_only=True)
            old_record["vocabulary"]=list(WORDS)
            torch.save(old_record,root/"invalid.pt")
            with self.assertRaises(ValueError):
                load_model(root/"invalid.pt")

    def test_procedural_preview_is_labeled_and_not_model_output(self):
        with tempfile.TemporaryDirectory() as d:
            report=preview(Path(d)/"preview",seed=7,split="validation",scale=1)
            self.assertEqual(report["samples"],24)
            self.assertEqual(report["classes"],12)
            self.assertIn("NOT diffusion-generated",report["source"])
            self.assertTrue((Path(d)/"preview"/"manifest.json").exists())
            with self.assertRaises(FileExistsError):
                preview(Path(d)/"preview",seed=7,split="validation",scale=1)

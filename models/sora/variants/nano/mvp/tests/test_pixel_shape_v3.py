"""V3 high-noise shape learning: valid Gaussian pairs, balanced replay, safety."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.pixel_diffusion import PixelConfig,PixelDenoiser,vocabulary_for
from src.pixel_dataset_v2 import balanced_training_batch,CONCEPTS
from src.pixel_shape_v3 import swapped_shape_tokens,high_noise_v3_objective
from scripts.pixel_diffusion import load_model,train


class ShapeLearningV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def new_model(self):
        return PixelDenoiser(PixelConfig(
            channels=16,diffusion_steps=8,dataset_version="v2",
            architecture="rgb-unet-spatial-text-v2",
        ))

    def source(self,root):
        model=self.new_model()
        path=Path(root)/"baseline.pt"
        torch.save({
            "schema_version":1,"architecture":model.config.architecture,
            "config":vars(model.config),
            "vocabulary":list(vocabulary_for("v2")),
            "cfg_drop_prob":0.15,"training_steps":800,
            "model":model.state_dict(),
        },path)
        return path

    def train_args(self,root,baseline):
        return SimpleNamespace(
            steps=2,batch_size=12,learning_rate=0.00001,seed=42,
            channels=16,cfg_drop_prob=0.15,dataset_version="v2",
            diffusion_steps=8,device="cpu",cpu_threads=1,log_every=1,
            output_root=str(Path(root)/"v3"),init_from_v1=None,
            init_from_v2=str(baseline),triangle_fraction=None,
            conditioning="spatial",silhouette_weight=0.15,
            paired_shape_weight=0.0,paired_shape_pairs=4,
            paired_silhouette_weight=0.5,preserve_weight=1.0,
            preserve_null_weight=0.5,shape_v3=True,
            v3_noise_weight=0.15,v3_contrast_weight=0.1,
            v3_margin=0.003,v3_low_fraction=0.55,
            v3_high_fraction=0.85,
        )

    def test_balanced_v3_batch_replays_all_old_shapers_in_both_languages(self):
        for step in (1,2):
            images,tokens=balanced_training_batch(step,12,seed=42)
            self.assertEqual(images.shape,(12,3,32,32))
            vocab=vocabulary_for("v2")
            observed=set()
            english=arabic=0
            for item in tokens:
                names={vocab[x] for x in item.tolist() if x!=0}
                shape=next((s for s in ("circle","square","triangle")
                           if s in names),None)
                arabic_shape=next((s for s in ("دائرة","مربع","مثلث")
                                  if s in names),None)
                self.assertTrue(bool(shape) != bool(arabic_shape))
                english+=shape is not None
                arabic+=arabic_shape is not None
            self.assertEqual((english,arabic),(6,6))
        # The same 12 concepts alternate EN/AR on consecutive steps.
        _,first=balanced_training_batch(1,12,seed=42)
        _,second=balanced_training_batch(2,12,seed=42)
        for left,right in zip(first,second):
            self.assertFalse(torch.equal(left,right))

    def test_swap_only_shape_word_with_both_languages_and_colors(self):
        _,tokens=balanced_training_batch(1,12,seed=42)
        swapped=swapped_shape_tokens(tokens)
        self.assertTrue((tokens!=swapped).sum(dim=1).eq(1).all())
        self.assertTrue((tokens!=swapped).any())
        self.assertTrue((tokens!=swapped).all(dim=1).logical_not().all())
        bad=tokens.clone()
        bad[:,:]=0
        with self.assertRaises(ValueError):
            swapped_shape_tokens(bad)
        with self.assertRaises(ValueError):
            swapped_shape_tokens(tokens.float())

    def test_high_noise_is_valid_same_xt_and_gradients_flow(self):
        model=self.new_model()
        images,tokens=balanced_training_batch(1,12,seed=42)
        original=model.forward
        seen=[]
        def spy(noisy,timestep,word_tokens):
            seen.append((noisy.detach().clone(),
                         timestep.detach().clone(),
                         word_tokens.detach().clone()))
            return original(noisy,timestep,word_tokens)
        with patch.object(model,"forward",side_effect=spy):
            result=high_noise_v3_objective(model,images,tokens)
        self.assertEqual(len(seen),1)
        pixels,steps,text=seen[0]
        self.assertTrue(torch.equal(pixels[:12],pixels[12:]))
        self.assertTrue(torch.equal(steps[:12],steps[12:]))
        self.assertTrue(torch.equal(text[:12],tokens))
        self.assertTrue(torch.equal(text[12:],swapped_shape_tokens(tokens)))
        self.assertGreaterEqual(int(steps.min()),4)
        self.assertLessEqual(int(steps.max()),6)
        self.assertTrue(torch.isfinite(result["total_loss"]))
        result["total_loss"].backward()
        self.assertGreater(float(model.input.weight.grad.abs().sum()),0)
        for item in ("correct_shape_word_win_rate",
                     "correct_shape_word_mse_advantage","contrast_loss"):
            self.assertTrue(torch.isfinite(result[item]))

    def test_v3_rejects_incompatible_recipe_before_writing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=self.source(tmp)
            args=self.train_args(tmp,source)
            for key,value in (
                ("paired_shape_weight",0.02),
                ("triangle_fraction",0.5),
                ("preserve_weight",0),
                ("batch_size",10),
                ("conditioning","pooled"),
                ("init_from_v2",None),
                ("dataset_version","v1"),
                ("v3_margin",float("nan")),
            ):
                modified=self.train_args(tmp,source)
                setattr(modified,key,value)
                with self.assertRaises(ValueError):
                    train(modified)
            self.assertFalse((Path(tmp)/"v3").exists())

    def test_two_step_v3_checkpoint_records_recipe_and_keeps_base_untouched(self):
        torch.manual_seed(23)
        with tempfile.TemporaryDirectory() as tmp:
            source=self.source(tmp)
            original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            new_ckpt=train(self.train_args(tmp,source))
            self.assertTrue(new_ckpt.is_file())
            self.assertNotEqual(source,new_ckpt)
            self.assertEqual(
                original_hash,hashlib.sha256(source.read_bytes()).hexdigest()
            )
            record=torch.load(new_ckpt,map_location="cpu",weights_only=True)
            self.assertEqual(
                record["shape_learning_recipe"],
                "v3-valid-q-high-noise-shape-contrast-v1")
            self.assertEqual(record["v3_high_noise"]["margin"],0.003)
            self.assertEqual(record["preserve_weight"],1.0)
            self.assertEqual(record["paired_shape_weight"],0)
            self.assertIsNone(record["triangle_fraction"])
            restored=load_model(new_ckpt)
            self.assertEqual(restored.config.architecture,
                             "rgb-unet-spatial-text-v2")

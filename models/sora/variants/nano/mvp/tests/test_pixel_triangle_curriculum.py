"""Train-only triangle emphasis and safe v2 fine-tuning regressions."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from src.pixel_dataset_v2 import (
    COLORS, SHAPES, shape_curriculum, balanced_training_batch,
)
from src.pixel_diffusion import (
    PixelConfig, PixelDenoiser, vocabulary_for,
)
from scripts.pixel_diffusion import (
    initialize_v2_from_v2, initialize_v2_from_v1, load_model, train,
)


class PixelTriangleCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create_v2(self, directory, channels=16, steps=8):
        config=PixelConfig(channels=channels,diffusion_steps=steps,
                           dataset_version="v2")
        model=PixelDenoiser(config)
        path=Path(directory)/"source-v2.pt"
        torch.save({
            "schema_version":1,
            "architecture":"rgb-unet-text-v1",
            "config":vars(config),
            "dataset_version":"v2",
            "vocabulary":list(vocabulary_for("v2")),
            "model":model.state_dict(),
            "cfg_drop_prob":0.15,
            "training_steps":2500,
        },path)
        return path,model

    def test_curriculum_balances_colors_and_keeps_old_shapes(self):
        for step in (1,2,3,4):
            labels=shape_curriculum(step,12,0.5)
            self.assertEqual(len(labels),12)
            self.assertEqual(sum(shape=="triangle" for _,shape in labels),6)
            self.assertEqual(sum(shape=="circle" for _,shape in labels),3)
            self.assertEqual(sum(shape=="square" for _,shape in labels),3)
            self.assertTrue(all(color in COLORS and shape in SHAPES
                                for color,shape in labels))
            self.assertEqual(labels,shape_curriculum(step,12,0.5))
            self.assertEqual(len({color for color,shape in labels
                                  if shape=="triangle"}),4)
        a,b=balanced_training_batch(10,12,42,triangle_fraction=0.5)
        c,d=balanced_training_batch(10,12,42,triangle_fraction=0.5)
        self.assertTrue(torch.equal(a,c))
        self.assertTrue(torch.equal(b,d))
        # Triangle word appears in six captions in English or Arabic.
        vocab=vocabulary_for("v2")
        shape_ids={vocab.index("triangle"),vocab.index("مثلث")}
        self.assertEqual(sum(any(int(token) in shape_ids for token in seq)
                             for seq in b),6)

    def test_invalid_curriculum_rejected(self):
        for value in (-0.1,1.1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                shape_curriculum(1,12,value)
        with self.assertRaises(ValueError):
            shape_curriculum(0,12,0.5)
        with self.assertRaises(ValueError):
            shape_curriculum(1,0,0.5)

    def test_v2_finetune_copies_weights_and_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory() as d:
            path,source=self.create_v2(d)
            target=PixelDenoiser(PixelConfig(
                channels=16,diffusion_steps=8,dataset_version="v2"))
            info=initialize_v2_from_v2(target,path)
            self.assertEqual(info["source_dataset_version"],"v2")
            self.assertEqual(info["source_training_steps"],2500)
            self.assertTrue(torch.equal(source.input.weight,target.input.weight))
            wrong=PixelDenoiser(PixelConfig(
                channels=32,diffusion_steps=8,dataset_version="v2"))
            with self.assertRaises(ValueError):
                initialize_v2_from_v2(wrong,path)
            wrong=PixelDenoiser(PixelConfig(
                channels=16,diffusion_steps=8,dataset_version="v1"))
            with self.assertRaises(ValueError):
                initialize_v2_from_v2(wrong,path)
            with self.assertRaises(ValueError):
                initialize_v2_from_v1(target,path)

    def test_short_finetuning_keeps_source_and_records_curriculum(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source,_=self.create_v2(root)
            digest_before=hashlib.sha256(source.read_bytes()).hexdigest()
            args=SimpleNamespace(
                steps=2,batch_size=12,learning_rate=0.0001,seed=42,
                channels=16,cfg_drop_prob=0.15,
                dataset_version="v2",diffusion_steps=8,
                device="cpu",cpu_threads=1,log_every=1,
                output_root=str(root/"fine-tuned"),
                init_from_v1=None,init_from_v2=str(source),
                triangle_fraction=0.5,
            )
            new_path=train(args)
            self.assertTrue(new_path.is_file())
            self.assertNotEqual(new_path,source)
            self.assertEqual(digest_before,
                             hashlib.sha256(source.read_bytes()).hexdigest())
            record=torch.load(new_path,map_location="cpu",weights_only=True)
            self.assertEqual(record["triangle_fraction"],0.5)
            self.assertEqual(record["warm_start"]["source_checkpoint"],
                             str(source))
            self.assertEqual(record["warm_start"]["source_training_steps"],
                             2500)
            restored=load_model(new_path)
            self.assertEqual(restored.config.dataset_version,"v2")
            self.assertTrue(torch.isfinite(restored.input.weight).all())

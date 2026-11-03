"""Safety, gradients, and complete toy-domain shape-field-v2 integration."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.pixel_dataset_v2 import balanced_training_batch,synthetic_example_v2
from src.pixel_diffusion import tokenize
from src.shape_first import mask_from_rgb
import src.shape_field_v2 as core
from src.shape_field_v2 import (
    ShapeFieldConfig,NeuralMaskField,NeuralColorizer,
    layout_from_mask,draw_layout_from_noise,keyword_ids,
    geometry_objective,color_objective,generate,
)
from scripts.shape_field_v2 import train,load_run,sample,evaluate
from scripts.pixel_shape_quality import audit_gallery


class ShapeFieldV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return ShapeFieldConfig(hidden=16,word_dim=8)

    def args(self,path):
        return SimpleNamespace(
            mask_steps=2,color_steps=2,batch_size=12,
            hidden=16,word_dim=8,learning_rate=0.0008,
            seed=42,device="cpu",cpu_threads=1,
            log_every=1,output_root=str(path),
        )

    def test_heldout_shapes_have_valid_bboxes_and_finite_coordinates(self):
        shapes=[]
        for shape in ("circle","square","triangle"):
            image,_,_,_=synthetic_example_v2(
                42,color="yellow",shape=shape,
                language="ar",split="validation",variant=0)
            shapes.append(image)
        masks=mask_from_rgb(torch.stack(shapes))
        layout=layout_from_mask(masks)
        self.assertEqual(layout.shape,(3,5))
        coords=core.coordinate_features(layout)
        self.assertEqual(coords.shape,(3,8,32,32))
        self.assertTrue(torch.isfinite(coords).all())
        self.assertTrue((layout[:,2:4]>1).all())
        random_a=draw_layout_from_noise(3,42)
        random_b=draw_layout_from_noise(3,42)
        self.assertTrue(torch.equal(random_a,random_b))
        self.assertFalse(torch.equal(random_a,draw_layout_from_noise(3,41)))
        self.assertTrue((random_a[:,2:4]>1).all())
        with self.assertRaises(ValueError):
            layout_from_mask(torch.zeros(2,1,32,32))

    def test_semantic_keywords_are_separate_learned_english_arabic_tokens(self):
        prompts=("blue circle","دائرة زرقاء","yellow triangle","مثلث أصفر")
        tokens=torch.stack([tokenize(prompt,dataset_version="v2") for prompt in prompts])
        shapes=keyword_ids(tokens,core.SHAPE_WORDS)
        colors=keyword_ids(tokens,core.COLOR_WORDS)
        self.assertEqual(shapes.shape,(4,))
        self.assertEqual(colors.shape,(4,))
        self.assertNotEqual(shapes[0].item(),shapes[1].item())
        self.assertNotEqual(colors[2].item(),colors[3].item())
        self.assertNotEqual(shapes[0].item(),colors[0].item())
        with self.assertRaises(ValueError):
            keyword_ids(torch.zeros(4,12,dtype=torch.long),core.SHAPE_WORDS)
        with self.assertRaises(ValueError):
            keyword_ids(tokens.float(),core.COLOR_WORDS)

    def test_mask_gradient_flows_to_shape_word_and_geometry_network(self):
        torch.manual_seed(29)
        model=NeuralMaskField(self.config())
        images,tokens=balanced_training_batch(1,12,42)
        result=geometry_objective(model,images,tokens)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertGreater(float(model.shape_words.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()),0)
        model.eval()
        with torch.no_grad():
            mask,soft,layout=model.sample(tokens[:2],seed=42)
        self.assertEqual(mask.shape,(2,1,32,32))
        self.assertEqual(soft.shape,mask.shape)
        self.assertTrue(torch.isfinite(soft).all())
        self.assertEqual(layout.shape,(2,5))

    def test_color_gradient_flows_to_word_embeddings_and_neural_foreground(self):
        torch.manual_seed(22)
        network=NeuralColorizer(self.config())
        rgb,tokens=balanced_training_batch(2,12,42)
        objective=color_objective(network,rgb,tokens)
        self.assertTrue(torch.isfinite(objective["loss"]))
        objective["loss"].backward()
        self.assertGreater(float(network.colors.weight.grad.abs().sum()),0)
        self.assertGreater(float(network.foreground[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(network.background.grad.abs().sum()),0)
        output=network(mask_from_rgb(rgb),tokens)
        self.assertEqual(output.shape,(12,3,32,32))
        self.assertTrue(torch.isfinite(output).all())

    def test_inference_no_PIL_shape_drawing_and_deterministic(self):
        torch.manual_seed(33)
        mask=NeuralMaskField(self.config())
        rgb=NeuralColorizer(self.config())
        with patch("src.pixel_dataset_v2.synthetic_example_v2",
                   side_effect=AssertionError("training-only shape drawing used")):
            first,hard,soft=generate(mask,rgb,"yellow triangle",42)
            same,hard2,soft2=generate(mask,rgb,"yellow triangle",42)
            ar,_,_=generate(mask,rgb,"مثلث أصفر",42)
        self.assertTrue(torch.equal(first,same))
        self.assertTrue(torch.equal(hard,hard2))
        self.assertTrue(torch.equal(soft,soft2))
        self.assertEqual(first.shape,(3,32,32))
        self.assertTrue(torch.isfinite(ar).all())
        self.assertFalse(hasattr(core,"ImageDraw"))
        with self.assertRaises(ValueError):
            generate(mask,rgb,"a realistic golden cow",42)

    def test_new_run_train_sample_eval_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run=train(self.args(root/"training"))
            self.assertTrue((run/"mask_field.pt").is_file())
            self.assertTrue((run/"colorizer.pt").is_file())
            self.assertTrue((run/"run.json").is_file())
            manifest=json.loads((run/"run.json").read_text())
            self.assertEqual(manifest["model_family"],"shape-field-v2")
            self.assertEqual(manifest["release_status"],"not-approved")
            network,colors,metadata=load_run(run)
            self.assertEqual(network.config,colors.config)
            one=sample(run,"yellow triangle",42,root/"sample1.png")
            two=sample(run,"yellow triangle",42,root/"sample2.png")
            self.assertEqual(Path(one["image"]).read_bytes(),
                             Path(two["image"]).read_bytes())
            self.assertTrue(Path(one["learned_mask"]).is_file())
            report=evaluate(run,seed=42,output_dir=root/"eval")
            self.assertEqual(len(report["items"]),24)
            self.assertTrue((root/"eval"/"preview_grid.png").is_file())
            self.assertTrue((root/"eval"/"learned_masks_preview_grid.png").is_file())
            self.assertEqual(len(list((root/"eval"/"learned_masks").glob("*.png"))),24)
            quality=audit_gallery(root/"eval")
            self.assertEqual(quality["sample_count"],24)
            weight=run/"mask_field.pt"
            original=weight.read_bytes()
            weight.write_bytes(original+b"corrupted")
            with self.assertRaisesRegex(ValueError,"corrupt"):
                load_run(run)
            weight.write_bytes(original)
            self.assertEqual(hashlib.sha256(original).hexdigest(),
                             manifest["sha256"]["mask_field.pt"])

    def test_invalid_recipe_does_not_create_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"new-run"
            for name,value in (
                ("mask_steps",0),("color_steps",0),("batch_size",13),
                ("learning_rate",0),("learning_rate",float("nan")),
                ("hidden",20),("cpu_threads",0),
            ):
                args=self.args(output)
                setattr(args,name,value)
                with self.assertRaises(ValueError):
                    train(args)
            self.assertFalse(output.exists())
            with self.assertRaises(FileNotFoundError):
                load_run(output)

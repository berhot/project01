"""V2.1 opt-in paired-shape objective and read-only shape-word diagnosis."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.shape_field_v2 import (
    ShapeFieldConfig,NeuralMaskField,SHAPE_WORDS,COLOR_WORDS,keyword_ids
)
from src.shape_field_v21 import (
    matched_layout_shape_batch,shape_word_separation_objective
)
from scripts.shape_field_v2 import train,load_run
from scripts.shape_field_word_audit import audit_words,binary_iou


class ShapeFieldV21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def recipe(self,output_root):
        return SimpleNamespace(
            mask_steps=2,color_steps=2,batch_size=12,
            hidden=16,word_dim=8,learning_rate=0.0008,
            seed=42,device="cpu",cpu_threads=1,
            log_every=1,output_root=str(output_root),
            shape_v21=True,pair_weight=1.0,
            rank_weight=0.1,rank_margin=0.03,
            paired_layout_count=1,
        )

    def test_paired_targets_share_exact_layout_and_all_three_distinct_masks(self):
        layout,truth,tokens,labels=matched_layout_shape_batch(
            1,2,seed=42)
        self.assertEqual(layout.shape,(12,5))
        self.assertEqual(truth.shape,(12,1,32,32))
        self.assertEqual(tokens.shape,(12,12))
        self.assertEqual(len(labels),12)
        self.assertEqual({shape for color,shape,lang in labels},
                         {"circle","square","triangle"})
        for start in (0,6):
            for index in range(start+1,start+6):
                self.assertTrue(torch.equal(layout[start],layout[index]))
            self.assertTrue(torch.equal(truth[start],truth[start+1]))
            self.assertTrue(torch.equal(truth[start+2],truth[start+3]))
            self.assertTrue(torch.equal(truth[start+4],truth[start+5]))
            self.assertFalse(torch.equal(truth[start],truth[start+2]))
            self.assertFalse(torch.equal(truth[start+2],truth[start+4]))
            self.assertEqual(
                {color for color,shape,language in labels[start:start+6]},
                {labels[start][0]}
            )
            self.assertEqual(
                {language for color,shape,language in labels[start:start+6]},
                {"en","ar"}
            )
            self.assertTrue(torch.all(
                keyword_ids(tokens[start:start+6],SHAPE_WORDS)[::2] !=
                keyword_ids(tokens[start:start+6],SHAPE_WORDS)[1::2]
            ))
        with self.assertRaises(ValueError):
            matched_layout_shape_batch(0,1)
        with self.assertRaises(ValueError):
            matched_layout_shape_batch(1,33)

    def test_correct_vs_wrong_shape_word_supervision_and_gradient(self):
        torch.manual_seed(12)
        mask_model=NeuralMaskField(ShapeFieldConfig(hidden=16,word_dim=8))
        layouts,targets,tokens,_=matched_layout_shape_batch(1,2,seed=42)
        results=shape_word_separation_objective(
            mask_model,layouts,targets,tokens)
        self.assertTrue(torch.isfinite(results["loss"]))
        self.assertGreater(float(results["reconstruction_loss"]),0)
        self.assertGreaterEqual(float(results["shape_rank_loss"]),0)
        self.assertTrue(torch.equal(
            keyword_ids(results["wrong_tokens"],COLOR_WORDS),
            keyword_ids(tokens,COLOR_WORDS)))
        self.assertTrue(torch.all(
            keyword_ids(results["wrong_tokens"],SHAPE_WORDS)!=
            keyword_ids(tokens,SHAPE_WORDS)
        ))
        results["loss"].backward()
        self.assertGreater(float(mask_model.shape_words.weight.grad.abs().sum()),0)
        self.assertGreater(float(mask_model.net[0].weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            shape_word_separation_objective(
                mask_model,layouts,targets[:3],tokens)
        with self.assertRaises(ValueError):
            shape_word_separation_objective(
                mask_model,layouts,targets,tokens,ranking_weight=float("nan"))

    def test_same_layout_word_swap_audit_never_assumes_separation(self):
        empty=torch.zeros(1,1,32,32)
        square=torch.zeros_like(empty)
        square[:,:,4:20,4:20]=1
        self.assertEqual(binary_iou(empty,empty),1.0)
        self.assertEqual(binary_iou(empty,square),0.0)
        self.assertEqual(binary_iou(square,square),1.0)

    def test_v21_creates_separate_immutable_checkpoint_and_readonly_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run=train(self.recipe(root/"training"))
            self.assertTrue(run.name.startswith("shape-field-v21-"))
            manifest=json.loads((run/"run.json").read_text())
            self.assertEqual(manifest["model_family"],"shape-field-v2")
            self.assertEqual(manifest["training_recipe"],"shape-word-matched-layout-v2.1")
            self.assertEqual(manifest["paired_shape_objective"]["pair_weight"],1.0)
            self.assertEqual(manifest["release_status"],"not-approved")
            files=(run/"mask_field.pt",run/"colorizer.pt",run/"run.json")
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files}
            models=load_run(run)
            self.assertEqual(len(models),3)
            audit=audit_words(run,seed=42)
            self.assertEqual(audit["sample_count"],24)
            self.assertEqual(len(audit["items"]),8)
            self.assertFalse(audit["release_approved"])
            self.assertTrue(audit["no_training"])
            for shape in ("circle","square","triangle"):
                self.assertEqual(audit["by_shape"][shape]["count"],8)
            for record in audit["items"]:
                self.assertEqual(len(record["prompts"]),3)
                self.assertEqual(len(record["pair_mask_iou"]),3)
            after={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in files}
            self.assertEqual(before,after)

    def test_v21_rejects_bad_paired_parameters_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/"runs"
            for name,value in (
                ("pair_weight",0),("pair_weight",float("nan")),
                ("rank_weight",-1),("rank_margin",float("inf")),
                ("paired_layout_count",0),("paired_layout_count",33),
            ):
                args=self.recipe(destination)
                setattr(args,name,value)
                with self.assertRaises(ValueError):
                    train(args)
            self.assertFalse(destination.exists())

    def test_original_v2_opt_out_does_not_use_v21_checkpoint_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.recipe(Path(temp)/"runs")
            args.shape_v21=False
            run=train(args)
            self.assertTrue(run.name.startswith("shape-field-v2-"))
            manifest=json.loads((run/"run.json").read_text())
            self.assertEqual(manifest["training_recipe"],"original-v2")
            self.assertIsNone(manifest["paired_shape_objective"])
            self.assertFalse(audit_words(run)["release_approved"])

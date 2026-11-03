"""Migration experiment tests: preserve v1 and learn the v2 vocabulary."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.pixel_diffusion import PixelConfig, PixelDenoiser, vocabulary_for
from scripts.pixel_diffusion import initialize_v2_from_v1, load_model, train
from scripts.pixel_checkpoints import list_checkpoints, select_latest


class V1ToV2WarmstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create_v1(self,root,channels=16,diffusion_steps=8):
        config=PixelConfig(channels=channels,diffusion_steps=diffusion_steps)
        model=PixelDenoiser(config)
        checkpoint=root/"pixel-20261101T010101Z-first"/"checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({
            "schema_version":1,"architecture":"rgb-unet-text-v1",
            "config":vars(config),
            "vocabulary":list(vocabulary_for("v1")),
            "cfg_drop_prob":0.15,"training_steps":10000,
            "model":model.state_dict(),
        },checkpoint)
        return checkpoint,model

    def test_copy_existing_weights_and_interpolate_new_token_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path,original=self.create_v1(root)
            upgraded=PixelDenoiser(
                PixelConfig(dataset_version="v2",channels=16,diffusion_steps=8)
            )
            manifest=initialize_v2_from_v1(upgraded,path)
            self.assertEqual(manifest["source_training_steps"],10000)
            self.assertTrue(torch.equal(original.input.weight,upgraded.input.weight))
            self.assertTrue(torch.equal(
                upgraded.text.token.weight[:len(vocabulary_for("v1"))],
                original.text.token.weight
            ))
            old=vocabulary_for("v1")
            new=vocabulary_for("v2")
            yellow_expected=(
                original.text.token.weight[old.index("red")]
                +original.text.token.weight[old.index("green")]
            )/2
            self.assertTrue(torch.allclose(
                upgraded.text.token.weight[new.index("yellow")],yellow_expected
            ))
            self.assertTrue(torch.isfinite(upgraded.output[-1].weight).all())

    def test_invalid_source_architecture_and_v2_to_v2_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_=self.create_v1(Path(directory))
            wrong_channels=PixelDenoiser(
                PixelConfig(dataset_version="v2",channels=32,diffusion_steps=8)
            )
            with self.assertRaises(ValueError):
                initialize_v2_from_v1(wrong_channels,path)
            wrong_target=PixelDenoiser(
                PixelConfig(dataset_version="v1",channels=16,diffusion_steps=8)
            )
            with self.assertRaises(ValueError):
                initialize_v2_from_v1(wrong_target,path)

    def test_checkpoint_listing_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,_=self.create_v1(root)
            records=list_checkpoints(root)
            self.assertEqual(records[0]["training_steps"],10000)
            selected=select_latest(records,"v1",min(10000,10000))
            self.assertEqual(selected["checkpoint"],str(source))
            with self.assertRaises(FileNotFoundError):
                select_latest(records,"v2")

    def test_short_warmstarted_v2_train_saves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,_=self.create_v1(root)
            args=SimpleNamespace(
                steps=2,batch_size=12,learning_rate=0.0001,seed=42,
                channels=16,cfg_drop_prob=0.15,
                dataset_version="v2",diffusion_steps=8,
                device="cpu",cpu_threads=1,log_every=1,
                output_root=str(root/"new-models"),init_from_v1=str(source),
            )
            checkpoint=train(args)
            loaded=load_model(checkpoint)
            self.assertEqual(loaded.config.dataset_version,"v2")
            record=torch.load(checkpoint,map_location="cpu",weights_only=True)
            self.assertEqual(record["warm_start"]["source_checkpoint"],str(source))
            self.assertEqual(record["training_steps"],2)
            self.assertEqual(record["cfg_drop_prob"],0.15)
            self.assertTrue(source.is_file())

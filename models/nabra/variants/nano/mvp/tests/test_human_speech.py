"""Optional offline integration on hash-pinned, human-recorded Google FLEURS clips."""
import contextlib
import io
import json
import unittest
import torch
from src.fixtures import FIXTURE_ROOT,fixture_manifests
from src.ingestion import inspect_split,write_manifest,sha256_file
from src.evaluator import evaluate_checkpoint
from src.inference import load_checkpoint,load_release,transcribe
from src.packaging import package_release
from src.tts import trainer as tts_trainer,packaging as tts_packaging
from src.tts.evaluator import evaluate as tts_evaluate
from src.tts.inference import synthesize
from tests.helpers import Sandbox
from unittest.mock import patch


@unittest.skipUnless((FIXTURE_ROOT/'provenance.json').is_file(),'Optional human fixtures absent: python -m scripts.fetch_test_speech')
class HumanSpeechIntegration(Sandbox):
    def setUp(self):
        super().setUp();self.metadata=fixture_manifests()
        self.train_root=FIXTURE_ROOT/'train';self.val_root=FIXTURE_ROOT/'val'
    def test_human_asr_training_release_inference(self):
        path=self.train_once()
        result,report=evaluate_checkpoint(path,'cpu')
        directory=package_release(path,'0.0.1-smoke',max_cer=result['cer'],approve=True)
        a=load_checkpoint(path,'cpu');b=load_release('0.0.1-smoke','cpu')
        audio=next(self.val_root.glob('*.wav'))
        self.assertEqual(transcribe(*a,audio),transcribe(*b,audio))
        self.assertEqual(json.loads(report.read_text())['checkpoint_sha256'],sha256_file(path))
    def test_human_tts_training_release_generation(self):
        # Preserve provenance, including real source/license and unknown speaker IDs.
        for split,directory in [('train',self.train_root),('val',self.val_root)]:
            records=inspect_split(directory)
            for record in records:
                record.update(speaker_id=None,provenance={'source_type':'human-speech','dataset':'google/fleurs',
                              'revision':self.metadata['revision'],'license':'CC-BY-4.0','partition':'engineering-fixture-only'})
            write_manifest(records,self.root/(split+'.jsonl'))
        with patch.object(tts_trainer,'TTS_RUN_ROOT',self.root/'tts-runs'),patch.object(tts_packaging,'TTS_RELEASE_ROOT',self.root/'tts-releases'):
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint=tts_trainer.train(self.root/'train.jsonl',self.root/'val.jsonl',steps=1,
                    config={'schema_version':1,'acoustic':{'hidden_size':8},'vocoder':{'channels':8}},device_name='cpu')
            report,path=tts_evaluate(checkpoint)
            tts_packaging.package(checkpoint,'0.0.1-smoke',path,report['metrics']['mel_l1'],report['metrics']['spectral_loss'],True)
            bundle=tts_packaging.load_release('0.0.1-smoke')
            waveform,info=synthesize(bundle,'مرحبا بكم في نبْرة للذكاء الاصطناعي',max_frames=40)
            self.assertTrue(torch.isfinite(waveform).all());self.assertGreater(len(waveform),0)
            self.assertFalse(report['production_ready'])

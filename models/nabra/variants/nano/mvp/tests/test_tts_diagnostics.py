"""Quality diagnostics test mechanics, never certify audible speech."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import torch
from src.features import read_wav
from src.ingestion import sha256_file
from src.tts import diagnostics as d
from src.tts.acoustic_model import AcousticConfig, AcousticModel, acoustic_loss
from src.tts.dataset import MelExtractor, collate_tts
from src.tts.vocoder import Vocoder, VocoderConfig
from tests.helpers import Sandbox, pair


class DiagnosticTests(Sandbox):
    def setUp(self):
        super().setUp()
        self.audio = self.train_root/'a.wav'
        self.record,self.tokenizer,self.item=d.pair_data(self.audio)
        self.patcher=patch.object(d,'TTS_RUN_ROOT',self.root/'diagnostics')
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def train_diagnostic(self,stage='vocoder',**kwargs):
        options=kwargs.pop('model_options', {'channels':8} if stage=='vocoder' else {'hidden_size':8})
        with contextlib.redirect_stdout(io.StringIO()),patch.object(d,'save_plots'):
            return d.train(stage,self.audio,steps=kwargs.pop('steps',1),model_options=options,**kwargs)

    def test_ground_truth_features_and_differentiability(self):
        source=read_wav(self.audio)
        self.assertTrue(torch.equal(self.item['mel'],MelExtractor()(source)))
        source.requires_grad_();d.mel_from_wave(source).square().mean().backward()
        self.assertTrue(torch.isfinite(source.grad).all());self.assertGreater(float(source.grad.abs().sum()),0)
        self.assertEqual(self.item['mel'].shape[1],80)
        self.assertEqual(Vocoder(VocoderConfig(channels=8))(self.item['mel'][None]).shape[1],len(self.item['mel'])*160)

    def test_residual_vocoder_gradients_and_legacy_config(self):
        legacy=Vocoder(VocoderConfig(**{'channels':8,'n_mels':80}))
        self.assertFalse(legacy.config.residual)
        model=Vocoder(VocoderConfig(channels=16,min_channels=8,residual=True))
        condition=self.item['mel'][None]
        output=model(condition)
        self.assertEqual(output.shape[-1],condition.shape[1]*160)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        clone=Vocoder(model.config);clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(model(condition),clone(condition)))

    def test_spectral_head_and_phase_objective(self):
        path=self.train_diagnostic(model_options={'channels':16,'head':'spectral'},objective='phase')
        model,ck=d.restore(path)
        self.assertEqual(ck['objective'],'phase')
        condition=self.item['mel'][None]
        output=model(condition)
        self.assertEqual(output.shape[-1],condition.shape[1]*160)
        self.assertTrue(torch.equal(output,d.restore(path)[0](condition)))
        output.square().mean().backward()
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()),0)
        with torch.no_grad():model.network[-1].weight.zero_();model.network[-1].bias.zero_()
        self.assertEqual(float(model(condition).detach().abs().max()),0)

    def test_frozen_vocoder_refinement(self):
        v=self.train_diagnostic();a=self.train_diagnostic('acoustic');digest=sha256_file(v)
        result=self.train_diagnostic('acoustic',resume=a,steps=2,vocoder_checkpoint=v)
        model,ck=d.restore(result)
        self.assertEqual(ck['vocoder_sha256'],digest);self.assertEqual(sha256_file(v),digest)
        row=json.loads((result.parent.parent/'logs/metrics.jsonl').read_text().splitlines()[0])
        self.assertGreater(row['parameter_update_l2'],0)
        self.assertTrue(__import__('math').isfinite(row['weighted_waveform_relative_mse']))
        with self.assertRaisesRegex(ValueError,'frozen vocoder mismatch'):
            self.train_diagnostic('acoustic',resume=result,steps=3)

    def test_attention_masks(self):
        pair(self.train_root,'short','ا',seconds=0.7,frequency=370)
        _,_,short=d.pair_data(self.train_root/'short.wav')
        batch=collate_tts([self.item,short]);model=AcousticModel(AcousticConfig(hidden_size=8))
        _,_,attention=model(batch['tokens'],batch['text_lengths'],batch['speaker'],batch['mel'])
        self.assertTrue(torch.allclose(attention.sum(-1),torch.ones_like(attention.sum(-1))))
        for i,n in enumerate(batch['text_lengths']):self.assertEqual(float(attention[i,:,n:].detach().abs().sum()),0)

    def test_stop_and_free_running_equivalence(self):
        model=AcousticModel(AcousticConfig(hidden_size=8)).eval();batch=collate_tts([self.item])
        with torch.no_grad():model.stop.weight.zero_();model.stop.bias.fill_(-100)
        free,_,_=model(batch['tokens'],batch['text_lengths'],batch['speaker'],batch['mel'],teacher_forcing_ratio=0)
        generated,info=model.generate(batch['tokens'],max_frames=len(self.item['mel']),min_frames=1,return_attention=True)
        self.assertTrue(torch.equal(free,generated));self.assertTrue(info['frame_limit_reached'])
        with torch.no_grad():model.stop.bias.fill_(100)
        generated,info=model.generate(batch['tokens'],max_frames=100,min_frames=8)
        self.assertEqual(generated.shape[1],8);self.assertTrue(info['stop_predicted'])
        with self.assertRaises(ValueError):model(*[batch[k] for k in ('tokens','text_lengths','speaker','mel')],teacher_forcing_ratio=2)

    def test_teacher_forced_and_free_gradients(self):
        model=AcousticModel(AcousticConfig(hidden_size=8));batch=collate_tts([self.item])
        a,_,_=model(batch['tokens'],batch['text_lengths'],batch['speaker'],batch['mel'])
        b,_,_=model(batch['tokens'],batch['text_lengths'],batch['speaker'],batch['mel'],teacher_forcing_ratio=0)
        self.assertFalse(torch.equal(a,b))
        loss,_=acoustic_loss(model,batch,teacher_forcing_ratio=0);loss.backward()
        self.assertGreater(float(model.mel.weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_vocoder_checkpoint_resume_and_pcm(self):
        digest=sha256_file(self.audio);first=self.train_diagnostic();second=self.train_diagnostic(resume=first,steps=2)
        model,ck=d.restore(second);previous=torch.load(first,weights_only=True)
        self.assertEqual(ck['step'],2)
        uninterrupted=self.train_diagnostic(steps=2)
        for k,value in d.restore(uninterrupted)[0].state_dict().items():
            self.assertTrue(torch.equal(value,model.state_dict()[k]))
        key=next(iter(ck['optimizer']['state']))
        self.assertGreater(ck['optimizer']['state'][key]['step'],previous['optimizer']['state'][key]['step'])
        self.assertEqual(sha256_file(self.audio),digest)
        output=second.parent.parent/'samples/vocoder_reconstruction.wav'
        self.assertEqual(len(read_wav(output)),len(self.item['waveform']))
        with torch.inference_mode():
            a=model(self.item['mel'][None]);b=d.restore(second)[0](self.item['mel'][None])
        self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):self.train_diagnostic(resume=second,steps=2)
        bad=dict(ck);bad['record']={**ck['record'],'audio_sha256':'bad'};path=self.root/'bad.pt';torch.save(bad,path)
        with self.assertRaisesRegex(ValueError,'data mismatch'):self.train_diagnostic(resume=path,steps=3)
        row=json.loads((second.parent.parent/'logs/metrics.jsonl').read_text().splitlines()[0])
        self.assertGreater(row['parameter_update_l2'],0);self.assertGreater(row['gradient_norm'],0)

    def test_full_generation_needs_both_weights(self):
        v=self.train_diagnostic();a=self.train_diagnostic('acoustic')
        with contextlib.redirect_stdout(io.StringIO()):metrics=d.full(a,v,self.root/'full')
        self.assertEqual(metrics['intelligibility'],'PENDING HUMAN REVIEW');self.assertFalse(metrics['production_ready'])
        self.assertTrue(metrics['deterministic']);self.assertTrue(metrics['distinct_text_outputs'])
        from src.tts.inference import write_wav
        import wave
        with wave.open(str(self.root/'full/tts_full_overfit.wav')) as audio:
            self.assertEqual((audio.getframerate(),audio.getnchannels(),audio.getsampwidth()),(16000,1,2))
        acoustic,ck=d.restore(a);vocoder,_=d.restore(v)
        tokens=self.item['tokens'][None]
        with torch.inference_mode():
            mel,_=acoustic.generate(tokens,max_frames=24);base=vocoder(mel)
            acoustic.mel.bias.add_(0.3);changed,_=acoustic.generate(tokens,max_frames=24)
            self.assertFalse(torch.equal(base,vocoder(changed)))
            vocoder.network[-2].bias.add_(0.1);self.assertFalse(torch.equal(base,vocoder(mel)))
        with self.assertRaises(FileExistsError):write_wav(self.root/'full/tts_full_overfit.wav',torch.zeros(400))
        with self.assertRaises(ValueError):write_wav(self.root/'invalid.wav',torch.tensor([float('nan')]))

    @unittest.skipUnless(__import__('os').getenv('NABRA_TEST_MPS')=='1' and torch.backends.mps.is_available(),'Optional MPS diagnostics')
    def test_mps_gradient_and_checkpoint(self):
        for stage,options in [('vocoder',{'channels':8}),('acoustic',{'hidden_size':8}),('vocoder',{'channels':16,'head':'spectral'})]:
            with self.subTest(stage=stage,options=options):
                path=self.train_diagnostic(stage,device_name='mps',model_options=options);model,ck=d.restore(path,'mps')
                self.assertEqual(ck['step'],1)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

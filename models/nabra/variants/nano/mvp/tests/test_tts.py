import contextlib
import io
import json
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from tests.helpers import Sandbox,pair,wav
from src.tts.tokenizer import TextTokenizer,normalize
from src.tts.dataset import MelConfig,MelExtractor,TTSDataset,collate_tts,inspect_sources
from src.tts.acoustic_model import AcousticModel,AcousticConfig,acoustic_loss
from src.tts.vocoder import Vocoder,VocoderConfig,reconstruction_loss
from src.tts import trainer,packaging
from src.tts.inference import load_checkpoint,synthesize,write_wav
from src.tts.evaluator import evaluate
from src.features import read_wav


class TTSTests(Sandbox):
    def setUp(self):
        super().setUp()
        for module,name,value in [(trainer,'TTS_RUN_ROOT',self.root/'tts-runs'),(packaging,'TTS_RELEASE_ROOT',self.root/'tts-releases')]:
            patcher=patch.object(module,name,value);patcher.start();self.addCleanup(patcher.stop)
        self.options={'schema_version':1,'acoustic':{'hidden_size':8},'vocoder':{'channels':8}}
    def training(self,**kwargs):
        args=dict(train_source=self.train_root,val_source=self.val_root,steps=1,config=self.options,device_name='cpu')
        args.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):return trainer.train(**args)
    def test_tokenizer_normalization_and_diacritics(self):
        tok=TextTokenizer(); text='  مرحبا   نبْرة '
        self.assertEqual(normalize(text),'مرحبا نبْرة')
        self.assertEqual(tok.encode(text),TextTokenizer.from_dict(tok.to_dict()).encode(text))
        with self.assertRaises(ValueError):tok.encode('🙂')
        with self.assertRaises(ValueError):tok.encode('')
    def test_manifest_silence_and_speakers(self):
        records=inspect_sources(self.train_root,self.val_root)
        self.assertIsNone(records['train'][0]['speaker_id'])
        self.assertIsNone(records['train'][0]['provenance']['license'])
        self.assertGreater(records['train'][0]['rms'],0)
        import wave
        with wave.open(str(self.train_root/'a.wav'),'wb') as w:
            w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes(16000))
        with self.assertRaisesRegex(ValueError,'Silent'):inspect_sources(self.train_root,self.val_root)
    def test_batch_masks_and_acoustic_gradients(self):
        pair(self.train_root,'b','ا',seconds=0.7,frequency=420)
        records=inspect_sources(self.train_root,self.val_root)
        tok=TextTokenizer();ds=TTSDataset(records['train'],tok)
        batch=collate_tts([ds[0],ds[1]])
        model=AcousticModel(AcousticConfig(vocab_size=tok.vocab_size,hidden_size=8))
        loss,metrics=acoustic_loss(model,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(model.mel.weight.grad.abs().sum().item(),0)
        self.assertNotEqual(batch['mel_lengths'][0],batch['mel_lengths'][1])
    def test_vocoder_real_gradients_and_length(self):
        model=Vocoder(VocoderConfig(channels=8))
        mel=torch.randn(1,8,80);output=model(mel)
        self.assertEqual(output.shape,(1,1280))
        loss,_=reconstruction_loss(output,torch.zeros_like(output));loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
    def test_tts_feature_config(self):
        waveform=read_wav(self.train_root/'a.wav')
        mel=MelExtractor()(waveform)
        self.assertEqual(mel.shape,(41,80));self.assertTrue(torch.isfinite(mel).all())
        with self.assertRaises(ValueError):MelConfig(hop_length=80)
    def test_training_checkpoint_resume(self):
        path=self.training();ck=torch.load(path,weights_only=True)
        resumed=self.training(resume=path,steps=2);new=torch.load(resumed,weights_only=True)
        self.assertEqual(new['step'],2)
        key=next(iter(ck['optimizer']['state']))
        self.assertGreater(new['optimizer']['state'][key]['step'],ck['optimizer']['state'][key]['step'])
        for field,value in [('dataset_fingerprints',{}),('tokenizer',{}),('config',{})]:
            broken=self.root/'bad.pt';torch.save({**ck,field:value},broken)
            with self.assertRaises(ValueError):self.training(resume=broken,steps=2)
    def test_actual_synthesis_depends_on_text_and_weights(self):
        path=self.training();bundle=load_checkpoint(path)
        a,_=synthesize(bundle,'مرحبا',max_frames=24);b,_=synthesize(bundle,'العربية',max_frames=24)
        repeat,_=synthesize(bundle,'مرحبا',max_frames=24)
        self.assertTrue(torch.equal(a,repeat));self.assertFalse(torch.equal(a,b))
        with torch.no_grad():bundle[1].network[-2].bias.add_(0.2)
        changed,_=synthesize(bundle,'مرحبا',max_frames=24)
        self.assertFalse(torch.equal(a,changed))
        out=self.root/'generated.wav';write_wav(out,a)
        restored=read_wav(out);self.assertEqual(len(restored),len(a));self.assertTrue(torch.isfinite(restored).all())
        with self.assertRaises(FileExistsError):write_wav(out,a)
    def test_release_lifecycle(self):
        path=self.training();report,report_path=evaluate(path)
        out=packaging.package(path,'0.1.0',report_path,report['metrics']['mel_l1'],report['metrics']['spectral_loss'],True)
        original=load_checkpoint(path);released=packaging.load_release('0.1.0')
        a,_=synthesize(original,'مرحبا',max_frames=24);b,_=synthesize(released,'مرحبا',max_frames=24)
        self.assertTrue(torch.equal(a,b))
        checkpoint=torch.load(path,weights_only=True)
        for component in ('acoustic','vocoder'):
            for k,v in load_file(str(out/(component+'.safetensors'))).items():self.assertTrue(torch.equal(v,checkpoint[component][k]))
        with self.assertRaises(FileExistsError):packaging.package(path,'0.1.0',report_path,100,100,True)
        with self.assertRaises(ValueError):packaging.package(path,'0.2.0',report_path,0,0,True)
        with self.assertRaises(ValueError):packaging.package(path,'0.2.0',report_path,100,100,False)
        bad={**report,'checkpoint_sha256':'stale'};report_path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):packaging.package(path,'0.2.0',report_path,100,100,True)
        (out/'tokenizer.json').write_text('{}')
        with self.assertRaises(ValueError):packaging.load_release('0.1.0')
        (out/'tokenizer.json').unlink()
        with self.assertRaises(FileNotFoundError):packaging.load_release('0.1.0')
        with self.assertRaises(ValueError):packaging.release_path('../escape')
    def test_actual_speech_integration(self):
        from src.paths import TRAIN_ROOT,VAL_ROOT
        if not TRAIN_ROOT.is_dir():self.skipTest('Speech corpus absent')
        # Current corpus is synthetic speech; this is engineering integration, not human-speech validation.
        path=self.training(train_source=TRAIN_ROOT,val_source=VAL_ROOT)
        report,report_path=evaluate(path)
        packaging.package(path,'0.1.0',report_path,report['metrics']['mel_l1'],report['metrics']['spectral_loss'],True)
        bundle=packaging.load_release('0.1.0')
        waveform,info=synthesize(bundle,'مرحبا بكم في نبْرة للذكاء الاصطناعي',max_frames=40)
        self.assertGreater(len(waveform),0);self.assertTrue(torch.isfinite(waveform).all())

    @__import__('unittest').skipUnless(__import__('os').getenv('NABRA_TEST_MPS')=='1' and torch.backends.mps.is_available(), 'Optional MPS TTS test; set NABRA_TEST_MPS=1')
    def test_mps_training_step(self):
        path=self.training(device_name='mps')
        bundle=load_checkpoint(path,'mps')
        waveform,_=synthesize(bundle,'مرحبا',max_frames=24)
        self.assertTrue(torch.isfinite(waveform).all())

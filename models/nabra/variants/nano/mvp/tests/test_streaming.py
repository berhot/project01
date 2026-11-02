import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.streaming import BufferedASR,StreamConfig,WaveRecognizer,select_microphone
from src.model import ModelConfig,NabraASR
from src.features import FeatureConfig,read_wav
from src.inference import transcribe
from src.tokenizer import CharTokenizer
from tests.helpers import Sandbox


class StreamingTests(Sandbox):
    def test_chunk_order(self):
        engine=BufferedASR(lambda x:'اب')
        engine.feed(np.ones(320,dtype=np.float32)*0.1,0)
        with self.assertRaisesRegex(ValueError,'Out-of-order'):engine.feed(np.zeros(320),2)
    def test_buffer_limits_and_final_uniqueness(self):
        engine=BufferedASR(lambda x:'اب',StreamConfig(max_segment_seconds=1,silence_seconds=0.2,partial_seconds=0.3))
        events=[]
        for i in range(40):
            events+=engine.feed(np.ones(1600,dtype=np.float32)*0.1,i)
            self.assertLessEqual(engine.stats()['retained_samples'],16320)
        events+=engine.close()
        finals=[e for e in events if e['kind']=='final']
        self.assertEqual(len(finals),4)
        self.assertEqual(len({e['segment_id'] for e in finals}),4)
        self.assertEqual(engine.close(),[])
        with self.assertRaises(ValueError):engine.feed(np.zeros(1))
    def test_silence_final_and_stabilization(self):
        responses=iter(['مرحبا','مرحبا بكم','مرحبا بكم'])
        engine=BufferedASR(lambda x:next(responses),StreamConfig(partial_seconds=0.3,silence_seconds=0.2))
        events=[]
        for _ in range(6):events+=engine.feed(np.ones(1600,dtype=np.float32)*0.1)
        for _ in range(2):events+=engine.feed(np.zeros(1600,dtype=np.float32))
        self.assertEqual([e['kind'] for e in events],['partial','partial','final'])
        self.assertEqual(events[1]['stable_prefix'],'مرحبا')
        self.assertEqual(len(engine.close()),0)
    def test_chunk_size_independent_buffer(self):
        waveform=np.linspace(0.02,0.2,7777,dtype=np.float32)
        results=[]
        for chunk in (127,1600):
            captured=[];engine=BufferedASR(lambda x:(captured.append(x.copy()) or 'اب'),StreamConfig(partial_seconds=10))
            for start in range(0,len(waveform),chunk):engine.feed(waveform[start:start+chunk])
            engine.close();results.append(captured[0])
        np.testing.assert_array_equal(results[0],results[1]);np.testing.assert_array_equal(results[0],waveform)
    def test_feature_inference_equivalence(self):
        tok=CharTokenizer(['ا','ب',' ']);model=NabraASR(ModelConfig(vocab_size=tok.vocab_size,hidden_size=8,num_layers=1)).eval()
        bundle=(model,tok,FeatureConfig());audio=self.train_root/'a.wav'
        self.assertEqual(WaveRecognizer(bundle)(read_wav(audio).numpy()),transcribe(*bundle,audio))
    def test_stereo_conversion_invalid_input(self):
        engine=BufferedASR(lambda x:'اب')
        engine.feed(np.zeros((320,2),dtype=np.float32))
        with self.assertRaises(ValueError):engine.feed(np.array([np.nan]))
        with self.assertRaises(ValueError):engine.feed(np.zeros(16001))
    def test_microphone_selection(self):
        import sounddevice as sd
        devices=[{'index':3,'name':'Built-in Microphone','channels':1,'default_sample_rate':48000}]
        with patch('src.streaming.input_devices',return_value=devices),patch.object(sd,'check_input_settings'):
            self.assertEqual(select_microphone('Built-in')['index'],3)
            self.assertEqual(select_microphone('3')['index'],3)
            with self.assertRaises(ValueError):select_microphone('missing')
        with patch('src.streaming.input_devices',return_value=[]):
            with self.assertRaises(RuntimeError):select_microphone()

    def test_graceful_microphone_interrupt(self):
        from src.streaming import capture
        import sounddevice as sd
        from unittest.mock import MagicMock
        stream=MagicMock()
        stream.__enter__.side_effect=KeyboardInterrupt
        engine=BufferedASR(lambda x:'اب')
        with patch('src.streaming.select_microphone',return_value={'index':0}),patch.object(sd,'InputStream',return_value=stream):
            stats=capture(engine,seconds=1)
        self.assertTrue(engine.closed)
        self.assertEqual(stats['audio_seconds'],0)

    def test_microphone_open_failure(self):
        from src.streaming import capture
        import sounddevice as sd
        engine=BufferedASR(lambda x:'اب')
        with patch('src.streaming.select_microphone',return_value={'index':0}),patch.object(sd,'InputStream',side_effect=RuntimeError('Unavailable microphone')):
            with self.assertRaisesRegex(RuntimeError,'Unavailable'):capture(engine,seconds=1)
        self.assertTrue(engine.closed)

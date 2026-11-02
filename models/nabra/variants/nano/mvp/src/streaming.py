"""Bounded-segment buffered recognition for the existing bidirectional ASR model.
This is NOT a causal streaming encoder. Partials reprocess only the active utterance.
"""
from dataclasses import dataclass
import json
import math
import queue
import time

import numpy as np
import torch

from .features import LogMelExtractor


@dataclass
class StreamConfig:
    sample_rate:int=16000
    frame_ms:int=20
    max_segment_seconds:float=8.0
    silence_seconds:float=0.6
    partial_seconds:float=0.75
    min_segment_seconds:float=0.25
    rms_threshold:float=0.008
    def __post_init__(self):
        if self.sample_rate!=16000 or self.frame_ms not in (10,20,40):raise ValueError('Unsupported stream format')
        for value in (self.max_segment_seconds,self.silence_seconds,self.partial_seconds,self.min_segment_seconds,self.rms_threshold):
            if not math.isfinite(value) or value<=0:raise ValueError('Invalid stream setting')
        if not self.min_segment_seconds<=self.max_segment_seconds<=20:raise ValueError('Invalid segment duration')
        if self.silence_seconds>=self.max_segment_seconds:raise ValueError('Silence threshold must be shorter than segment')


class WaveRecognizer:
    def __init__(self,bundle):
        self.model,self.tokenizer,self.features=bundle
        self.extractor=LogMelExtractor(self.features)
    @torch.inference_mode()
    def __call__(self,waveform):
        self.model.eval()
        x=self.extractor(torch.from_numpy(waveform.copy()))
        logits,lengths=self.model(x[None].to(next(self.model.parameters()).device),torch.tensor([len(x)]))
        return self.tokenizer.ctc_decode(logits[0,:int(lengths[0])].argmax(-1).cpu().tolist())


class BufferedASR:
    def __init__(self,recognize,config=None):
        self.recognize=recognize;self.config=config or StreamConfig()
        self.pending=np.empty(0,dtype=np.float32)
        self.segment=[];self.segment_samples=0;self.quiet=0;self.last_partial=0
        self.previous='';self.segment_id=0;self.next_sequence=0;self.input_samples=0
        self.processing_seconds=0.0;self.calls=0;self.closed=False
    def _decode(self,final):
        wave=np.concatenate(self.segment)
        started=time.perf_counter();text=self.recognize(wave);elapsed=time.perf_counter()-started
        self.processing_seconds+=elapsed;self.calls+=1
        # Stable prefix means agreement with the previous partial, not irrevocable output.
        before=self.previous.split();after=text.split();common=[]
        for a,b in zip(before,after):
            if a!=b:break
            common.append(a)
        self.previous=text
        return {'segment_id':self.segment_id,'kind':'final' if final else 'partial','text':text,
                'stable_prefix':text if final else ' '.join(common),'audio_seconds':len(wave)/16000,
                'processing_seconds':elapsed,'mode':'buffered-bidirectional','replace_partial':True}
    def _finish(self):
        events=[]
        if self.segment_samples>=int(self.config.min_segment_seconds*16000):events.append(self._decode(True))
        self.segment=[];self.segment_samples=0;self.quiet=0;self.last_partial=0;self.previous='';self.segment_id+=1
        return events
    def _frame(self,frame):
        c=self.config;events=[]
        speech=float(np.sqrt(np.mean(frame**2)))>=c.rms_threshold
        if not self.segment and not speech:return events
        self.segment.append(frame.copy());self.segment_samples+=len(frame)
        self.quiet=0 if speech else self.quiet+len(frame)
        if self.quiet>=int(c.silence_seconds*16000) or self.segment_samples>=int(c.max_segment_seconds*16000):
            return self._finish()
        if self.segment_samples-self.last_partial>=int(c.partial_seconds*16000) and self.segment_samples>=int(c.min_segment_seconds*16000):
            events.append(self._decode(False));self.last_partial=self.segment_samples
        return events
    def feed(self,samples,sequence=None):
        if self.closed:raise ValueError('Stream is closed')
        if sequence is not None and sequence!=self.next_sequence:raise ValueError('Out-of-order audio chunk')
        self.next_sequence+=1
        samples=np.asarray(samples,dtype=np.float32)
        if samples.ndim==2:samples=samples.mean(axis=1)
        if samples.ndim!=1 or not np.isfinite(samples).all():raise ValueError('Invalid microphone samples')
        if len(samples)>16000:raise ValueError('Feed chunks of at most one second')
        self.input_samples+=len(samples)
        data=np.concatenate([self.pending,samples]);size=self.config.sample_rate*self.config.frame_ms//1000
        events=[]
        for start in range(0,len(data)-size+1,size):events.extend(self._frame(data[start:start+size]))
        self.pending=data[(len(data)//size)*size:].copy()
        return events
    def close(self):
        if self.closed:return []
        events=[]
        if len(self.pending):events.extend(self._frame(self.pending));self.pending=np.empty(0,dtype=np.float32)
        events.extend(self._finish());self.closed=True
        return events
    def stats(self):
        duration=self.input_samples/16000
        return {'mode':'buffered-bidirectional','audio_seconds':duration,'processing_seconds':self.processing_seconds,
                'real_time_factor':self.processing_seconds/duration if duration else 0,'decode_calls':self.calls,
                'retained_samples':self.segment_samples+len(self.pending),
                'max_buffer_seconds':self.config.max_segment_seconds+self.config.frame_ms/1000}


def input_devices():
    import sounddevice as sd
    return [{'index':i,'name':d['name'],'channels':d['max_input_channels'],'default_sample_rate':d['default_samplerate']}
            for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0]


def select_microphone(selection=None):
    import sounddevice as sd
    devices=input_devices()
    if not devices:raise RuntimeError('No microphone input devices available')
    if selection is None:
        index=int(sd.default.device[0])
        matches=[d for d in devices if d['index']==index]
    elif str(selection).isdigit():matches=[d for d in devices if d['index']==int(selection)]
    else:matches=[d for d in devices if str(selection).casefold() in d['name'].casefold()]
    if len(matches)!=1:raise ValueError('Microphone selection is missing or ambiguous; use --list-devices and an index')
    sd.check_input_settings(device=matches[0]['index'],channels=1,dtype='float32',samplerate=16000)
    return matches[0]


def capture(engine,selection=None,chunk_ms=100,seconds=None,emit=None):
    """Capture at 16 kHz through PortAudio/CoreAudio; callbacks never run inference."""
    import sounddevice as sd
    if not 20<=chunk_ms<=1000:raise ValueError('chunk-ms must be between 20 and 1000')
    if seconds is not None and (not math.isfinite(seconds) or seconds<=0):raise ValueError('Invalid capture duration')
    device=select_microphone(selection);chunks=queue.Queue(maxsize=32);failure=[];emitted=0
    def callback(indata,frames,timing,status):
        if status:failure.append(str(status));raise sd.CallbackAbort
        try:chunks.put_nowait(indata[:,0].copy())
        except queue.Full:failure.append('Audio queue overflow; inference cannot keep up');raise sd.CallbackAbort
    def dispatch(events):
        nonlocal emitted
        for event in events:
            emitted+=1
            if emit:emit(event)
    start=time.monotonic();sequence=0
    try:
        with sd.InputStream(device=device['index'],samplerate=16000,channels=1,dtype='float32',
                            blocksize=16000*chunk_ms//1000,callback=callback) as stream:
            while seconds is None or time.monotonic()-start<seconds:
                if failure:raise RuntimeError(failure[0])
                try:chunk=chunks.get(timeout=0.25)
                except queue.Empty:
                    if not stream.active:raise RuntimeError('Microphone stream stopped')
                    continue
                dispatch(engine.feed(chunk,sequence));sequence+=1
    except KeyboardInterrupt:
        pass
    finally:
        # The input stream is closed now; finish the bounded queued audio before finalizing.
        while not chunks.empty():
            dispatch(engine.feed(chunks.get_nowait(),sequence));sequence+=1
        dispatch(engine.close())
    return {'device':device,'capture_seconds':time.monotonic()-start,'events':emitted,**engine.stats()}

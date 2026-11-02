from pathlib import Path
import wave

import numpy as np
import torch

from .tokenizer import TextTokenizer
from .dataset import MelConfig
from .acoustic_model import AcousticConfig, AcousticModel
from .vocoder import VocoderConfig, Vocoder


def build(config,tokenizer_data,speakers,acoustic_state,vocoder_state,device='cpu'):
    tokenizer=TextTokenizer.from_dict(tokenizer_data)
    features=MelConfig(**config['features']);a=AcousticConfig(**config['acoustic']);v=VocoderConfig(**config['vocoder'])
    if a.vocab_size!=tokenizer.vocab_size or a.n_mels!=features.n_mels or v.n_mels!=features.n_mels or a.num_speakers!=len(speakers):
        raise ValueError('TTS model/tokenizer/features/speaker mismatch')
    acoustic=AcousticModel(a);vocoder=Vocoder(v)
    acoustic.load_state_dict(acoustic_state,strict=True);vocoder.load_state_dict(vocoder_state,strict=True)
    return acoustic.to(device).eval(),vocoder.to(device).eval(),tokenizer,features,speakers


def load_checkpoint(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('schema_version')!=1 or data.get('task')!='tts':raise ValueError('Unsupported TTS checkpoint')
    return build(data['config'],data['tokenizer'],data['speakers'],data['acoustic'],data['vocoder'],device)


@torch.inference_mode()
def synthesize(bundle,text,speaker=None,max_frames=800):
    acoustic,vocoder,tokenizer,features,speakers=bundle
    if speaker is None:
        if len(speakers)!=1:raise ValueError('Select a speaker for this model')
        index=0
    else:
        if speaker not in speakers:raise ValueError('Unknown speaker')
        index=speakers.index(speaker)
    ids=tokenizer.encode(text)
    if len(ids)>512:raise ValueError('Text too long; split into sentences')
    device=next(acoustic.parameters()).device
    mel,info=acoustic.generate(torch.tensor([ids],device=device),index,max_frames=max_frames,min_frames=min(20,max_frames))
    waveform=vocoder(mel)[0].cpu()
    if waveform.numel()==0 or not torch.isfinite(waveform).all():raise ValueError('Invalid generated waveform')
    return waveform,{**info,'sample_rate':features.sample_rate,'samples':len(waveform),
                    'duration_seconds':len(waveform)/features.sample_rate,'rms':float(waveform.square().mean().sqrt()),
                    'peak':float(waveform.abs().max()),'quality_status':'experimental-unverified-intelligibility'}


def write_wav(path,waveform,sample_rate=16000):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    if waveform.ndim!=1 or waveform.numel()==0 or not torch.isfinite(waveform).all():raise ValueError('Invalid output waveform')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as file:
        with wave.open(file,'wb') as audio:
            audio.setparams((1,2,sample_rate,0,'NONE','not compressed'))
            audio.writeframes((waveform.clamp(-1,1).numpy()*32767).round().astype('<i2').tobytes())

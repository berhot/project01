from dataclasses import dataclass, asdict
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..features import FeatureConfig, create_mel_filterbank, read_wav
from ..ingestion import read_records, validate_splits, sha256_file
from ..runtime import fingerprint
from .tokenizer import normalize


@dataclass
class MelConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    log_scale: float = 8.0
    def __post_init__(self):
        if (self.sample_rate,self.hop_length)!=(16000,160):
            raise ValueError('This vocoder requires 16 kHz / hop 160')
        if self.n_fft<2 or self.n_mels<1 or not 0 < self.log_scale < 100:
            raise ValueError('Invalid TTS feature configuration')


class MelExtractor:
    """Absolute log power, fixed scaling: unlike ASR, retain loudness for synthesis."""
    def __init__(self, config=None):
        self.config=config or MelConfig()
        self.bank=create_mel_filterbank(FeatureConfig(n_fft=self.config.n_fft,n_mels=self.config.n_mels))
    def __call__(self,waveform):
        c=self.config
        if waveform.ndim!=1 or len(waveform)<=c.n_fft//2 or not torch.isfinite(waveform).all():
            raise ValueError('Invalid TTS waveform')
        spectrum=torch.stft(waveform,n_fft=c.n_fft,hop_length=c.hop_length,
                            window=torch.hann_window(c.n_fft,device=waveform.device),return_complex=True)
        mel=self.bank.to(waveform.device) @ spectrum.abs().square()
        return torch.log(mel.clamp_min(1e-5)).T / c.log_scale


def provenance(record):
    """Recover known local generator provenance only when the content hash matches."""
    audio=Path(record['audio'])
    for parent in (audio.parent,audio.parent.parent):
        meta=parent/'synthetic-samples.json'
        if meta.is_file():
            data=json.loads(meta.read_text())
            if any(r.get('audio_sha256')==record['audio_sha256'] for r in data.get('samples',[])):
                return {'source_type':'synthetic-speech','generator':data.get('generator'),'voice':data.get('voice'),'license':None}
    return {'source_type':'unspecified','license':None}


def inspect_sources(train_source,val_source):
    splits={'train':read_records(train_source),'val':read_records(val_source)}
    validate_splits(splits['train'],splits['val'])
    for records in splits.values():
        for record in records:
            wave=read_wav(record['audio'])
            rms=float(wave.square().mean().sqrt())
            silence=float((wave.abs()<1e-4).float().mean())
            if rms<1e-5 or silence>0.98: raise ValueError('Silent or nearly silent TTS recording')
            record.update(normalized_text=normalize(record['transcript']),rms=rms,silence_fraction=silence)
            record.setdefault('speaker_id',None)
            record.setdefault('provenance',provenance(record))
            if not record['normalized_text']: raise ValueError('Empty normalized transcript')
    return splits


def speaker_key(record):
    # Reserved conditioning slot, not a claim about a real speaker's identity.
    value=record.get('speaker_id')
    if value is not None and (not isinstance(value,str) or not value): raise ValueError('Invalid speaker ID')
    return '<unspecified>' if value is None else value


class TTSDataset(Dataset):
    def __init__(self,records,tokenizer,config=None,speakers=None):
        if not records: raise ValueError('Empty TTS dataset')
        self.records=records;self.tokenizer=tokenizer;self.extractor=MelExtractor(config)
        self.speakers=speakers or sorted({speaker_key(r) for r in records})
        self.items=[]
        for record in records:
            key=speaker_key(record)
            if key not in self.speakers: raise ValueError('Unseen speaker in evaluation')
            wave=read_wav(record['audio'])
            self.items.append({'tokens':torch.tensor(tokenizer.encode(record['transcript'])),
                               'mel':self.extractor(wave),'waveform':wave,'speaker':self.speakers.index(key)})
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]


def collate_tts(items):
    return {'tokens':pad_sequence([x['tokens'] for x in items],batch_first=True),
            'text_lengths':torch.tensor([len(x['tokens']) for x in items]),
            'mel':pad_sequence([x['mel'] for x in items],batch_first=True),
            'mel_lengths':torch.tensor([len(x['mel']) for x in items]),
            'speaker':torch.tensor([x['speaker'] for x in items]),
            'waveforms':[x['waveform'] for x in items]}

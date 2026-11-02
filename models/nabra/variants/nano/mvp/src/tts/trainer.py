from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch

from ..paths import BERHOT_ROOT, TRAIN_ROOT, VAL_ROOT, validate_run_id
from ..runtime import select_device, fingerprint, atomic_json, environment
from ..training_control import TrainingControl, accelerator_rng, restore_accelerator_rng
from .tokenizer import TextTokenizer
from .multisentence import sample_index
from .dataset import MelConfig, TTSDataset, collate_tts, inspect_sources, speaker_key
from .acoustic_model import AcousticConfig, AcousticModel, acoustic_loss
from .vocoder import VocoderConfig, Vocoder, reconstruction_loss

TTS_RUN_ROOT=BERHOT_ROOT/'storage/artifacts/nabra/tts/nano/mvp'


def configs(options=None):
    options=options or {}
    if options.get('schema_version',1)!=1:raise ValueError('Unsupported TTS config schema')
    tokenizer=TextTokenizer()
    mel=MelConfig(**options.get('features',{}))
    acoustic=AcousticConfig(**{'vocab_size':tokenizer.vocab_size,'n_mels':mel.n_mels,**options.get('acoustic',{})})
    vocoder=VocoderConfig(**{'n_mels':mel.n_mels,**options.get('vocoder',{})})
    if acoustic.vocab_size!=tokenizer.vocab_size or acoustic.n_mels!=mel.n_mels or vocoder.n_mels!=mel.n_mels:
        raise ValueError('TTS configuration mismatch')
    return tokenizer,mel,acoustic,vocoder


def train(train_source=None,val_source=None,steps=100,learning_rate=0.001,seed=42,device_name='cpu',
          config=None,resume=None,run_id=None,save_interval=25,max_seconds=None):
    if steps<1 or save_interval<1 or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Invalid training arguments')
    random.seed(seed);torch.manual_seed(seed)
    device=select_device(device_name)
    records=inspect_sources(train_source or TRAIN_ROOT,val_source or VAL_ROOT)
    tokenizer,mel_config,acoustic_config,vocoder_config=configs(config)
    speakers=sorted({speaker_key(r) for r in records['train']})
    if config and 'num_speakers' in config.get('acoustic',{}) and acoustic_config.num_speakers!=len(speakers):
        raise ValueError('Speaker count mismatch')
    acoustic_config.num_speakers=len(speakers)
    data=TTSDataset(records['train'],tokenizer,mel_config,speakers)
    validation=TTSDataset(records['val'],tokenizer,mel_config,speakers)
    fingerprints={k:fingerprint(v) for k,v in records.items()}
    acoustic=AcousticModel(acoustic_config).to(device);vocoder=Vocoder(vocoder_config).to(device)
    optimizer=torch.optim.AdamW(list(acoustic.parameters())+list(vocoder.parameters()),lr=learning_rate,weight_decay=0.0)
    config_data={'features':asdict(mel_config),'acoustic':asdict(acoustic_config),'vocoder':asdict(vocoder_config)}
    start=0
    if resume:
        checkpoint=torch.load(resume,map_location='cpu',weights_only=True)
        if checkpoint.get('schema_version')!=1 or checkpoint.get('task')!='tts':raise ValueError('Invalid TTS checkpoint')
        if checkpoint['dataset_fingerprints']!=fingerprints:raise ValueError('TTS dataset mismatch')
        if checkpoint.get('sampling_strategy')!='epoch-shuffle-v1' or checkpoint.get('seed')!=seed:
            raise ValueError('TTS resume sampling strategy or seed mismatch; start a fresh run for older checkpoints')
        restored_config = dict(checkpoint['config'])
        if 'vocoder' in restored_config:
            restored_config['vocoder'] = asdict(VocoderConfig(**restored_config['vocoder']))
        if restored_config!=config_data or checkpoint['speakers']!=speakers:raise ValueError('TTS config mismatch')
        if checkpoint['tokenizer']!=tokenizer.to_dict():raise ValueError('TTS tokenizer mismatch')
        acoustic.load_state_dict(checkpoint['acoustic']);vocoder.load_state_dict(checkpoint['vocoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start=checkpoint['step']
        if steps<=start:raise ValueError('Target steps must exceed checkpoint step')
        torch.set_rng_state(checkpoint['torch_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        restore_accelerator_rng(checkpoint.get('accelerator_rng', {}), device)
    run_id=validate_run_id(run_id or ('tts-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]))
    run=TTS_RUN_ROOT/run_id
    run.mkdir(parents=True,exist_ok=False)
    (run/'checkpoints').mkdir();(run/'logs').mkdir();(run/'evaluations').mkdir()
    atomic_json(run/'logs/run.json',{'schema_version':1,'task':'tts','run_id':run_id,'config':config_data,
                'seed':seed,'learning_rate':optimizer.param_groups[0]['lr'],'requested_learning_rate':learning_rate,'resume':str(resume) if resume else None,'device':str(device),'environment':environment(),
                'dataset_fingerprints':fingerprints,'source_manifest':records,'speakers':speakers,
                'quality_status':'experimental-engineering-only','start_step':start,'target_steps':steps,'sampling_strategy':'epoch-shuffle-v1','max_seconds':max_seconds})
    started=time.monotonic()
    stop_reason = None
    with TrainingControl(max_seconds) as stop_control:
      for step in range(start+1,steps+1):
          acoustic.train();vocoder.train();optimizer.zero_grad(set_to_none=True)
          item=data[sample_index(step,len(data),seed)]
          batch=collate_tts([item])
          a_loss,a_metrics=acoustic_loss(acoustic,batch)
          # Training crops are Mel-conditioned waveform examples, never inference playback.
          frames=min(32,len(item['mel'])-1)
          offset=random.randrange(max(1,len(item['mel'])-frames))
          target=item['waveform'][offset*160:(offset+frames)*160]
          condition=item['mel'][offset:offset+frames][None].to(device)
          target=torch.nn.functional.pad(target,(0,frames*160-len(target)))[None].to(device)
          predicted=vocoder(condition)
          v_loss,v_metrics=reconstruction_loss(predicted,target)
          loss=a_loss+v_loss.to(device)
          if not torch.isfinite(loss):raise FloatingPointError('Non-finite TTS loss')
          loss.backward();torch.nn.utils.clip_grad_norm_(list(acoustic.parameters())+list(vocoder.parameters()),1.0)
          optimizer.step()
          record={'step':step,'loss':float(loss.detach()),**a_metrics,**v_metrics}
          with (run/'logs/metrics.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
          stop_reason = stop_control.reason()
          if step==start+1 or step%save_interval==0 or step==steps or stop_reason:
              print(json.dumps(record),flush=True)
              checkpoint={'schema_version':1,'task':'tts','run_id':run_id,'step':step,
                  'epoch':(step-1)//len(data)+1,'acoustic':acoustic.state_dict(),'vocoder':vocoder.state_dict(),
                  'optimizer':optimizer.state_dict(),'config':config_data,'tokenizer':tokenizer.to_dict(),'speakers':speakers,
                  'dataset_fingerprints':fingerprints,'source_manifest':records,'torch_rng_state':torch.get_rng_state(),
                  'python_rng_state':random.getstate(),'seed':seed,'environment':environment(),
                  'sampling_strategy':'epoch-shuffle-v1','accelerator_rng':accelerator_rng(device)}
              path=run/'checkpoints'/f'step_{step:06d}.pt';temp=path.with_suffix('.tmp')
              try:torch.save(checkpoint,temp);temp.replace(path)
              finally:temp.unlink(missing_ok=True)
              if stop_reason:
                  break
    atomic_json(run/'logs/summary.json',{'checkpoint':str(path),'steps':step,'requested_steps':steps,'stop_requested':bool(stop_reason),'status':stop_reason or 'completed','target_steps':steps,'training_seconds':time.monotonic()-started,
                'parameters':{'acoustic':sum(p.numel() for p in acoustic.parameters()),'vocoder':sum(p.numel() for p in vocoder.parameters())},
                'training_recordings':len(data),'validation_recordings':len(validation),'last_metrics':record})
    print('Checkpoint:',path,flush=True)
    return path

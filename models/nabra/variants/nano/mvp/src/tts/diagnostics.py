"""Bounded single-pair learning experiments. Metrics never establish intelligibility."""
from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import time
import uuid

import torch
from torch.nn import functional as F

from ..features import read_wav
from ..ingestion import inspect_recording, sha256_file
from ..runtime import atomic_json, environment, select_device
from .dataset import MelConfig, MelExtractor, TTSDataset, collate_tts, provenance
from .tokenizer import TextTokenizer
from .acoustic_model import AcousticConfig, AcousticModel, acoustic_loss
from .vocoder import VocoderConfig, Vocoder, reconstruction_loss
from .inference import write_wav, load_checkpoint
from .trainer import TTS_RUN_ROOT


def pair_data(audio):
    audio = Path(audio)
    record = inspect_recording(audio, audio.with_suffix('.txt'))
    record['provenance'] = provenance(record)
    tokenizer = TextTokenizer()
    data = TTSDataset([record], tokenizer)
    return record, tokenizer, data[0]


def mel_from_wave(waveform, config=None):
    # Complex STFT's CPU fallback retains the gradient on Apple Silicon.
    return MelExtractor(config)(waveform.cpu() if waveform.device.type == 'mps' else waveform)


def noise_metrics(predicted, target):
    """Reference-aligned noise diagnostics; never used as an inference filter."""
    n=min(len(predicted),len(target))
    a,b=predicted.detach().cpu()[:n],target.detach().cpu()[:n]
    energy=F.avg_pool1d(b.square()[None,None],321,stride=1,padding=160)[0,0]
    quiet=energy<b.square().mean()*0.01
    metrics={'quiet_samples':int(quiet.sum())}
    if quiet.any():
        metrics.update(quiet_output_rms=float(a[quiet].square().mean().sqrt()),
                       quiet_target_rms=float(b[quiet].square().mean().sqrt()),
                       quiet_error_rms=float((a[quiet]-b[quiet]).square().mean().sqrt()))
    frequencies=torch.fft.rfftfreq(n,1/16000)
    for name,waveform in [('output',a),('target',b)]:
        spectrum=torch.fft.rfft(waveform)
        high=torch.fft.irfft(spectrum*(frequencies>=4000),n=n)
        metrics['above_4khz_'+name+'_rms']=float(high.square().mean().sqrt())
    return metrics


def wave_metrics(predicted, target):
    predicted, target = predicted.detach().cpu(), target.detach().cpu()
    n = min(len(predicted), len(target))
    _, metrics = reconstruction_loss(predicted[:n][None], target[:n][None])
    a, b = mel_from_wave(predicted[:n]), mel_from_wave(target[:n])
    error = (predicted[:n] - target[:n]).square().mean()
    metrics.update(mel_l1=float((a-b).abs().mean()), samples=len(predicted),
                   target_samples=len(target), duration_seconds=len(predicted)/16000,
                   rms=float(predicted.square().mean().sqrt()), peak=float(predicted.abs().max()),
                   clipping_fraction=float((predicted.abs() >= 1).float().mean()),
                   snr_db=float(10*torch.log10(target[:n].square().mean().clamp_min(1e-12)/error.clamp_min(1e-12))))
    metrics.update(noise_metrics(predicted,target))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('Nonfinite waveform metric')
    return metrics


def mel_metrics(predicted, target):
    predicted, target = predicted.detach().cpu(), target.detach().cpu()
    n = min(len(predicted), len(target))
    return {'frame_l1': float((predicted[:n]-target[:n]).abs().mean()),
            'frame_rmse': float((predicted[:n]-target[:n]).square().mean().sqrt()),
            'predicted_frames': len(predicted), 'target_frames': len(target),
            'length_ratio': len(predicted)/len(target), 'finite': bool(torch.isfinite(predicted).all())}


def attention_metrics(attention):
    a = attention.detach().cpu()[0]
    position = (a*torch.arange(a.shape[-1])).sum(-1)
    return {'start_position': float(position[0]), 'end_position': float(position[-1]),
            'text_tokens': a.shape[-1], 'backward_step_fraction': float((position[1:] < position[:-1]-0.1).float().mean()),
            'mean_max_probability': float(a.max(-1).values.mean())}


def save_plots(directory, target, predicted, attention):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    panels=[('target_mel',target.T,'Mel band'),('predicted_mel',predicted.T,'Mel band')]
    if attention is not None: panels.append(('attention',attention[0].T,'Text token index'))
    for name, value, ylabel in panels:
        fig, ax = plt.subplots(figsize=(10, 3))
        limits={} if name=='attention' else {'vmin':float(target.min()),'vmax':float(target.max())}
        im = ax.imshow(value.detach().cpu().numpy(), origin='lower', aspect='auto', interpolation='nearest',**limits)
        ax.set(xlabel='Decoder step' if name=='attention' else 'Mel frame (10 ms)', ylabel=ylabel, title=name.replace('_',' '))
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(directory/(name+'.png'), dpi=140); plt.close(fig)


def save_state(path, data):
    temp = path.with_suffix('.tmp')
    try:
        torch.save(data, temp); temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def restore(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'tts-quality-diagnostic' or checkpoint.get('schema_version') != 1 or checkpoint.get('stage') not in {'vocoder','acoustic'}:
        raise ValueError('Not a TTS diagnostic checkpoint')
    cls, config = (Vocoder, VocoderConfig) if checkpoint['stage']=='vocoder' else (AcousticModel, AcousticConfig)
    model = cls(config(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint


@torch.inference_mode()
def acoustic_outputs(model, item, max_frames=None):
    device = next(model.parameters()).device
    batch = collate_tts([item]); target = batch['mel'].to(device)
    forced, _, forced_attention = model(batch['tokens'].to(device), batch['text_lengths'], batch['speaker'].to(device), target)
    generated, info = model.generate(batch['tokens'].to(device), max_frames=max_frames or min(4000,len(item['mel'])*2), return_attention=True)
    attention = info.pop('attention')
    metrics = {'teacher_forced': mel_metrics(forced[0], item['mel']),
               'free_running': mel_metrics(generated[0], item['mel']), 'generation': info,
               'attention': attention_metrics(attention), 'teacher_forced_attention': attention_metrics(forced_attention)}
    return generated, attention, metrics


def train(stage, audio, steps=500, learning_rate=0.001, seed=42, device_name='cpu',
          resume=None, initialize=None, run_id=None, save_interval=250, model_options=None,
          free_running_weight=0.5, objective="reconstruction", vocoder_checkpoint=None):
    if objective not in {'reconstruction','phase'}: raise ValueError('Invalid diagnostic objective')
    if stage not in {'vocoder','acoustic'} or steps<1 or save_interval<1 or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Invalid diagnostic training arguments')
    if not 0<=free_running_weight<=1 or (resume and initialize):
        raise ValueError('Invalid initialization or free-running loss weight')
    torch.manual_seed(seed)
    device = select_device(device_name)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    record, tokenizer, item = pair_data(audio)
    options = model_options or {}
    config = VocoderConfig(**options) if stage=='vocoder' else AcousticConfig(vocab_size=tokenizer.vocab_size, **options)
    if objective=='phase' and (stage!='vocoder' or config.head!='spectral'): raise ValueError('Phase objective needs a spectral vocoder')
    if config.n_mels != 80: raise ValueError('Diagnostic features require 80 Mel bands')
    model = (Vocoder(config) if stage=='vocoder' else AcousticModel(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    start = 0
    if initialize:
        bundle = load_checkpoint(initialize, device)
        source = bundle[1 if stage=='vocoder' else 0]
        if asdict(source.config)!=asdict(config): raise ValueError('Initialization config mismatch')
        model.load_state_dict(source.state_dict())
    if resume:
        model, previous = restore(resume, device)
        if previous['stage']!=stage or asdict(model.config)!=asdict(config): raise ValueError('Resume stage/config mismatch')
        if any(previous['record'][k]!=record[k] for k in ('audio_sha256','transcript_sha256')): raise ValueError('Resume data mismatch')
        if previous['mel_config']!=asdict(MelConfig()) or previous['tokenizer']!=tokenizer.to_dict(): raise ValueError('Resume preprocessing mismatch')
        if previous['free_running_weight']!=free_running_weight: raise ValueError('Resume loss settings mismatch')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
        if previous.get('vocoder_sha256') and (not vocoder_checkpoint or sha256_file(vocoder_checkpoint)!=previous['vocoder_sha256']):
            raise ValueError('Resume frozen vocoder mismatch')
        optimizer.load_state_dict(previous['optimizer'])
        for group in optimizer.param_groups: group['lr'] = learning_rate
        start = previous['step']
        if steps<=start: raise ValueError('steps must exceed resume step')
        torch.set_rng_state(previous['rng'])
        if device.type=='cuda' and previous.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(previous['cuda_rng'])
        if device.type=='mps' and previous.get('mps_rng') is not None: torch.mps.set_rng_state(previous['mps_rng'])
    frozen_vocoder = None
    if vocoder_checkpoint:
        if stage!='acoustic':raise ValueError('Frozen vocoder refinement needs the acoustic stage')
        frozen_vocoder,vc=restore(vocoder_checkpoint,device)
        if vc['stage']!='vocoder' or vc['record']['audio_sha256']!=record['audio_sha256'] or vc['mel_config']!=asdict(MelConfig()):
            raise ValueError('Frozen vocoder source mismatch')
        frozen_vocoder.eval().requires_grad_(False)
    from ..paths import validate_run_id
    run = TTS_RUN_ROOT/validate_run_id(run_id or f'recovery-{stage}-{uuid.uuid4().hex[:10]}')
    run.mkdir(parents=True,exist_ok=False)
    for folder in ('checkpoints','samples','logs'): (run/folder).mkdir()
    metadata = {'stage':stage, 'record':record, 'model_config':asdict(config), 'mel_config':asdict(MelConfig()),
                'steps':steps, 'start_step':start, 'learning_rate':learning_rate, 'seed':seed, 'device':str(device),
                'resume':str(resume) if resume else None, 'initialize':str(initialize) if initialize else None,
                'environment':environment(), 'free_running_weight':free_running_weight, 'objective':objective,
                'vocoder_checkpoint':str(vocoder_checkpoint) if vocoder_checkpoint else None,
                'vocoder_sha256':sha256_file(vocoder_checkpoint) if vocoder_checkpoint else None,
                'intelligibility':'PENDING HUMAN REVIEW', 'training_scope':'single-pair overfit, not generalization'}
    atomic_json(run/'logs/run.json',metadata)
    target = item['waveform'].to(device); condition = item['mel'][None].to(device)
    batch = collate_tts([item]); began=time.monotonic()
    for step in range(start+1,steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if stage=='vocoder':
            # Full-context conditioning removes random-crop boundary mismatch.
            predicted = model(condition)[0][:len(target)]
            loss, metrics = reconstruction_loss(predicted[None], target[None])
            consistency = (mel_from_wave(predicted)-item['mel'].to('cpu' if device.type=='mps' else device)).abs().mean()
            loss = loss+2*consistency.to(device)
            metrics['mel_consistency'] = float(consistency.detach())
            if model.config.head == 'spectral':
                target_cpu = target.cpu() if device.type=='mps' else target
                reference = torch.stft(target_cpu,n_fft=400,hop_length=160,
                                       window=torch.hann_window(400,device=target_cpu.device),return_complex=True)
                spectrum = model.spectrum(condition)[0]
                phase_sensitive = (spectrum-reference).abs().mean()/reference.abs().mean().clamp_min(1e-5)
                loss = loss + phase_sensitive.to(device) + 9*F.l1_loss(predicted,target)
                metrics['complex_relative_l1'] = float(phase_sensitive.detach())
                if objective=='phase':
                    complex_mse=(spectrum-reference).abs().square().mean()/reference.abs().square().mean().clamp_min(1e-8)
                    waveform_mse=(predicted-target).square().mean()/target.square().mean().clamp_min(1e-8)
                    loss=complex_mse.to(device)+waveform_mse+0.1*consistency.to(device)
                    metrics.update(complex_relative_mse=float(complex_mse.detach()),waveform_relative_mse=float(waveform_mse.detach()))
        else:
            forced, metrics = acoustic_loss(model, batch)
            free, free_metrics, free_prediction = acoustic_loss(model, batch, teacher_forcing_ratio=0.0, return_prediction=True)
            loss = (1-free_running_weight)*forced+free_running_weight*free
            metrics.update({'free_'+k:v for k,v in free_metrics.items()})
            if frozen_vocoder is not None:
                generated=frozen_vocoder(free_prediction)[0][:len(target)]
                # Quiet-region emphasis is a TRAINING weight, never a source-audio
                # mask or recorded amplitude envelope applied during synthesis.
                energy=F.avg_pool1d(target.square()[None,None],321,stride=1,padding=160)[0,0]
                quiet=(energy<target.square().mean()*0.01).float()
                weighted_mse=((generated-target).square()*(1+4*quiet)).mean()/target.square().mean().clamp_min(1e-8)
                loss=loss+0.5*weighted_mse
                metrics.update(waveform_relative_mse=float(((generated-target).square().mean()/target.square().mean()).detach()),
                               weighted_waveform_relative_mse=float(weighted_mse.detach()))
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diagnostic loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        measured = step==start+1 or step%save_interval==0 or step==steps
        before = [p.detach().clone() for p in model.parameters()] if measured else None
        optimizer.step()
        row = {'step':step, 'loss':float(loss.detach()), 'gradient_norm':float(grad), **metrics}
        if measured:
            row['parameter_update_l2'] = math.sqrt(sum(float((p.detach()-b).square().sum()) for p,b in zip(model.parameters(),before)))
        with (run/'logs/metrics.jsonl').open('a') as file: file.write(json.dumps(row,allow_nan=False)+'\n')
        if measured:
            model.eval()
            with torch.inference_mode():
                if stage=='vocoder':
                    output = model(condition)[0].cpu()[:len(target)]
                    diagnostic = wave_metrics(output, item['waveform'])
                    write_wav(run/'samples'/f'step_{step:06d}.wav', output)
                    if step==steps: write_wav(run/'samples/vocoder_reconstruction.wav',output)
                else:
                    mel, attention, diagnostic = acoustic_outputs(model, item)
                    tensor_data={'target_mel':item['mel'], 'predicted_mel':mel.cpu(), 'attention':attention.cpu()}
                    torch.save(tensor_data,run/'samples'/f'step_{step:06d}.pt')
                    if step==steps:save_plots(run/'samples',item['mel'],mel[0],attention)
            atomic_json(run/'samples'/f'step_{step:06d}.json',diagnostic)
            checkpoint = {'schema_version':1,'task':'tts-quality-diagnostic','stage':stage,'step':step,
                          'model_config':asdict(config),'mel_config':asdict(MelConfig()),'tokenizer':tokenizer.to_dict(),
                          'record':record,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                          'rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                          'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None,
                          'free_running_weight':free_running_weight, 'objective':objective,
                          'vocoder_checkpoint':str(vocoder_checkpoint) if vocoder_checkpoint else None,
                          'vocoder_sha256':sha256_file(vocoder_checkpoint) if vocoder_checkpoint else None}
            path = run/'checkpoints'/f'step_{step:06d}.pt'; save_state(path,checkpoint)
            print(json.dumps({'step':step, 'training':row, 'evaluation':diagnostic}),flush=True)
    summary = {**metadata,'checkpoint':str(path),'training_seconds':time.monotonic()-began,
               'parameters':sum(p.numel() for p in model.parameters()), 'last_training':row,'evaluation':diagnostic}
    atomic_json(run/'logs/summary.json',summary)
    return path


@torch.inference_mode()
def full(acoustic_checkpoint, vocoder_checkpoint, output, text=None, device_name='cpu'):
    device=select_device(device_name)
    acoustic,a=restore(acoustic_checkpoint,device); vocoder,v=restore(vocoder_checkpoint,device)
    if a['stage']!='acoustic' or v['stage']!='vocoder' or a['mel_config']!=v['mel_config']:
        raise ValueError('Incompatible diagnostic components')
    if a['record']['audio_sha256']!=v['record']['audio_sha256']:raise ValueError('Overfit components use different recordings')
    acoustic.eval();vocoder.eval();tokenizer=TextTokenizer.from_dict(a['tokenizer'])
    text=text or a['record']['transcript']
    mel,info=acoustic.generate(torch.tensor([tokenizer.encode(text)],device=device),max_frames=1600,return_attention=True)
    attention=info.pop('attention')
    waveform=vocoder(mel)[0].cpu()
    repeat=vocoder(mel)[0].cpu()
    different_text='أنا أتعلم العربية' if text!='أنا أتعلم العربية' else 'مرحبا'
    alternative,_=acoustic.generate(torch.tensor([tokenizer.encode(different_text)],device=device),max_frames=1600)
    alt_wave=vocoder(alternative)[0].cpu()
    target=read_wav(a['record']['audio'])
    if sha256_file(a['record']['audio'])!=a['record']['audio_sha256']:raise ValueError('Original recording changed')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    write_wav(output/'tts_full_overfit.wav',waveform)
    write_wav(output/'different_text.wav',alt_wave)
    torch.save({'mel':mel.cpu(),'attention':attention.cpu()},output/'generated_mel.pt')
    metrics={**wave_metrics(waveform,target),**info,'text':text,'different_text':different_text,
             'distinct_text_outputs':alt_wave.shape!=waveform.shape or not torch.equal(alt_wave,waveform),
             'deterministic':torch.equal(waveform,repeat),'audio_sha256':sha256_file(output/'tts_full_overfit.wav'),
             'acoustic_checkpoint_sha256':sha256_file(acoustic_checkpoint),'vocoder_checkpoint_sha256':sha256_file(vocoder_checkpoint),
             'intelligibility':'PENDING HUMAN REVIEW','production_ready':False}
    atomic_json(output/'evaluation.json',metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2));return metrics


def main(stage):
    parser=argparse.ArgumentParser(description=f'Bounded NABRA {stage} quality diagnostic')
    parser.add_argument('--device',default='cpu',choices=['cpu','auto','mps','cuda'])
    parser.add_argument('--threads',type=int,default=2)
    if stage=='full':
        parser.add_argument('--acoustic-checkpoint',required=True);parser.add_argument('--vocoder-checkpoint',required=True)
        parser.add_argument('--output',required=True);parser.add_argument('--text')
    else:
        parser.add_argument('--audio',required=True);parser.add_argument('--steps',type=int,default=500)
        parser.add_argument('--learning-rate',type=float,default=0.001);parser.add_argument('--seed',type=int,default=42)
        parser.add_argument('--resume');parser.add_argument('--initialize');parser.add_argument('--run-id')
        parser.add_argument('--save-interval',type=int,default=250);parser.add_argument('--model-options',default='{}')
        parser.add_argument('--free-running-weight',type=float,default=0.5)
        parser.add_argument('--objective',choices=['reconstruction','phase'],default='reconstruction')
        parser.add_argument('--vocoder-checkpoint')
    args=parser.parse_args()
    if args.threads<1:parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if stage=='full':full(args.acoustic_checkpoint,args.vocoder_checkpoint,args.output,args.text,args.device)
    else:train(stage,args.audio,args.steps,args.learning_rate,args.seed,args.device,args.resume,args.initialize,args.run_id,
               args.save_interval,json.loads(args.model_options),args.free_running_weight,args.objective,args.vocoder_checkpoint)

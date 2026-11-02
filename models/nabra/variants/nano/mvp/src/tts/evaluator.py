import time
from pathlib import Path
import torch

from ..ingestion import sha256_file
from ..runtime import atomic_json, fingerprint
from .dataset import TTSDataset, collate_tts, inspect_sources
from .acoustic_model import acoustic_loss
from .vocoder import reconstruction_loss
from .inference import load_checkpoint,synthesize


@torch.inference_mode()
def evaluate(checkpoint_path,device='cpu',output=None):
    checkpoint_path=Path(checkpoint_path)
    data=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    bundle=load_checkpoint(checkpoint_path,device)
    acoustic,vocoder,tokenizer,features,speakers=bundle
    records=data['source_manifest']['val']
    # Revalidate each referenced audio hash at evaluation time.
    for record in records:
        if sha256_file(record['audio'])!=record['audio_sha256']:raise ValueError('Evaluation audio changed')
    if fingerprint(records)!=data['dataset_fingerprints']['val']:raise ValueError('Evaluation dataset fingerprint mismatch')
    ds=TTSDataset(records,tokenizer,features,speakers)
    rows=[]
    for item in ds:
        _,metrics=acoustic_loss(acoustic,collate_tts([item]))
        target=item['waveform'][None].to(device)
        output_wave=vocoder(item['mel'][None].to(device))[:,:target.shape[1]]
        _,vmetrics=reconstruction_loss(output_wave,target)
        rows.append({**metrics,**vmetrics})
    averages={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    texts=['مرحبا بكم في نبْرة للذكاء الاصطناعي','أنا أتعلم العربية']
    start=time.monotonic()
    waves=[];synthesis=[]
    for text in texts:
        waveform,info=synthesize(bundle,text,max_frames=400)
        repeat,_=synthesize(bundle,text,max_frames=400)
        if not torch.equal(waveform,repeat):raise ValueError('Nondeterministic CPU synthesis')
        waves.append(waveform);synthesis.append({'text':text,**info})
    distinct=waves[0].shape!=waves[1].shape or not torch.equal(waves[0],waves[1])
    if not distinct:raise ValueError('Synthesis ignores text')
    report={'schema_version':1,'task':'tts','checkpoint_sha256':sha256_file(checkpoint_path),
            'run_id':data['run_id'],'step':data['step'],'validation_fingerprint':data['dataset_fingerprints']['val'],
            'evaluated_recordings':len(ds),'metrics':averages,'synthesis':synthesis,'distinct_text_outputs':distinct,
            'synthesis_seconds':time.monotonic()-start,'device':str(device),
            'intelligibility':'not established','production_ready':False}
    destination=Path(output) if output else checkpoint_path.parent.parent/'evaluations'/f'{checkpoint_path.stem}.json'
    if destination.exists():raise FileExistsError(destination)
    atomic_json(destination,report)
    return report,destination

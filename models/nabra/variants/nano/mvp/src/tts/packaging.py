import json
import math
from pathlib import Path
import re
import shutil
import uuid
import torch
from safetensors.torch import save_file,load_file

from ..paths import BERHOT_ROOT
from ..ingestion import sha256_file
from ..runtime import atomic_json
from .inference import build,load_checkpoint

TTS_RELEASE_ROOT=BERHOT_ROOT/'storage/releases/nabra/tts/nano'
REQUIRED={'acoustic.safetensors','vocoder.safetensors','config.json','tokenizer.json','speakers.json','evaluation.json'}


def release_path(version):
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?',version):raise ValueError('Invalid release version')
    return TTS_RELEASE_ROOT/version


def package(checkpoint_path,version,report_path,max_mel_l1,max_spectral_loss,approve=False):
    if not approve:raise ValueError('Explicit approval required')
    if any(not math.isfinite(x) or x<0 for x in (max_mel_l1,max_spectral_loss)):raise ValueError('Invalid release thresholds')
    directory=release_path(version)
    if directory.exists():raise FileExistsError(directory)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    load_checkpoint(checkpoint_path,'cpu')
    report=json.loads(Path(report_path).read_text())
    if report.get('schema_version')!=1 or report.get('task')!='tts':raise ValueError('Invalid evaluation schema')
    if report['checkpoint_sha256']!=sha256_file(checkpoint_path):raise ValueError('Stale evaluation report')
    if report['run_id']!=checkpoint['run_id'] or report['step']!=checkpoint['step'] or report['validation_fingerprint']!=checkpoint['dataset_fingerprints']['val']:
        raise ValueError('Evaluation provenance mismatch')
    for key,limit in [('mel_l1',max_mel_l1),('spectral_loss',max_spectral_loss)]:
        value=report['metrics'][key]
        if not math.isfinite(value) or not 0<=value<=limit:raise ValueError(f'{key} exceeds release threshold')
    if not report['distinct_text_outputs'] or report['evaluated_recordings']<1:raise ValueError('Missing synthesis verification')
    directory.parent.mkdir(parents=True,exist_ok=True)
    temporary=directory.parent/('.'+version+'-'+uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for component in ('acoustic','vocoder'):
            weights={k:v.detach().cpu().contiguous() for k,v in checkpoint[component].items()}
            path=temporary/(component+'.safetensors');save_file(weights,str(path))
            restored=load_file(str(path))
            if any(not torch.equal(v,restored[k]) for k,v in weights.items()):raise ValueError('Exported weights differ')
        atomic_json(temporary/'config.json',checkpoint['config'])
        atomic_json(temporary/'tokenizer.json',checkpoint['tokenizer'])
        atomic_json(temporary/'speakers.json',checkpoint['speakers'])
        atomic_json(temporary/'evaluation.json',report)
        atomic_json(temporary/'manifest.json',{'schema_version':1,'task':'tts','version':version,
            'source_checkpoint_sha256':sha256_file(checkpoint_path),'source_run_id':checkpoint['run_id'],'step':checkpoint['step'],
            'dataset_fingerprints':checkpoint['dataset_fingerprints'],'approval':{'approved':True,'max_mel_l1':max_mel_l1,'max_spectral_loss':max_spectral_loss},
            'quality_status':'experimental-engineering-release','production_ready':False,
            'files':{name:sha256_file(temporary/name) for name in sorted(REQUIRED)}})
        if directory.exists():raise FileExistsError(directory)
        temporary.rename(directory)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return directory


def load_release(version,device='cpu'):
    directory=release_path(version)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema_version')!=1 or manifest.get('task')!='tts' or manifest['version']!=version:raise ValueError('Invalid release manifest')
    if not manifest['approval']['approved'] or not REQUIRED.issubset(manifest['files']):raise ValueError('Unapproved or incomplete release')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or name in {'.','..'}:raise ValueError('Invalid release path')
        if sha256_file(directory/name)!=digest:raise ValueError('Release hash mismatch')
    def read(name):return json.loads((directory/name).read_text())
    return build(read('config.json'),read('tokenizer.json'),read('speakers.json'),
                 load_file(str(directory/'acoustic.safetensors')),load_file(str(directory/'vocoder.safetensors')),device)

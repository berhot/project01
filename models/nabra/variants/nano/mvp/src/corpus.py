"""Prepare portable paired-speech datasets without changing the shared intake."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata

import torch

from .features import FeatureConfig, LogMelExtractor, read_wav
from .ingestion import inspect_recording, read_records, sha256_file, validate_splits
from .runtime import atomic_json, fingerprint

AUDIO_SUFFIXES = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}


def prompt_key(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def grouped_split(records, val_fraction=0.2, seed=42, split_by='prompt'):
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1')
    if split_by not in {'prompt', 'speaker', 'source'}:
        raise ValueError('Unknown split strategy')
    groups = {}
    for record in records:
        key = prompt_key(record['transcript']) if split_by == 'prompt' else record.get(split_by + '_id')
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f'{split_by}_id is required for every recording with this split strategy')
        groups.setdefault(key, []).append(record)
    if len(groups) < 2:
        raise ValueError('Need at least two distinct split groups; collect more data or choose an appropriate split strategy')
    order = sorted(groups, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    count = max(1, min(len(order)-1, round(len(order)*val_fraction)))
    held = set(order[:count])
    result = {'train': [], 'val': []}
    for key in order:
        result['val' if key in held else 'train'].extend(groups[key])
    validate_splits(result['train'], result['val'])
    return result


def prepare(source, output, convert=False, val_fraction=0.2, seed=42, split_by='prompt'):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_dir():
        raise ValueError('Source must be a directory of audio/transcript pairs')
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Prepared output and source directories must be separate')
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise ValueError('No supported audio files found')
    expected = [p.with_suffix('.txt') for p in files]
    if len(set(expected)) != len(expected):
        raise ValueError('Multiple audio formats share a transcript basename; retain one source per pair')
    if set(source.rglob('*.txt')) != set(expected):
        raise ValueError('Every recording must have exactly one matching .txt transcript; unmatched files found')
    if convert and not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg is required for --convert')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed conversion/validation never publishes a partly usable dataset.
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as temporary:
        stage = Path(temporary); (stage/'audio').mkdir()
        records = []; seen = set()
        for audio in files:
            transcript = audio.with_suffix('.txt')
            source_hash, text_hash = sha256_file(audio), sha256_file(transcript)
            identifier = hashlib.sha256(str(audio.relative_to(source)).encode()).hexdigest()[:20]
            target = stage/'audio'/(identifier+'.wav')
            label = target.with_suffix('.txt')
            label.write_bytes(transcript.read_bytes())
            if convert:
                # Decode at most 21 s. The strict <=20 s validator then rejects
                # overlong inputs rather than publishing truncated transcripts.
                result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n',
                                         '-i', str(audio), '-map', '0:a:0', '-t', '21',
                                         '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)],
                                        capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise ValueError(f'Audio conversion failed for {audio.name}: {result.stderr.strip()}')
            else:
                if audio.suffix.lower() != '.wav':
                    raise ValueError('Non-WAV input requires --convert')
                shutil.copyfile(audio, target)
            record = inspect_recording(target, label)
            if record['audio_sha256'] in seen:
                raise ValueError(f'Duplicate standardized audio: {audio.name}')
            seen.add(record['audio_sha256'])
            metadata_path = audio.with_suffix('.json')
            metadata_hash = sha256_file(metadata_path) if metadata_path.exists() else None
            metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
            if not isinstance(metadata, dict):
                raise ValueError('Recording metadata must be a JSON object')
            for key in ('speaker_id', 'source_id'):
                if metadata.get(key) is not None and (not isinstance(metadata[key], str) or not metadata[key].strip()):
                    raise ValueError(f'{key} must be a nonempty string or null')
                record[key] = metadata.get(key)
            from .tts.dataset import provenance
            known = provenance({'audio': str(audio), 'audio_sha256': source_hash})
            supplied = metadata.get('provenance', {})
            if not isinstance(supplied, dict):
                raise ValueError('provenance must be an object')
            record['provenance'] = {**known, **supplied}
            if known['source_type'] == 'synthetic-speech':
                record['provenance'].update({k: known[k] for k in ('source_type','generator','voice')})
            record['source'] = {'audio': str(audio.relative_to(source)), 'audio_sha256': source_hash,
                                'transcript_sha256': text_hash, 'metadata_sha256': metadata_hash}
            if sha256_file(audio) != source_hash or sha256_file(transcript) != text_hash:
                raise ValueError('Source changed during preparation')
            if metadata_hash and sha256_file(metadata_path) != metadata_hash:
                raise ValueError('Metadata changed during preparation')
            record['audio'] = str(target.relative_to(stage))
            record['transcript_file'] = str(label.relative_to(stage))
            records.append(record)
        splits = grouped_split(records, val_fraction, seed, split_by)
        for split, rows in splits.items():
            with (stage/(split+'.jsonl')).open('x', encoding='utf-8') as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
        summary = {'schema_version': 1, 'processing': 'nabra-paired-speech-v1', 'converted': convert,
                   'seed': seed, 'split_by': split_by, 'requested_validation_group_fraction': val_fraction,
                   'splits': {key: {'recordings':len(rows),'seconds':sum(r['duration_seconds'] for r in rows),
                                    'fingerprint':fingerprint(rows)} for key,rows in splits.items()},
                   'notes': ['Fraction applies to groups, not exact recording counts.',
                             'A prompt split is not speaker-disjoint; a speaker split need not be prompt-disjoint.']}
        if convert:
            summary['ffmpeg_version'] = subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
        atomic_json(stage/'dataset.json', summary)
        # Exclusive reservation prevents replacement of an existing publication.
        output.mkdir(exist_ok=False)
        try:
            for child in stage.iterdir(): child.rename(output/child.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return summary


def preflight(train_source, val_source, task='both', config=None):
    if task not in {'asr','tts','both'}:
        raise ValueError('Unknown task')
    config = config or {}
    if config.get('schema_version',1) != 1:
        raise ValueError('Unsupported config schema')
    if task=='both' and config:
        raise ValueError('Use --task asr or tts with a model config')
    splits = {'train':read_records(train_source),'val':read_records(val_source)}
    validate_splits(splits['train'],splits['val'])
    warnings = []
    for field, key in [('prompts',lambda r:prompt_key(r['transcript'])),
                       ('speakers',lambda r:r.get('speaker_id')),('sources',lambda r:r.get('source_id'))]:
        overlap = ({key(r) for r in splits['train']} & {key(r) for r in splits['val']}) - {None,''}
        if overlap: warnings.append(f'{len(overlap)} shared {field} across train/validation')
    report = {'schema_version':1, 'task':task, 'status':'PASS', 'warnings':warnings,
              'splits': {k:{'recordings':len(v),'seconds':sum(r['duration_seconds'] for r in v),
                            'fingerprint':fingerprint(v)} for k,v in splits.items()}}
    if task in {'asr','both'}:
        from .tokenizer import CharTokenizer
        from .model import NabraASR, ModelConfig
        tok = CharTokenizer.from_transcripts([r['transcript'] for r in splits['train']])
        extractor = LogMelExtractor(FeatureConfig(**config.get('features',{})))
        model = ModelConfig(**{'vocab_size':tok.vocab_size,**config.get('model',{})})
        if model.vocab_size!=tok.vocab_size or model.n_mels!=extractor.config.n_mels:
            raise ValueError('ASR model/features/vocabulary mismatch')
        for rows in splits.values():
            for record in rows:
                ids = tok.encode(record['transcript'])
                needed = len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
                frames = len(extractor(read_wav(record['audio'])))
                if int(NabraASR.output_lengths(torch.tensor(frames))) < needed:
                    raise ValueError(f'CTC alignment impossible: {record["audio"]}')
        report['asr']={'vocab_size':tok.vocab_size,'ctc_alignment':'feasible for all records'}
    if task in {'tts','both'}:
        from .tts.trainer import configs
        from .tts.dataset import speaker_key
        tok,features,acoustic,_ = configs(config)
        speakers = {speaker_key(r) for r in splits['train']}
        if 'num_speakers' in config.get('acoustic',{}) and acoustic.num_speakers!=len(speakers):
            raise ValueError('TTS speaker count mismatch')
        for rows in splits.values():
            for record in rows:
                tok.encode(record['transcript'])
                if speaker_key(record) not in speakers:
                    raise ValueError('TTS validation contains unseen speaker; use a prompt split within known speakers')
                wave = read_wav(record['audio'])
                if float(wave.square().mean().sqrt())<1e-5 or float((wave.abs()<1e-4).float().mean())>0.98:
                    raise ValueError('Silent or nearly silent TTS recording')
                if float((wave.abs()>=0.999).float().mean())>0.001:
                    warnings.append(f'Potential audio clipping: {record["audio"]}')
        report['tts']={'speakers':sorted(speakers),'tokenizer':'all transcripts encodable',
                       'note':'Checks format/compatibility, not intelligibility or licensing.'}
    return report


def main(command):
    parser=argparse.ArgumentParser(description='Prepare or preflight Nabra speech data')
    if command=='prepare':
        parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
        parser.add_argument('--convert',action='store_true')
        parser.add_argument('--val-fraction',type=float,default=0.2)
        parser.add_argument('--seed',type=int,default=42)
        parser.add_argument('--split-by',choices=['prompt','speaker','source'],default='prompt')
    else:
        parser.add_argument('--train-source',required=True);parser.add_argument('--val-source',required=True)
        parser.add_argument('--task',choices=['asr','tts','both'],default='both')
        parser.add_argument('--config');parser.add_argument('--output')
    args=parser.parse_args()
    if command=='prepare':
        result=prepare(args.source,args.output,args.convert,args.val_fraction,args.seed,args.split_by)
    else:
        result=preflight(args.train_source,args.val_source,args.task,
                         json.loads(Path(args.config).read_text()) if args.config else None)
        if args.output:
            with Path(args.output).open('x',encoding='utf-8') as file:
                file.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

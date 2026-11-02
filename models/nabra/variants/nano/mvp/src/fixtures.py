"""Explicit, optional download of three CC-BY-4.0 human-speech test fixtures.
Tests never call this downloader. No models or production data are downloaded.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from .paths import BERHOT_ROOT,MVP_ROOT
from .ingestion import sha256_file,inspect_split,write_manifest
from .runtime import atomic_json

FIXTURE_ROOT=BERHOT_ROOT/'storage/cache/datasets/fleurs-ar-smoke-v1'


def fixture_manifests(root=FIXTURE_ROOT):
    root=Path(root)
    config=json.loads((MVP_ROOT/'configs/human_speech_fixture.json').read_text())
    for record in config['records']:
        audio=root/record['audio']
        if sha256_file(audio)!=record['audio_sha256'] or sha256_file(audio.with_suffix('.txt'))!=record['transcript_sha256']:
            raise ValueError('Human fixture integrity mismatch')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(FIXTURE_ROOT));args=parser.parse_args()
    out=Path(args.output).resolve()
    config=json.loads((MVP_ROOT/'configs/human_speech_fixture.json').read_text())
    if out.exists():
        fixture_manifests(out);print('Existing fixtures verified:',out);return
    if not shutil.which('curl') or not shutil.which('ffmpeg'):raise RuntimeError('Fixture download needs curl and ffmpeg')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fleurs-',dir=out.parent) as temp:
        staging=Path(temp)/'fixture';staging.mkdir()
        index=Path(temp)/'rows.json'
        url='https://datasets-server.huggingface.co/rows?dataset=google%2Ffleurs&config=ar_eg&split=validation&offset=0&length=10'
        subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',url,'-o',str(index)],check=True)
        rows={r['row_idx']:r['row'] for r in json.loads(index.read_text())['rows']}
        for expected in config['records']:
            row=rows[expected['row_index']];audio_url=row['audio'][0]['src']
            if config['revision'] not in audio_url or row['id']!=expected['upstream_id']:
                raise ValueError('Upstream fixture revision changed; review instead of silently replacing')
            destination=staging/expected['audio'];destination.parent.mkdir(parents=True,exist_ok=True)
            original=staging/'originals'/destination.name;original.parent.mkdir(exist_ok=True)
            subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',audio_url,'-o',str(original)],check=True)
            if sha256_file(original)!=expected['original_audio_sha256']:raise ValueError('Original fixture hash mismatch')
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-i',str(original),
                            '-ar','16000','-ac','1','-c:a','pcm_s16le',str(destination)],check=True)
            destination.with_suffix('.txt').write_text(row['transcription']+'\n',encoding='utf-8')
        fixture_manifests(staging)
        atomic_json(staging/'provenance.json',config)
        (staging/'ATTRIBUTION.md').write_text(
            '# Google FLEURS Arabic fixtures\n\nAlexis Conneau et al. (2022), FLEURS.\n'
            'https://huggingface.co/datasets/google/fleurs\nhttps://arxiv.org/abs/2205.12446\n'
            'CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n\n'
            'Modified: PCM16 conversion and fixture-only repartition of upstream validation. '
            'Publisher transcripts unchanged. This is not a benchmark split. '
            'Speaker identity unknown; no endorsement implied.\n')
        staging.rename(out)
    print('Human speech fixtures downloaded and verified:',out)

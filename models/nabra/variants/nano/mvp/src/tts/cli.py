"""Argument handling for the thin TTS entry points."""
import argparse
import json
from pathlib import Path
import torch

from ..paths import TRAIN_ROOT,VAL_ROOT,BERHOT_ROOT
from ..runtime import select_device, atomic_json
from ..ingestion import write_manifest


def main(command):
    parser=argparse.ArgumentParser(description='NABRA experimental TTS '+command)
    if command in {'inspect','train'}:
        parser.add_argument('--train-source',default=str(TRAIN_ROOT));parser.add_argument('--val-source',default=str(VAL_ROOT))
    if command in {'train','evaluate','inference'}:
        parser.add_argument('--device',default='cpu',choices=['cpu','auto','mps','cuda'])
        parser.add_argument('--threads',type=int,default=2)
    if command=='inspect':
        parser.add_argument('--output',required=True)
    elif command=='train':
        parser.add_argument('--steps',type=int,default=100);parser.add_argument('--learning-rate',type=float,default=0.001)
        parser.add_argument('--seed',type=int,default=42);parser.add_argument('--resume');parser.add_argument('--run-id')
        parser.add_argument('--config');parser.add_argument('--save-interval',type=int,default=25)
        parser.add_argument('--max-seconds',type=float,help='Stop and checkpoint after the current step when the time budget expires')
    elif command=='evaluate':
        parser.add_argument('--checkpoint',required=True);parser.add_argument('--output')
    elif command=='inference':
        source=parser.add_mutually_exclusive_group(required=True)
        source.add_argument('--checkpoint');source.add_argument('--release')
        parser.add_argument('--text',required=True);parser.add_argument('--output',required=True)
        parser.add_argument('--max-frames',type=int,default=800);parser.add_argument('--speaker')
    elif command=='release':
        parser.add_argument('--checkpoint',required=True);parser.add_argument('--version',required=True)
        parser.add_argument('--report',required=True);parser.add_argument('--max-mel-l1',type=float,required=True)
        parser.add_argument('--max-spectral-loss',type=float,required=True);parser.add_argument('--approve',action='store_true')
    args=parser.parse_args()
    if hasattr(args,'threads'):
        if args.threads<1:parser.error('threads must be positive')
        torch.set_num_threads(args.threads)
    if command=='inspect':
        from .dataset import inspect_sources
        records=inspect_sources(args.train_source,args.val_source)
        out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
        for split,rows in records.items():write_manifest(rows,out/(split+'.jsonl'))
        print(json.dumps({'directory':str(out),'counts':{k:len(v) for k,v in records.items()}},indent=2))
    elif command=='train':
        from .trainer import train
        train(train_source=args.train_source,val_source=args.val_source,steps=args.steps,learning_rate=args.learning_rate,
              seed=args.seed,device_name=args.device,config=json.loads(Path(args.config).read_text()) if args.config else None,
              resume=args.resume,run_id=args.run_id,save_interval=args.save_interval,max_seconds=args.max_seconds)
    elif command=='evaluate':
        from .evaluator import evaluate
        report,path=evaluate(args.checkpoint,select_device(args.device),args.output)
        print(json.dumps(report,ensure_ascii=False,indent=2));print('Report:',path)
    elif command=='inference':
        from .inference import load_checkpoint,synthesize,write_wav
        from .packaging import load_release
        device=select_device(args.device)
        bundle=load_release(args.release,device) if args.release else load_checkpoint(args.checkpoint,device)
        waveform,info=synthesize(bundle,args.text,args.speaker,args.max_frames)
        write_wav(args.output,waveform,info['sample_rate'])
        atomic_json(str(args.output)+'.json',{'text':args.text,'checkpoint':args.checkpoint,'release':args.release,**info})
        print(json.dumps(info,ensure_ascii=False,indent=2))
    else:
        from .packaging import package
        print(package(args.checkpoint,args.version,args.report,args.max_mel_l1,args.max_spectral_loss,args.approve))

import argparse
import json
from pathlib import Path
import time
from collections import deque
import torch
from .inference import load_checkpoint,load_release
from .features import read_wav
from .runtime import select_device,atomic_json
from .streaming import StreamConfig,BufferedASR,WaveRecognizer,input_devices,capture


def main(microphone=False):
    parser=argparse.ArgumentParser(description='Buffered near-real-time ASR (bidirectional; not causal streaming)')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint');source.add_argument('--release')
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    parser.add_argument('--chunk-ms',type=int,default=100)
    parser.add_argument('--max-segment-seconds',type=float)
    parser.add_argument('--partial-seconds',type=float)
    parser.add_argument('--silence-seconds',type=float)
    parser.add_argument('--rms-threshold',type=float)
    parser.add_argument('--config',help='Versioned buffered ASR configuration')
    parser.add_argument('--output',help='Optional JSON event/latency report')
    if microphone:
        parser.add_argument('--list-devices',action='store_true');parser.add_argument('--microphone')
        parser.add_argument('--seconds',type=float)
    else:parser.add_argument('--audio',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if microphone and args.list_devices:
        print(json.dumps(input_devices(),ensure_ascii=False,indent=2));return
    if not args.checkpoint and not args.release:parser.error('Select --checkpoint or --release')
    if not 20<=args.chunk_ms<=1000:parser.error('chunk-ms must be between 20 and 1000')
    device=select_device(args.device)
    bundle=load_release(args.release,device) if args.release else load_checkpoint(args.checkpoint,device)
    options={}
    if args.config:
        supplied=json.loads(Path(args.config).read_text())
        if supplied.get('schema_version')!=1:parser.error('Unsupported streaming config schema')
        options.update(supplied['stream'])
    for name in ('max_segment_seconds','partial_seconds','silence_seconds','rms_threshold'):
        if getattr(args,name) is not None:options[name]=getattr(args,name)
    config=StreamConfig(**options)
    engine=BufferedASR(WaveRecognizer(bundle),config)
    # Keep a bounded preview; an optional JSONL log stores the full event history.
    events=deque(maxlen=128)
    event_count=0
    event_log=None
    if args.output:
        output=Path(args.output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        event_log=Path(str(output)+'.events.jsonl')
        with event_log.open('x'):pass
    def emit(event):
        nonlocal event_count
        event_count+=1
        events.append(event)
        line=json.dumps(event,ensure_ascii=False)
        if event_log:
            with event_log.open('a',encoding='utf-8') as file:file.write(line+'\n')
        print(line,flush=True)
    if microphone:stats=capture(engine,args.microphone,args.chunk_ms,args.seconds,emit)
    else:
        wave=read_wav(args.audio).numpy();size=16000*args.chunk_ms//1000;started=time.monotonic()
        for i,start in enumerate(range(0,len(wave),size)):
            for event in engine.feed(wave[start:start+size],i):emit(event)
        for event in engine.close():emit(event)
        stats={**engine.stats(),'wall_seconds':time.monotonic()-started,'source':args.audio}
    report={'config':vars(config),'events':list(events),'event_count':event_count,
            'events_truncated':event_count>len(events),'stats':stats}
    print(json.dumps(stats,indent=2))
    if args.output:
        if Path(args.output).exists():raise FileExistsError(args.output)
        atomic_json(args.output,report)

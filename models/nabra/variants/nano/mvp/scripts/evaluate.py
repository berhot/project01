import argparse
from src.evaluator import evaluate_checkpoint
from src.runtime import select_device


def main():
    parser=argparse.ArgumentParser(description='Evaluate and verify an ASR checkpoint')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--val-source',help='Optional paired directory or JSONL manifest')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    args=parser.parse_args()
    result,path=evaluate_checkpoint(args.checkpoint,select_device(args.device),args.val_source)
    print('CER:',result['cer']);print('WER:',result['wer']);print('Report:',path)


if __name__=='__main__':main()

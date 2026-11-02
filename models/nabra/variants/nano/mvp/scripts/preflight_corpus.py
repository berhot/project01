"""Validate speech manifest integrity and split leakage before training."""
import argparse
import json
from src.corpus_preparation import preflight

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--require-speaker-disjoint", action="store_true")
    a=p.parse_args()
    print(json.dumps(preflight(a.train,a.val,a.require_speaker_disjoint),ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()

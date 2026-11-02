"""Prepare an immutable Nabra speech corpus from an indexed source collection."""
import argparse
import json
from src.corpus_preparation import prepare

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a=p.parse_args()
    print(json.dumps(prepare(a.index,a.output,a.val_fraction,a.seed),ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()

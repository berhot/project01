# JSM Nano MVP

Ingestion → Tokenization → Dataset → Training → Checkpoint → Evaluation → Release → Inference.


## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r models/jsm/variants/nano/mvp/requirements.txt
cd models/jsm/variants/nano/mvp
```

If the root environment already exists, activate it and install the requirements.
On Windows, activate with `.venv\Scripts\activate` instead.

## Inspect the data and model

Run these commands from the MVP directory:

```bash
python -m scripts.ingestion
python -m scripts.tokenizer
python -m scripts.dataset
python -m scripts.model
```

## Train

```bash
python3 -m scripts.train \
  --steps 1000 \
  --batch-size 8 \
  --block-size 64 \
  --eval-interval 100 \
  --save-interval 500
```


Step 1 — Find your latest training run
Your training script automatically creates a run ID.

Run:

```bash
RUN_DIR=$(find /Users/farisalsaiari/Desktop/BERHOT/storage/artifacts/jsm/nano/mvp \
  -mindepth 1 -maxdepth 1 -type d -name 'nano-*' \
  -print | sort | tail -n 1)

RUN_ID=$(basename "$RUN_DIR")

CHECKPOINT="$RUN_DIR/checkpoints/step_001000.pt"

echo "Run ID: $RUN_ID"
echo "Checkpoint: $CHECKPOINT"

test -f "$CHECKPOINT" && echo "Checkpoint found"
```


Step 2 — Evaluate the checkpoint


```bash
python3 -m scripts.evaluate \
  --run-id "$RUN_ID" \
  --checkpoint "$CHECKPOINT" \
  --data /Users/farisalsaiari/Desktop/BERHOT/intake/wikipedia-ar
```


Step 3 — Test inference

```bash
python3 -m scripts.inference \
  --checkpoint "$CHECKPOINT" \
  --prompt "السعودية" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 20 \
  --seed 42
```


Step 4 — Package an experimental release

First, read the actual evaluation report:

```bash
cat "$RUN_DIR/evaluations/step_001000.json"
```


For this experimental MVP, if you've reviewed the result and want to approve packaging with a maximum loss threshold of 3.0, run:

```bash
python3 -m scripts.release \
  --run-id "$RUN_ID" \
  --checkpoint "$CHECKPOINT" \
  --version 0.1.0 \
  --max-val-loss 3.0 \
  --approve
```


Step 5 — Test the packaged release

```bash
python3 -m scripts.inference \
  --release 0.1.0 \
  --prompt "السعودية" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 20 \
  --seed 42
```





# ingestion
python3 -m scripts.ingestion

# Tokenizer
touch scripts/tokenizer.py

# Dataset
python3 -m scripts.dataset

# Transformer
python3 -m scripts.model

# Train
python3 -m scripts.train \
  --data data/saudi.txt \
  --output artifacts/jsm_mini_char.pt \
  --steps 1000 \
  --batch-size 8 \
  --block-size 64 \
  --eval-interval 100

# Inference
python3 -m scripts.inference \
  --checkpoint artifacts/jsm_mini_char.pt \
  --prompt "السعودية" \
  --max-new-tokens 200 \
  --temperature 0.8



  python3 -m scripts.train \
  --data data/saudi.txt \
  --resume artifacts/jsm_mini_char.pt \
  --output artifacts/jsm_mini_char_2000.pt \
  --steps 2000 \
  --batch-size 8 \
  --block-size 64 \
  --eval-interval 100
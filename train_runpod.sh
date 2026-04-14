#!/bin/bash
# Full training run — tune flags as needed.
set -e

python -u main.py \
    --puzzles data/raw/sudoku-3m.csv \
    --limit 3000000 \
    --epochs 40 \
    --balance \
    --checkpoint checkpoints/model_v2.pt \
    2>&1 | tee training_v2.log

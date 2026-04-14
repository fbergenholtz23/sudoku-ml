#!/bin/bash
# Full training run — tune flags as needed.
set -e

python main.py \
    --puzzles data/raw/sudoku-3m.csv \
    --limit 1000000 \
    --epochs 40 \
    --no-cache \
    --balance

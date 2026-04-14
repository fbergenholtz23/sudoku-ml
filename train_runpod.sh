#!/bin/bash
# Full training run — tune flags as needed.
set -e

python -u main.py \
    --puzzles data/raw/sudoku-3m.csv \
    --limit 1000000 \
    --epochs 40 \
    --balance \
    2>&1 | tee training.log

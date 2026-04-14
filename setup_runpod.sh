#!/bin/bash
# Run this once after cloning the repo on a RunPod instance.
# Usage: bash setup_runpod.sh <kaggle-dataset-slug>
# Example: bash setup_runpod.sh username/sudoku-3m-puzzles
set -e

DATASET_SLUG="${1:-}"

echo "=== Installing dependencies ==="
pip install -r requirements.txt
pip install -e .
pip install kaggle tqdm

echo "=== Setting up data directory ==="
mkdir -p data/raw data/processed checkpoints

if [ -n "$DATASET_SLUG" ]; then
    echo "=== Downloading dataset from Kaggle: $DATASET_SLUG ==="
    # Requires KAGGLE_USERNAME and KAGGLE_KEY env vars to be set,
    # or ~/.kaggle/kaggle.json to exist.
    kaggle datasets download -d "$DATASET_SLUG" -p data/raw --unzip
    echo "Downloaded files:"
    ls data/raw/
else
    echo ""
    echo "No Kaggle dataset slug provided. Upload sudoku-3m.csv manually:"
    echo "  runpodctl send data/raw/sudoku-3m.csv   (from your local machine)"
    echo "  or use the RunPod web file manager."
    echo ""
fi

echo "=== Setup complete. Run training with: ==="
echo "  bash train_runpod.sh"

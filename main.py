"""
Entry point: generate training data from puzzle files, train the model,
and optionally run inference on a test puzzle.

Usage:
    python main.py --puzzles data/raw/sudoku.csv --epochs 20
    python main.py --puzzle-file ../sudoku-solver/src/main/resources/sudoku/easy.txt --solve-only
"""

import argparse
import os
import numpy as np
import torch

from src.data.loader import load_puzzle_file, load_kaggle_csv
from src.data.generator import generate_dataset, select_balanced_puzzles
from src.data.cache import save_steps, load_arrays, cache_exists
from src.model.train import train
from src.model.inference import load_model, solve


def _validate(puzzle: np.ndarray, solution: np.ndarray) -> list[str]:
    """Return a list of error strings. Empty list means the solution is valid."""
    digits = set(range(1, 10))
    errors = []

    if (solution == 0).any():
        empty = int((solution == 0).sum())
        errors.append(f"{empty} cell(s) left empty")
        return errors  # remaining checks are noise if board is incomplete

    # Original clues must be preserved
    mask = puzzle != 0
    if not (solution[mask] == puzzle[mask]).all():
        errors.append("solution changed one or more original clues")

    # Rows
    for r in range(9):
        if set(solution[r]) != digits:
            errors.append(f"row {r} is invalid: {sorted(solution[r])}")

    # Columns
    for c in range(9):
        if set(solution[:, c]) != digits:
            errors.append(f"col {c} is invalid: {sorted(solution[:, c])}")

    # Boxes
    for br in range(3):
        for bc in range(3):
            box = solution[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()
            if set(box) != digits:
                errors.append(f"box ({br},{bc}) is invalid: {sorted(box)}")

    return errors


def _print_validation(puzzle: np.ndarray, solution: np.ndarray) -> None:
    errors = _validate(puzzle, solution)
    if not errors:
        print("Solution is valid!")
    else:
        print(f"Solution is INVALID ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")


def main():
    parser = argparse.ArgumentParser(description="Sudoku ML solver")
    parser.add_argument("--puzzles", help="Path to Kaggle CSV with puzzle/solution columns")
    parser.add_argument("--puzzle-file", help="Path to a single .txt puzzle file")
    parser.add_argument("--limit", type=int, default=50_000, help="Max puzzles to load from CSV")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--cache", default="data/processed/steps", help="Path (without .npz) to cache generated steps")
    parser.add_argument("--workers", type=int, default=None, help="CPU workers for generation (default: all cores)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache and regenerate")
    parser.add_argument("--balance", action="store_true", help="Classify puzzles by difficulty and sample equally from each strategy bucket")
    parser.add_argument("--solve-only", action="store_true", help="Load checkpoint and solve puzzle-file")
    args = parser.parse_args()

    if args.solve_only:
        assert args.puzzle_file, "--puzzle-file required for --solve-only"
        device = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")
        board = load_puzzle_file(args.puzzle_file)
        print("Puzzle:\n", board)
        model = load_model(args.checkpoint, device=device)
        model.to(device)
        solved = solve(model, board, device=device)
        print("Solved:\n", solved)
        _print_validation(board, solved)
        return

    # --- Load or generate training steps ---
    if not args.no_cache and cache_exists(args.cache):
        boards, rows, cols, digits, strategies = load_arrays(args.cache)
    else:
        puzzles: list[np.ndarray] = []
        difficulties = None
        if args.puzzles:
            puzzles, _, difficulties = load_kaggle_csv(args.puzzles, limit=args.limit)
            print(f"Loaded {len(puzzles)} puzzles from CSV" +
                  (" (with difficulty scores)" if difficulties is not None else ""))
        elif args.puzzle_file:
            puzzles = [load_puzzle_file(args.puzzle_file)]
            print(f"Loaded 1 puzzle from {args.puzzle_file}")
        else:
            print("No puzzle source provided. Use --puzzles or --puzzle-file.")
            return

        workers = args.workers or os.cpu_count() or 1
        if args.balance:
            puzzles = select_balanced_puzzles(puzzles, difficulties=difficulties, workers=workers)
        print(f"Generating training steps using {workers} workers...")
        steps = generate_dataset(puzzles, workers=workers)
        print(f"Generated {len(steps)} training steps")

        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        save_steps(steps, args.cache)
        boards, rows, cols, digits, strategies = load_arrays(args.cache)

    unique, counts = np.unique(strategies, return_counts=True)
    for name, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # --- Train ---
    train(boards, rows, cols, digits, strategies, epochs=args.epochs, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

"""
Generate training data by running the strategy solver step-by-step.
Each step produces a (board_state, row, col, digit, strategy) record.
"""

import os
import random
import multiprocessing as mp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src.strategies import STRATEGIES
from src.strategies.backtrack import backtrack


Step = tuple[np.ndarray, int, int, int, str]


def solve_with_labels(board: np.ndarray, use_backtrack_fallback: bool = True) -> tuple[list[Step], np.ndarray]:
    """
    Apply strategies greedily until no progress can be made.
    Returns (steps, final_board).
    Each step: (board_snapshot_before_move, row, col, digit, strategy_name).

    If use_backtrack_fallback=True and the board is not yet solved, a pure
    backtracking solver fills in the remaining cells (labeled "backtrack").
    These steps are still valid training signal — the board state leading up
    to them reflects real strategy application.
    """
    board = board.copy()
    steps: list[Step] = []

    changed = True
    while changed:
        changed = False
        for strategy in STRATEGIES:
            move = strategy(board)
            if move:
                r, c, digit, name = move
                steps.append((board.copy(), r, c, digit, name))
                board[r, c] = digit
                changed = True
                break  # restart from simplest strategy after each placement

    if use_backtrack_fallback and not is_solved(board):
        solved = backtrack(board)
        if solved is not None:
            empty_cells = [(r, c) for r in range(9) for c in range(9) if board[r, c] == 0]
            for r, c in empty_cells:
                steps.append((board.copy(), r, c, int(solved[r, c]), "backtrack"))
                board[r, c] = solved[r, c]

    return steps, board


def is_solved(board: np.ndarray) -> bool:
    return int(board[board == 0].size) == 0


def _classify_worker(board: np.ndarray) -> str:
    """Return the hardest strategy needed to solve this puzzle."""
    board = board.copy()
    hardest = "naked_single"
    changed = True
    while changed:
        changed = False
        for strategy in STRATEGIES:
            move = strategy(board)
            if move:
                r, c, digit, name = move
                if name not in ("naked_single", "backtrack"):
                    hardest = name
                board[r, c] = digit
                changed = True
                break
    return hardest


def select_balanced_puzzles(
    puzzles: list[np.ndarray],
    difficulties: "np.ndarray | None" = None,
    easy_ratio: int = 5,
    workers: int | None = None,
) -> list[np.ndarray]:
    """
    Classify every puzzle by its hardest required strategy, then return a
    selection that keeps ALL hard puzzles and samples easy_ratio * hard_count
    naked-single puzzles.

    easy_ratio: how many naked-single puzzles to keep per hard puzzle.
        Default 20 — keeps the easy examples while ensuring hard ones
        aren't drowned out.
    """
    if workers is None:
        workers = os.cpu_count() or 1

    if difficulties is not None:
        return _balance_by_difficulty_score(puzzles, difficulties, easy_ratio)

    # Fall back to running our own strategy classifier
    labels: list[str] = []
    chunksize = max(1, len(puzzles) // (workers * 4))
    with mp.Pool(processes=workers) as pool:
        with tqdm(total=len(puzzles), desc="Classifying", unit="puzzle") as pbar:
            for label in pool.imap(_classify_worker, puzzles, chunksize=chunksize):
                labels.append(label)
                pbar.update(1)

    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for puzzle, label in zip(puzzles, labels):
        groups[label].append(puzzle)

    print("Puzzle difficulty distribution:")
    for strategy, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {strategy}: {len(group)}")

    hard_groups = {k: v for k, v in groups.items() if k != "naked_single"}
    if not hard_groups:
        print("Warning: no hard puzzles found — try a larger --limit")
        return puzzles

    total_hard = sum(len(v) for v in hard_groups.values())
    if total_hard < 500:
        print(f"Warning: only {total_hard} hard puzzles found — model may not learn harder strategies well.")
        print("Consider a larger --limit or a dataset with more difficult puzzles.")

    selected: list[np.ndarray] = []
    for strategy, group in hard_groups.items():
        selected.extend(group)
    easy = groups.get("naked_single", [])
    n_easy = min(len(easy), total_hard * easy_ratio)
    selected.extend(random.sample(easy, n_easy))
    random.shuffle(selected)
    print(f"Selected {len(selected)} puzzles ({total_hard} hard + {n_easy} easy, ratio 1:{easy_ratio})")
    return selected


def _balance_by_difficulty_score(
    puzzles: list[np.ndarray],
    difficulties: "np.ndarray",
    easy_ratio: int,
) -> list[np.ndarray]:
    """
    Use the CSV's pre-computed difficulty scores to bucket and balance puzzles.
    Buckets: 0.0 (trivial), 0.1-1.9 (easy), 2.0-3.4 (medium), 3.5-5.0 (hard), 5.0+ (expert).
    Keeps all hard/expert puzzles and samples easy_ratio * hard_count from easier buckets.
    """
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for puzzle, d in zip(puzzles, difficulties):
        if d == 0.0:
            buckets["trivial"].append(puzzle)
        elif d < 2.0:
            buckets["easy"].append(puzzle)
        elif d < 3.5:
            buckets["medium"].append(puzzle)
        elif d < 5.0:
            buckets["hard"].append(puzzle)
        else:
            buckets["expert"].append(puzzle)

    print("Puzzle difficulty distribution:")
    for name in ("trivial", "easy", "medium", "hard", "expert"):
        if name in buckets:
            print(f"  {name}: {len(buckets[name])}")

    hard = buckets.get("hard", []) + buckets.get("expert", [])
    if not hard:
        print("Warning: no hard/expert puzzles found — try a larger --limit")
        return puzzles

    # Sample from easier buckets proportionally, capped so selection stays meaningful
    target_easy = len(hard) * easy_ratio
    easy_pool = buckets.get("trivial", []) + buckets.get("easy", []) + buckets.get("medium", [])
    n_easy = min(len(easy_pool), target_easy)
    if n_easy == len(easy_pool):
        print(f"Note: easy_ratio={easy_ratio} exceeds available easy puzzles — using all {n_easy}")

    selected = hard + random.sample(easy_pool, n_easy)
    random.shuffle(selected)
    print(f"Selected {len(selected)} puzzles ({len(hard)} hard/expert + {n_easy} easier, ratio 1:{easy_ratio})")
    return selected


def _solve_worker(board: np.ndarray) -> list[Step]:
    """Top-level wrapper so multiprocessing can pickle it."""
    steps, _ = solve_with_labels(board)
    return steps


def generate_dataset(puzzles: list[np.ndarray], workers: int | None = None) -> list[Step]:
    """
    Run the solver on a list of puzzles and collect all steps.
    Uses multiprocessing when workers > 1 (defaults to all CPU cores).
    """
    if workers is None:
        workers = os.cpu_count() or 1

    if workers <= 1 or len(puzzles) < 100:
        all_steps: list[Step] = []
        for board in tqdm(puzzles, desc="Generating", unit="puzzle"):
            steps, _ = solve_with_labels(board)
            all_steps.extend(steps)
        return all_steps

    # Cap chunksize — large chunks cause slow IPC serialization on macOS
    chunksize = min(500, max(1, len(puzzles) // (workers * 4)))
    all_steps = []
    with mp.Pool(processes=workers) as pool:
        with tqdm(total=len(puzzles), desc="Generating", unit="puzzle") as pbar:
            for steps in pool.imap_unordered(_solve_worker, puzzles, chunksize=chunksize):
                all_steps.extend(steps)
                pbar.update(1)
    return all_steps

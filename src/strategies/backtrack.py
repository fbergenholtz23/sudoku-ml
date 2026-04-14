"""
Backtracking solver — used as a fallback when the strategy-based solver
gets stuck (e.g. on hard/expert puzzles that require guessing).
"""

import numpy as np
from .candidates import get_candidates


def backtrack(board: np.ndarray) -> np.ndarray | None:
    """
    Solve the board via depth-first backtracking.
    Returns a solved copy, or None if unsolvable.
    Picks the empty cell with the fewest candidates first (MRV heuristic).
    """
    board = board.copy()
    return _solve(board)


def _solve(board: np.ndarray) -> np.ndarray | None:
    # Find the empty cell with the minimum remaining values
    best_cell = None
    best_cands: set[int] = set()
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                cands = get_candidates(board, r, c)
                if not cands:
                    return None  # dead end
                if best_cell is None or len(cands) < len(best_cands):
                    best_cell = (r, c)
                    best_cands = cands

    if best_cell is None:
        return board  # solved

    r, c = best_cell
    for digit in sorted(best_cands):
        board[r, c] = digit
        result = _solve(board)
        if result is not None:
            return result
        board[r, c] = 0

    return None

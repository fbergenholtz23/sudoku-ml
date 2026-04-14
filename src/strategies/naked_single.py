import numpy as np
from .candidates import get_candidates


def naked_single(board: np.ndarray) -> tuple[int, int, int, str] | None:
    """
    Strategy 1 — Naked Single.
    A cell has exactly one remaining candidate. Return (row, col, digit, strategy).
    """
    for r in range(9):
        for c in range(9):
            cands = get_candidates(board, r, c)
            if len(cands) == 1:
                return (r, c, next(iter(cands)), "naked_single")
    return None

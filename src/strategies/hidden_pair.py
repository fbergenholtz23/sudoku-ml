from itertools import combinations
import numpy as np
from .candidates import all_candidates
from .units import ALL_UNITS


def hidden_pair(board: np.ndarray) -> tuple[int, int, int, str] | None:
    """
    Strategy 4 — Hidden Pair (and Triple).
    Find N digits that appear only in N cells within a unit.
    Eliminate all other candidates from those N cells.
    Returns the first elimination as (row, col, digit, strategy).
    """
    cands = all_candidates(board)

    for n in (2, 3):
        for unit in ALL_UNITS:
            empty = [(r, c) for r, c in unit if board[r, c] == 0]
            unplaced = [d for d in range(1, 10) if not any(board[r, c] == d for r, c in unit)]
            for digit_subset in combinations(unplaced, n):
                cells = [
                    (r, c) for r, c in empty
                    if any(d in cands[(r, c)] for d in digit_subset)
                ]
                if len(cells) == n:
                    for r, c in cells:
                        for digit in cands[(r, c)] - set(digit_subset):
                            return (r, c, digit, f"hidden_{'pair' if n == 2 else 'triple'}_elimination")
    return None

from itertools import combinations
import numpy as np
from .candidates import all_candidates
from .units import ALL_UNITS


def naked_pair(board: np.ndarray) -> tuple[int, int, int, str] | None:
    """
    Strategy 3 — Naked Pair (and Triple).
    Find N cells in a unit whose combined candidates are exactly N digits.
    Eliminate those digits from all other cells in the unit.
    Returns the first elimination as (row, col, digit, strategy).
    """
    cands = all_candidates(board)

    for n in (2, 3):
        for unit in ALL_UNITS:
            empty = [(r, c) for r, c in unit if board[r, c] == 0]
            for subset in combinations(empty, n):
                combined = set().union(*(cands[cell] for cell in subset))
                if len(combined) == n:
                    others = [cell for cell in empty if cell not in subset]
                    for r, c in others:
                        for digit in combined & cands[(r, c)]:
                            return (r, c, digit, f"naked_{'pair' if n == 2 else 'triple'}_elimination")
    return None

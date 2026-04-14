import numpy as np
from .candidates import get_candidates
from .units import ALL_UNITS


def hidden_single(board: np.ndarray) -> tuple[int, int, int, str] | None:
    """
    Strategy 2 — Hidden Single.
    Within a unit, a digit has only one cell that can hold it.
    Return (row, col, digit, strategy).
    """
    for unit in ALL_UNITS:
        for digit in range(1, 10):
            cells = [
                (r, c) for r, c in unit
                if board[r, c] == 0 and digit in get_candidates(board, r, c)
            ]
            if len(cells) == 1:
                r, c = cells[0]
                return (r, c, digit, "hidden_single")
    return None

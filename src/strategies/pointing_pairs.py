import numpy as np
from .candidates import get_candidates
from .units import BOXES, ROWS, COLS


def pointing_pairs(board: np.ndarray) -> tuple[int, int, int, str] | None:
    """
    Strategy 5 — Pointing Pairs / Triples.
    If all candidates for a digit within a box lie in the same row or column,
    eliminate that digit from the rest of that row/column outside the box.
    Returns the first elimination as (row, col, digit, strategy).
    """
    for box in BOXES:
        box_cells = set(box)
        for digit in range(1, 10):
            cells = [
                (r, c) for r, c in box
                if board[r, c] == 0 and digit in get_candidates(board, r, c)
            ]
            if len(cells) < 2:
                continue

            rows = {r for r, c in cells}
            cols = {c for r, c in cells}

            if len(rows) == 1:
                r = next(iter(rows))
                for _, c in ROWS[r]:
                    if (r, c) not in box_cells and board[r, c] == 0:
                        if digit in get_candidates(board, r, c):
                            return (r, c, digit, "pointing_pairs_elimination")

            if len(cols) == 1:
                col = next(iter(cols))
                for r, _ in COLS[col]:
                    if (r, col) not in box_cells and board[r, col] == 0:
                        if digit in get_candidates(board, r, col):
                            return (r, col, digit, "pointing_pairs_elimination")
    return None

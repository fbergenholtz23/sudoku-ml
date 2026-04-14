import numpy as np


def get_candidates(board: np.ndarray, r: int, c: int) -> set[int]:
    """Return the set of valid digits for an empty cell (r, c)."""
    if board[r, c] != 0:
        return set()
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    used = (
        set(board[r])
        | set(board[:, c])
        | set(board[box_r:box_r+3, box_c:box_c+3].flatten())
    ) - {0}
    return set(range(1, 10)) - used


def all_candidates(board: np.ndarray) -> dict[tuple[int, int], set[int]]:
    """Return candidates for every empty cell on the board."""
    return {
        (r, c): get_candidates(board, r, c)
        for r in range(9)
        for c in range(9)
        if board[r, c] == 0
    }

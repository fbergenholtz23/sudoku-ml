"""Encode board states and moves as tensors for model input/output."""

import numpy as np


def board_to_input(board: np.ndarray) -> np.ndarray:
    """
    Full model input: placed digits + candidates, fully vectorized.
    Output shape: (18, 9, 9) — channels-first for PyTorch Conv2d.
      channels 0–8:  one-hot placed digits
      channels 9–17: candidate digits per empty cell
    """
    out = np.zeros((18, 9, 9), dtype=np.float32)
    empty = board == 0  # (9, 9)

    for d_idx in range(9):
        d = d_idx + 1
        placed = board == d  # (9, 9)

        # Placed channel
        out[d_idx] = placed.astype(np.float32)

        # Candidate channel: empty cell where d isn't in same row/col/box
        row_conflict = placed.any(axis=1, keepdims=True)   # (9, 1) broadcast over cols
        col_conflict = placed.any(axis=0, keepdims=True)   # (1, 9) broadcast over rows
        box_conflict = np.repeat(np.repeat(
            placed.reshape(3, 3, 3, 3).any(axis=(2, 3)),   # (3, 3)
            3, axis=0), 3, axis=1)                          # (9, 9)

        out[9 + d_idx] = (empty & ~row_conflict & ~col_conflict & ~box_conflict).astype(np.float32)

    return out


def move_to_target(r: int, c: int, digit: int) -> int:
    """Encode a move as a flat index into (81 * 9) output space."""
    cell = r * 9 + c
    return cell * 9 + (digit - 1)


def target_to_move(idx: int) -> tuple[int, int, int]:
    """Decode flat index back to (row, col, digit)."""
    digit = (idx % 9) + 1
    cell = idx // 9
    r, c = divmod(cell, 9)
    return r, c, digit

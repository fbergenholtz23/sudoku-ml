"""PyTorch Dataset wrapping generated strategy steps."""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.encoding import board_to_input


class SudokuStepDataset(Dataset):
    """
    Each sample is one strategy step:
      x: (9, 9) int8 tensor (raw board) — encoded to (18, 9, 9) on GPU/model
      y: int — flat index into (81 * 9) move space

    Accepts compact numpy arrays directly (boards, rows, cols, digits) to avoid
    Python list overhead on large datasets.
    """

    def __init__(
        self,
        boards: np.ndarray,   # (N, 9, 9) int8
        rows: np.ndarray,     # (N,) int8
        cols: np.ndarray,     # (N,) int8
        digits: np.ndarray,   # (N,) int8
    ):
        # Store compact int8 arrays (~400 MB for 4.7M steps)
        self.boards = np.ascontiguousarray(boards, dtype=np.int8)
        # Vectorized target calculation (much faster for millions of items)
        self.ys = torch.from_numpy(
            (rows.astype(np.int64) * 9 + cols.astype(np.int64)) * 9 + (digits.astype(np.int64) - 1)
        )

    @classmethod
    def from_steps(cls, steps: list) -> "SudokuStepDataset":
        """Build from a Python list of (board, r, c, digit, strategy) tuples."""
        n = len(steps)
        boards = np.empty((n, 9, 9), dtype=np.int8)
        rows = np.empty(n, dtype=np.int8)
        cols = np.empty(n, dtype=np.int8)
        digits = np.empty(n, dtype=np.int8)
        for i, (board, r, c, d, _) in enumerate(steps):
            boards[i] = board
            rows[i] = r
            cols[i] = c
            digits[i] = d
        return cls(boards, rows, cols, digits)

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return raw board as long for F.one_hot on GPU
        return torch.from_numpy(self.boards[idx]).long(), self.ys[idx]

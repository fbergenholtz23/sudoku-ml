"""PyTorch Dataset wrapping generated strategy steps."""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.encoding import board_to_input, move_to_target

# Precompute float32 tensors in RAM up to this many samples (~2 GB)
_PRECOMPUTE_LIMIT = 500_000


class SudokuStepDataset(Dataset):
    """
    Each sample is one strategy step:
      x: (18, 9, 9) float tensor — placed digits + candidates
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
        n = len(boards)
        if n <= _PRECOMPUTE_LIMIT:
            xs = np.empty((n, 18, 9, 9), dtype=np.float32)
            for i in tqdm(range(n), desc="Encoding", unit="step", leave=False):
                xs[i] = board_to_input(boards[i])
            ys = np.array(
                [move_to_target(int(rows[i]), int(cols[i]), int(digits[i])) for i in range(n)],
                dtype=np.int64,
            )
            self.xs = torch.from_numpy(xs)
            self.ys = torch.from_numpy(ys)
            self._precomputed = True
        else:
            # Store compact int8 arrays (~400 MB for 4.7M steps vs ~27 GB float32)
            self.boards = np.ascontiguousarray(boards, dtype=np.int8)
            self.ys = torch.from_numpy(np.array(
                [move_to_target(int(rows[i]), int(cols[i]), int(digits[i])) for i in range(n)],
                dtype=np.int64,
            ))
            self._precomputed = False

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
        return len(self.xs) if self._precomputed else len(self.boards)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._precomputed:
            return self.xs[idx], self.ys[idx]
        return torch.from_numpy(board_to_input(self.boards[idx])), self.ys[idx]

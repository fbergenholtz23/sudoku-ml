"""Save and load generated training steps to/from disk."""

from pathlib import Path
import numpy as np


def save_steps(steps: list, path: str | Path) -> None:
    """
    Save steps to a compressed .npz file.
    Steps are stored as parallel arrays for efficient I/O.
    """
    n = len(steps)
    boards = np.empty((n, 9, 9), dtype=np.int8)
    rows = np.empty(n, dtype=np.int8)
    cols = np.empty(n, dtype=np.int8)
    digits = np.empty(n, dtype=np.int8)
    strategies = np.empty(n, dtype=object)

    for i, (board, r, c, d, name) in enumerate(steps):
        boards[i] = board
        rows[i] = r
        cols[i] = c
        digits[i] = d
        strategies[i] = name

    np.savez_compressed(
        path,
        boards=boards,
        rows=rows,
        cols=cols,
        digits=digits,
        strategies=strategies.astype(str),
    )
    print(f"Saved {n} steps to {path}.npz")


def load_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load steps directly as numpy arrays — memory-efficient for large datasets.
    Returns (boards, rows, cols, digits, strategies) where:
      boards:     (N, 9, 9) int8
      rows:       (N,)      int8
      cols:       (N,)      int8
      digits:     (N,)      int8
      strategies: (N,)      str
    """
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(".npz")
    data = np.load(p, allow_pickle=False)
    boards = data["boards"]
    rows = data["rows"]
    cols = data["cols"]
    digits = data["digits"]
    strategies = data["strategies"]
    print(f"Loaded {len(rows)} steps from {p}")
    return boards, rows, cols, digits, strategies


def load_steps(path: str | Path) -> list:
    """Load steps as a Python list of tuples (use only for small datasets)."""
    boards, rows, cols, digits, strategies = load_arrays(path)
    return [
        (boards[i], int(rows[i]), int(cols[i]), int(digits[i]), str(strategies[i]))
        for i in range(len(rows))
    ]


def cache_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.with_suffix(".npz").exists() if not p.suffix else p.exists()

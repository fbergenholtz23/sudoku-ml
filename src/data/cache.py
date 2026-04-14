"""Save and load generated training steps to/from disk."""

import os
from pathlib import Path
import numpy as np


def save_steps(steps: list, path: str | Path) -> None:
    """
    Save steps. For large datasets, we save boards as a separate .npy file
    to allow memory-mapping during training, saving massive amounts of RAM.
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

    # Save metadata/labels in a compressed npz
    p = Path(path)
    np.savez_compressed(
        p.with_suffix(".npz"),
        rows=rows,
        cols=cols,
        digits=digits,
        strategies=strategies.astype(str),
    )
    
    # Save boards in a raw npy file for memory-mapping
    np.save(p.with_suffix(".boards.npy"), boards)
    
    print(f"Saved {n} steps to {p}.npz and {p}.boards.npy")


def load_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load steps. Boards are memory-mapped to save RAM.
    """
    p = Path(path)
    npz_path = p.with_suffix(".npz")
    npy_path = p.with_suffix(".boards.npy")
    
    # Load labels
    data = np.load(npz_path, allow_pickle=False)
    rows = data["rows"]
    cols = data["cols"]
    digits = data["digits"]
    strategies = data["strategies"]
    
    # Memory-map the boards (this doesn't load them into RAM yet)
    if npy_path.exists():
        boards = np.load(npy_path, mmap_mode="r")
    else:
        # Fallback for old cache format
        print("Warning: old cache format detected, loading into RAM...")
        boards = data["boards"]
        
    print(f"Loaded {len(rows)} steps from {p} (boards mmap-ed: {npy_path.exists()})")
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
    return p.with_suffix(".npz").exists()

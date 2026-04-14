"""Load sudoku puzzles from various formats."""

from pathlib import Path
import numpy as np


def load_puzzle_file(path: str | Path) -> np.ndarray:
    """
    Load a single puzzle from a space-delimited .txt file.
    First line is board size (e.g. 3 for a 9x9 board).
    Remaining lines are rows with digits separated by spaces (0 = empty).
    """
    lines = Path(path).read_text().strip().splitlines()
    size = int(lines[0])
    n = size * size
    board = np.array(
        [[int(x) for x in line.split()] for line in lines[1:n+1]],
        dtype=np.int8,
    )
    assert board.shape == (n, n), f"Expected ({n},{n}) board, got {board.shape}"
    return board


def load_kaggle_csv(
    path: str | Path,
    limit: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray | None]:
    """
    Load puzzles from a Kaggle sudoku CSV.
    Supports column names: ('puzzle'/'solution') or ('quizzes'/'solutions').
    Also reads a 'difficulty' column when present.
    Values are 81-char strings with digits (0 = empty).

    Returns (puzzles, solutions, difficulties) where difficulties is a float32
    numpy array if the column exists, otherwise None.
    """
    import pandas as pd

    headers = pd.read_csv(path, nrows=0).columns.tolist()
    if "puzzle" in headers and "solution" in headers:
        puzzle_col, solution_col = "puzzle", "solution"
    elif "quizzes" in headers and "solutions" in headers:
        puzzle_col, solution_col = "quizzes", "solutions"
    else:
        raise ValueError(f"Unrecognised CSV columns: {headers}")

    has_difficulty = "difficulty" in headers
    dtype_overrides = {puzzle_col: str, solution_col: str}
    df = pd.read_csv(path, nrows=limit, dtype=dtype_overrides)

    # Replace '.' with '0' for datasets that use dots for empty cells
    df[puzzle_col] = df[puzzle_col].str.replace(".", "0", regex=False)

    puzzles, solutions = [], []
    for _, row in df.iterrows():
        puzzles.append(np.array(list(row[puzzle_col]), dtype=np.int8).reshape(9, 9))
        solutions.append(np.array(list(row[solution_col]), dtype=np.int8).reshape(9, 9))

    difficulties = np.array(df["difficulty"].values, dtype=np.float32) if has_difficulty else None
    return puzzles, solutions, difficulties


def board_from_string(s: str) -> np.ndarray:
    """Parse an 81-character string into a (9, 9) board."""
    assert len(s) == 81
    return np.array(list(s), dtype=np.int8).reshape(9, 9)

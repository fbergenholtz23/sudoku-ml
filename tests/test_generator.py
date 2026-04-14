import numpy as np
from src.data.generator import solve_with_labels, is_solved
from src.data.loader import load_puzzle_file
from pathlib import Path


EASY_PATH = Path(__file__).parent.parent.parent / "sudoku-solver/src/main/resources/sudoku/easy.txt"


def test_solve_almost_complete():
    board = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 0],
    ], dtype=np.int8)
    steps, final = solve_with_labels(board)
    assert len(steps) == 1
    assert is_solved(final)
    assert final[8, 8] == 9


def test_solve_easy_puzzle():
    if not EASY_PATH.exists():
        return  # skip if not available
    board = load_puzzle_file(EASY_PATH)
    steps, final = solve_with_labels(board)
    assert len(steps) > 0
    assert is_solved(final)

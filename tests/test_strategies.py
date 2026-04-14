import numpy as np
import pytest
from src.strategies.candidates import get_candidates
from src.strategies.naked_single import naked_single
from src.strategies.hidden_single import hidden_single


# Almost-complete board: only cell (8,8) is empty, and 9 is the only candidate
ALMOST_DONE = np.array([
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


def test_get_candidates_single_missing():
    cands = get_candidates(ALMOST_DONE, 8, 8)
    assert cands == {9}


def test_naked_single_finds_move():
    move = naked_single(ALMOST_DONE)
    assert move is not None
    r, c, digit, name = move
    assert (r, c) == (8, 8)
    assert digit == 9
    assert name == "naked_single"


def test_naked_single_no_move_on_full_board():
    full = ALMOST_DONE.copy()
    full[8, 8] = 9
    assert naked_single(full) is None


def test_hidden_single_finds_move():
    # Build a board where naked single doesn't fire but hidden single does
    board = np.array([
        [0, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 1, 4, 3, 6, 5, 8, 9, 7],
        [3, 6, 5, 8, 9, 7, 2, 1, 4],
        [8, 9, 7, 2, 1, 4, 3, 6, 5],
        [5, 3, 1, 6, 4, 2, 9, 7, 8],
        [6, 4, 2, 9, 7, 8, 5, 3, 1],
        [9, 7, 8, 5, 3, 1, 6, 4, 2],
    ], dtype=np.int8)
    # Only one empty cell (0,0) — should be caught by naked_single first,
    # but hidden_single should also find it.
    move = hidden_single(board)
    assert move is not None
    r, c, digit, _ = move
    assert board[r, c] == 0
    assert digit in get_candidates(board, r, c)

import numpy as np
from src.data.encoding import board_to_input, move_to_target, target_to_move


BOARD = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
], dtype=np.int8)


def test_board_to_input_shape():
    t = board_to_input(BOARD)
    assert t.shape == (18, 9, 9)
    assert t.dtype == np.float32


def test_placed_channels_correct():
    t = board_to_input(BOARD)
    # Cell (0,0) has digit 5 → channel 4 (digit 5 - 1) should be 1
    assert t[4, 0, 0] == 1.0
    # Empty cell (0,2) → all placed channels should be 0
    assert t[:9, 0, 2].sum() == 0.0


def test_move_roundtrip():
    for r in range(9):
        for c in range(9):
            for d in range(1, 10):
                idx = move_to_target(r, c, d)
                assert target_to_move(idx) == (r, c, d)

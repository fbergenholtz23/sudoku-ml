"""Shared unit definitions (rows, columns, boxes)."""

ROWS = [[(r, c) for c in range(9)] for r in range(9)]
COLS = [[(r, c) for r in range(9)] for c in range(9)]
BOXES = [
    [(3 * (b // 3) + i, 3 * (b % 3) + j) for i in range(3) for j in range(3)]
    for b in range(9)
]
ALL_UNITS = ROWS + COLS + BOXES

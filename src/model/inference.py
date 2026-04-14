"""Run the trained model step-by-step to solve a puzzle."""

import numpy as np
import torch

from src.model.network import SudokuNet
from src.data.encoding import board_to_input, target_to_move
from src.strategies.candidates import get_candidates
from src.strategies.backtrack import backtrack


def load_model(checkpoint_path: str, device: str = "cpu", channels: int = 128, num_res_blocks: int = 12) -> SudokuNet:
    model = SudokuNet(channels=channels, num_res_blocks=num_res_blocks)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _get_mrv_moves(
    model: SudokuNet, board: np.ndarray, device: str
) -> list[tuple[int, int, int]] | None:
    """
    Find the most constrained empty cell (fewest valid candidates) and return
    its candidates ranked by model confidence for that cell.

    Using MRV (Minimum Remaining Values) keeps the branching factor small —
    typically 2-3 choices rather than scanning all 729 possible placements.
    Returns None if any empty cell has no candidates (contradiction).
    """
    # Move the raw board directly to GPU/model for encoding
    x = torch.from_numpy(board).long().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0)  # (81, 9)

    best_moves = None
    best_count = 10  # more than max possible candidates (9)

    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                continue
            valid = get_candidates(board, r, c)
            if not valid:
                return None  # contradiction — this cell has no candidates
            if len(valid) < best_count:
                best_count = len(valid)
                # Rank the valid digits by model confidence for this specific cell
                cell_logits = logits[r * 9 + c]  # shape (9,)
                ranked = sorted(valid, key=lambda d: cell_logits[d - 1].item(), reverse=True)
                best_moves = [(r, c, d) for d in ranked]
                if best_count == 1:
                    break  # forced move — can't do better
        if best_count == 1:
            break

    return best_moves


def solve(
    model: SudokuNet,
    board: np.ndarray,
    device: str = "cpu",
    max_backtracks: int = 150,
) -> np.ndarray:
    """
    Solve using MRV-guided backtracking.

    At each step the solver finds the most constrained empty cell (fewest
    valid candidates) and uses the model to rank those candidates by
    confidence. Trying the most-constrained cell first keeps the branching
    factor small (usually 2-3 choices) and detects dead ends early, so the
    solver rarely needs more than a handful of backtracks even on hard puzzles.

    Stack entries: (board_snapshot, remaining_moves_list)
    The remaining_moves list is modified in place as moves are consumed.
    """
    board = board.copy()

    if not (board == 0).any():
        return board  # already solved

    initial_moves = _get_mrv_moves(model, board, device)
    if initial_moves is None:
        print("Solver got stuck — initial board has a contradiction.")
        return board

    # Stack of (board_state, [remaining moves to try from this state])
    stack: list[tuple[np.ndarray, list]] = [(board, initial_moves)]
    backtracks = 0

    while stack:
        current_board, moves = stack[-1]

        # No moves left at this level — backtrack
        if not moves:
            stack.pop()
            backtracks += 1
            if backtracks > max_backtracks:
                print(f"Solver reached backtrack limit ({max_backtracks}).")
                break
            continue

        # Try the next model-ranked move for the MRV cell
        r, c, digit = moves.pop(0)
        new_board = current_board.copy()
        new_board[r, c] = digit

        # Solved?
        if not (new_board == 0).any():
            return new_board

        # Get the next MRV cell and model-ranked candidates
        next_moves = _get_mrv_moves(model, new_board, device)
        if next_moves is None:
            # Contradiction — don't push, try the next digit at this level
            continue

        stack.append((new_board, next_moves))

    # Model couldn't finish — fall back to guaranteed MRV backtracking
    print("Model solver reached its limit — falling back to backtracking.")
    result = backtrack(board)
    if result is None:
        print("Solver got stuck — board has no solution.")
        return board
    return result

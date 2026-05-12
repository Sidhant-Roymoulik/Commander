import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from BattleshipBoardV1 import BattleshipBoardV1


def compute_remaining_sizes(board: BattleshipBoardV1) -> list:
    remaining = []
    for _ship_id, ((r1, c1), (r2, c2)) in board.ships:
        size = max(abs(r2 - r1), abs(c2 - c1)) + 1
        cells = (
            [(r1, c) for c in range(c1, c2 + 1)]
            if r1 == r2
            else [(r, c1) for r in range(r1, r2 + 1)]
        )
        if not all(bool(board.hits[r, c]) for r, c in cells):
            remaining.append(size)
    return remaining


def build_forbidden_mask(board: BattleshipBoardV1) -> np.ndarray:
    forbidden = board.misses.copy()
    for _ship_id, ((r1, c1), (r2, c2)) in board.ships:
        cells = (
            [(r1, c) for c in range(c1, c2 + 1)]
            if r1 == r2
            else [(r, c1) for r in range(r1, r2 + 1)]
        )
        if all(bool(board.hits[r, c]) for r, c in cells):
            for r, c in cells:
                forbidden[r, c] = True
    return forbidden


def compute_valid_cells(
    board: BattleshipBoardV1, remaining_sizes: list, forbidden: np.ndarray
) -> np.ndarray:
    rows, cols = board.rows, board.cols
    valid = np.zeros((rows, cols), dtype=bool)
    unique_sizes = list(set(remaining_sizes))
    for r in range(rows):
        for c in range(cols):
            for size in unique_sizes:
                # Horizontal placements passing through (r, c)
                for c0 in range(max(0, c - size + 1), min(cols - size, c) + 1):
                    if all(not forbidden[r, c0 + i] for i in range(size)):
                        valid[r, c] = True
                        break
                if valid[r, c]:
                    break
                # Vertical placements passing through (r, c)
                for r0 in range(max(0, r - size + 1), min(rows - size, r) + 1):
                    if all(not forbidden[r0 + i, c] for i in range(size)):
                        valid[r, c] = True
                        break
                if valid[r, c]:
                    break
    return valid

"""Tests for AI targeting heuristic: remaining sizes, forbidden mask, valid cells."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from BattleshipBoardV1 import BattleshipBoardV1
from api.targeting import compute_remaining_sizes, build_forbidden_mask, compute_valid_cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_board(*ship_placements):
    """Create an empty board and place ships given as (r1,c1,r2,c2) tuples."""
    board = BattleshipBoardV1(ship_sizes=[])
    for r1, c1, r2, c2 in ship_placements:
        board.place_ship(r1, c1, r2, c2)
    return board


def sink(board, r1, c1, r2, c2):
    """Attack every cell of a ship to sink it."""
    if r1 == r2:
        for c in range(c1, c2 + 1):
            board.attack(r1, c)
    else:
        for r in range(r1, r2 + 1):
            board.attack(r, c1)


def valid_for(board):
    remaining = compute_remaining_sizes(board)
    forbidden = build_forbidden_mask(board)
    return compute_valid_cells(board, remaining, forbidden)


# ---------------------------------------------------------------------------
# compute_remaining_sizes
# ---------------------------------------------------------------------------

def test_remaining_sizes_all_intact():
    board = make_board((0, 0, 0, 4), (2, 0, 2, 1))  # size-5, size-2
    assert sorted(compute_remaining_sizes(board)) == [2, 5]
    print("PASS test_remaining_sizes_all_intact")


def test_remaining_sizes_one_sunk():
    board = make_board((0, 0, 0, 4), (2, 0, 2, 1))  # size-5, size-2
    sink(board, 2, 0, 2, 1)
    assert compute_remaining_sizes(board) == [5]
    print("PASS test_remaining_sizes_one_sunk")


def test_remaining_sizes_all_sunk():
    board = make_board((0, 0, 0, 1))  # size-2
    sink(board, 0, 0, 0, 1)
    assert compute_remaining_sizes(board) == []
    print("PASS test_remaining_sizes_all_sunk")


def test_remaining_sizes_partial_hit_not_sunk():
    board = make_board((0, 0, 0, 2))  # size-3
    board.attack(0, 0)  # only one hit — not sunk
    assert compute_remaining_sizes(board) == [3]
    print("PASS test_remaining_sizes_partial_hit_not_sunk")


# ---------------------------------------------------------------------------
# build_forbidden_mask
# ---------------------------------------------------------------------------

def test_forbidden_mask_misses():
    board = make_board((5, 5, 5, 5))  # size-1 placeholder (won't be attacked)
    board.attack(0, 0)  # miss (no ship there)
    forbidden = build_forbidden_mask(board)
    assert forbidden[0, 0] == True,  "miss cell must be forbidden"
    assert forbidden[0, 1] == False, "untouched cell must not be forbidden"
    print("PASS test_forbidden_mask_misses")


def test_forbidden_mask_sunk_ship_cells():
    board = make_board((3, 3, 3, 5))  # size-3 horizontal
    sink(board, 3, 3, 3, 5)
    forbidden = build_forbidden_mask(board)
    assert all(forbidden[3, c] for c in range(3, 6)), "sunk ship cells must be forbidden"
    print("PASS test_forbidden_mask_sunk_ship_cells")


def test_forbidden_mask_partial_hit_not_forbidden():
    board = make_board((3, 3, 3, 5))  # size-3 horizontal
    board.attack(3, 3)  # hit but ship not sunk
    forbidden = build_forbidden_mask(board)
    assert forbidden[3, 3] == False, "partial hit cell must NOT be forbidden"
    assert forbidden[3, 4] == False
    assert forbidden[3, 5] == False
    print("PASS test_forbidden_mask_partial_hit_not_forbidden")


# ---------------------------------------------------------------------------
# compute_valid_cells — core logic
# ---------------------------------------------------------------------------

def test_no_valid_cells_when_no_ships_remain():
    board = make_board((0, 0, 0, 1))  # size-2
    sink(board, 0, 0, 0, 1)
    valid = valid_for(board)
    assert not valid.any(), "no valid cells when all ships are sunk"
    print("PASS test_no_valid_cells_when_no_ships_remain")


def test_unattacked_open_cell_is_valid():
    board = make_board((0, 0, 0, 1))  # size-2
    valid = valid_for(board)
    assert valid[5, 5], "open cell with no nearby constraints should be valid"
    print("PASS test_unattacked_open_cell_is_valid")


def test_cell_squeezed_out_by_misses_is_invalid():
    """A single unattacked cell with misses on every side cannot host a size-2+ ship."""
    board = make_board((9, 8, 9, 9))  # size-2 remaining ship far away
    # Surround cell (5,5) with misses on all four sides
    for r, c in [(4, 5), (6, 5), (5, 4), (5, 6)]:
        board.attack(r, c)  # all misses (no ship there)
    # (5,5) is a 1-cell island — no size-2 ship can pass through it
    valid = valid_for(board)
    assert not valid[5, 5], "1-cell island with misses on all sides should be invalid for size-2+ ships"
    print("PASS test_cell_squeezed_out_by_misses_is_invalid")


# ---------------------------------------------------------------------------
# The key scenario: cell past the end of a sunk ship
# ---------------------------------------------------------------------------

def test_cell_past_sunk_ship_blocked_by_miss_is_invalid():
    """
    Ship: row 5, cols 3-5 (size-3), sunk.
    Misses at (5,2) and (5,6) flank it on the same row.
    Misses on the rows above and below seal off the ship lane.
    The first open cell in the same row beyond the miss is (5,7).

    With a remaining size-2 ship, (5,7) IS still valid because
    a size-2 ship can fit at (5,7)-(5,8).

    But (5,6) is a miss — it must be invalid.
    """
    # Sunk size-3; remaining size-2 ship placed far away
    board = make_board((5, 3, 5, 5), (0, 0, 0, 1))
    sink(board, 5, 3, 5, 5)
    board.attack(5, 2)  # miss — left flank
    board.attack(5, 6)  # miss — right flank

    valid = valid_for(board)
    assert not valid[5, 6], "(5,6) is a miss — must be invalid"
    assert not valid[5, 3], "sunk cell must be invalid"
    assert not valid[5, 4], "sunk cell must be invalid"
    assert not valid[5, 5], "sunk cell must be invalid"
    # (5,7) is open and a size-2 ship can fit at (5,7)-(5,8) — should be valid
    assert valid[5, 7], "(5,7) can host a size-2 ship and should be valid"
    print("PASS test_cell_past_sunk_ship_blocked_by_miss_is_invalid")


def test_isolated_cell_after_sunk_ship_at_edge():
    """
    Ship: row 5, cols 7-9 (size-3), sunk at the right edge of the board.
    Miss at (5,6) cuts off the row.
    The only unattacked cell in row 5 to the right of (5,6) is... none (board ends at col 9).
    So no cell in row 5, col > 6 should be valid.
    """
    board = make_board((5, 7, 5, 9), (0, 0, 0, 1))  # size-3 at edge + size-2 elsewhere
    sink(board, 5, 7, 5, 9)
    board.attack(5, 6)  # miss left of sunk ship

    valid = valid_for(board)
    # Sunk cells and miss
    for c in range(6, 10):
        assert not valid[5, c], f"(5,{c}) should be invalid (sunk or miss)"
    print("PASS test_isolated_cell_after_sunk_ship_at_edge")


def test_cell_adjacent_to_sunk_valid_if_another_ship_fits():
    """
    Sunk size-2 at (5,5)-(5,6). Remaining: size-3.
    Cell (5,7) is adjacent to the sunk ship and has open space: (5,7),(5,8),(5,9).
    A size-3 ship fits there, so (5,7) should be valid.
    """
    board = make_board((5, 5, 5, 6), (0, 0, 0, 2))  # size-2 + size-3
    sink(board, 5, 5, 5, 6)

    valid = valid_for(board)
    assert valid[5, 7], "(5,7) can host remaining size-3 ship and should be valid"
    print("PASS test_cell_adjacent_to_sunk_valid_if_another_ship_fits")


def test_cell_adjacent_to_sunk_invalid_if_no_ship_fits():
    """
    Sunk size-2 at (5,5)-(5,6). Remaining: size-3.
    Miss at (5,8) leaves only (5,7) open in that row segment.
    A size-3 ship cannot fit in a single-cell gap, so (5,7) should be invalid.
    """
    board = make_board((5, 5, 5, 6), (0, 0, 0, 2))  # size-2 + size-3
    sink(board, 5, 5, 5, 6)
    board.attack(5, 8)  # miss — only (5,7) left in that segment
    # Also block vertical: misses above and below (5,7)
    board.attack(4, 7)
    board.attack(6, 7)

    valid = valid_for(board)
    assert not valid[5, 7], "(5,7) is a 1-cell gap; size-3 ship can't fit — should be invalid"
    print("PASS test_cell_adjacent_to_sunk_invalid_if_no_ship_fits")


def test_late_game_one_ship_size_2_remaining():
    """
    Only a size-2 ship remains. Every cell where a size-2 ship cannot fit
    (isolated by misses or sunk cells) should be invalid.
    """
    # Place and sink a size-5 ship; keep a size-2 ship alive at (0,0)-(0,1)
    board = make_board((5, 0, 5, 4), (0, 0, 0, 1))
    sink(board, 5, 0, 5, 4)

    valid = valid_for(board)

    # Row 5 is all sunk/misses — no cell in row 5 should be valid
    for c in range(5):
        assert not valid[5, c], f"(5,{c}) is sunk — invalid"

    # Cell (0,0) is unattacked, size-2 ship could be at (0,0)-(0,1) — valid
    assert valid[0, 0], "(0,0) is a valid size-2 target"
    print("PASS test_late_game_one_ship_size_2_remaining")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_remaining_sizes_all_intact()
    test_remaining_sizes_one_sunk()
    test_remaining_sizes_all_sunk()
    test_remaining_sizes_partial_hit_not_sunk()

    test_forbidden_mask_misses()
    test_forbidden_mask_sunk_ship_cells()
    test_forbidden_mask_partial_hit_not_forbidden()

    test_no_valid_cells_when_no_ships_remain()
    test_unattacked_open_cell_is_valid()
    test_cell_squeezed_out_by_misses_is_invalid()

    test_cell_past_sunk_ship_blocked_by_miss_is_invalid()
    test_isolated_cell_after_sunk_ship_at_edge()
    test_cell_adjacent_to_sunk_valid_if_another_ship_fits()
    test_cell_adjacent_to_sunk_invalid_if_no_ship_fits()
    test_late_game_one_ship_size_2_remaining()

    print("\nAll heuristic tests passed!")

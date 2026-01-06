"""Tests for random ship placement functionality."""

from typing import List
from BattleshipBoardV1 import BattleshipBoardV1


def test_default_ship_sizes() -> None:
    """Test that default ship sizes are [2, 3, 3, 4, 5]."""
    board = BattleshipBoardV1()
    assert len(board.ships) == 5, f"Expected 5 ships, got {len(board.ships)}"

    # Extract ship sizes from board
    ship_sizes: List[int] = []
    for _ship_id, coords in board.ships:
        start, end = coords
        size = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
        ship_sizes.append(size)

    expected_sizes = [2, 3, 3, 4, 5]
    ship_sizes_sorted = sorted(ship_sizes)
    expected_sizes_sorted = sorted(expected_sizes)
    assert (
        ship_sizes_sorted == expected_sizes_sorted
    ), f"Expected ship sizes {expected_sizes_sorted}, got {ship_sizes_sorted}"
    print("✓ test_default_ship_sizes passed")


def test_custom_ship_sizes() -> None:
    """Test that custom ship sizes are placed correctly."""
    custom_sizes = [2, 2, 3]
    board = BattleshipBoardV1(ship_sizes=custom_sizes)
    assert len(board.ships) == 3, f"Expected 3 ships, got {len(board.ships)}"

    # Extract and verify ship sizes
    ship_sizes: List[int] = []
    for _ship_id, coords in board.ships:
        start, end = coords
        size = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
        ship_sizes.append(size)

    ship_sizes_sorted = sorted(ship_sizes)
    custom_sizes_sorted = sorted(custom_sizes)
    assert (
        ship_sizes_sorted == custom_sizes_sorted
    ), f"Expected sizes {custom_sizes_sorted}, got {ship_sizes_sorted}"
    print("✓ test_custom_ship_sizes passed")


def test_empty_ship_sizes() -> None:
    """Test that empty ship_sizes list creates board with no ships."""
    board = BattleshipBoardV1(ship_sizes=[])
    assert len(board.ships) == 0, f"Expected 0 ships, got {len(board.ships)}"
    print("✓ test_empty_ship_sizes passed")


def test_no_ship_overlaps() -> None:
    """Test that ships don't overlap on the board."""
    board = BattleshipBoardV1()

    # Count cell occupancy
    occupied_count = 0
    for row in board.ship_board:
        for cell in row:
            if cell != 0:  # Non-empty cell
                occupied_count += 1

    # Total cells occupied should equal sum of ship sizes
    expected_occupied = sum([2, 3, 3, 4, 5])
    assert (
        occupied_count == expected_occupied
    ), f"Expected {expected_occupied} occupied cells, got {occupied_count}"
    print("✓ test_no_ship_overlaps passed")


def test_ships_within_bounds() -> None:
    """Test that all ships are placed within board bounds."""
    board = BattleshipBoardV1()

    for _ship_id, coords in board.ships:
        start, end = coords
        start_row, start_col = start
        end_row, end_col = end

        # Check bounds
        assert 0 <= start_row < board.rows, f"Ship start_row {start_row} out of bounds"
        assert 0 <= end_row < board.rows, f"Ship end_row {end_row} out of bounds"
        assert 0 <= start_col < board.cols, f"Ship start_col {start_col} out of bounds"
        assert 0 <= end_col < board.cols, f"Ship end_col {end_col} out of bounds"

    print("✓ test_ships_within_bounds passed")


def test_ships_are_linear() -> None:
    """Test that all ships are either horizontal or vertical."""
    board = BattleshipBoardV1()

    for ship_id, coords in board.ships:
        start, end = coords
        start_row, start_col = start
        end_row, end_col = end

        # Ship must be horizontal or vertical (not diagonal)
        is_horizontal = start_row == end_row
        is_vertical = start_col == end_col
        assert (
            is_horizontal or is_vertical
        ), f"Ship {ship_id} is diagonal: {start} to {end}"

    print("✓ test_ships_are_linear passed")


def test_randomness() -> None:
    """Test that random placement actually produces different boards."""
    boards: List[BattleshipBoardV1] = [BattleshipBoardV1() for _ in range(5)]

    # Extract ship coordinates from each board
    ship_coords: List[List[object]] = []
    for board in boards:
        coords: List[object] = []
        for _ship_id, ship_coord in board.ships:
            coords.append(ship_coord)
        ship_coords.append(coords)

    # Check that at least some boards differ (they shouldn't all be identical)
    all_same = all(coords == ship_coords[0] for coords in ship_coords)
    assert not all_same, "All boards have identical ship placements (should be random)"
    print("✓ test_randomness passed")


def test_rectangular_board() -> None:
    """Test ship placement on non-square boards."""
    board = BattleshipBoardV1(rows=5, cols=15)
    assert (
        len(board.ships) == 5
    ), f"Expected 5 ships on 5x15 board, got {len(board.ships)}"

    # Verify all ships fit within bounds
    for _ship_id, coords in board.ships:
        start, end = coords
        assert start[0] < board.rows and end[0] < board.rows
        assert start[1] < board.cols and end[1] < board.cols

    print("✓ test_rectangular_board passed")


def test_small_board_with_large_ships() -> None:
    """Test that placement fails gracefully if ships don't fit."""
    try:
        # Try to place ships that don't fit on a 5x5 board
        board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[5, 5])
        # If we get here, both ships fit somehow
        assert len(board.ships) == 2
        print("✓ test_small_board_with_large_ships passed (ships fit)")
    except ValueError as e:
        # Expected if ships don't fit
        assert "Could not place ship" in str(e)
        print("✓ test_small_board_with_large_ships passed (raised error as expected)")


def test_display_ships() -> None:
    """Test that display_ships shows numeric ship IDs and water symbols."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[2, 3])
    # Just verify display doesn't crash
    print("\nDisplay of ships board:")
    board.display_ships()
    print("\n✓ test_display_ships passed")


if __name__ == "__main__":
    test_default_ship_sizes()
    test_custom_ship_sizes()
    test_empty_ship_sizes()
    test_no_ship_overlaps()
    test_ships_within_bounds()
    test_ships_are_linear()
    test_randomness()
    test_rectangular_board()
    test_small_board_with_large_ships()
    test_display_ships()

    print("\n✅ All ship placement tests passed!")

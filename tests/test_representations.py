"""Unit tests for board representation conversions."""

from typing import List
import numpy as np
from BattleshipBoardV1 import BattleshipBoardV1
from utils.representation_utils import (
    tensor_to_board,
    batch_boards_to_tensors,
    batch_tensors_to_boards,
    normalize_tensor,
    denormalize_tensor,
    get_board_stats,
)


def test_to_numpy_array_shape() -> None:
    """Test that to_numpy_array returns correct shape."""
    board = BattleshipBoardV1(rows=10, cols=10)
    array = board.to_numpy_array()
    assert array.shape == (10, 10, 3), f"Expected (10, 10, 3), got {array.shape}"
    assert array.dtype == np.float32, f"Expected float32, got {array.dtype}"
    print("✓ test_to_numpy_array_shape passed")


def test_to_numpy_array_empty_board() -> None:
    """Test that empty board is all zeros."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    array = board.to_numpy_array()
    assert np.allclose(array, 0.0), "Empty board should be all zeros"
    print("✓ test_to_numpy_array_empty_board passed")


def test_to_numpy_array_with_ship() -> None:
    """Test that ships are reflected in channel 0."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    board.place_ship(0, 0, 0, 2)  # Horizontal ship
    array = board.to_numpy_array()

    # Check channel 0 (ships)
    assert array[0, 0, 0] == 1.0, "Ship at (0,0) should be 1.0"
    assert array[0, 1, 0] == 1.0, "Ship at (0,1) should be 1.0"
    assert array[0, 2, 0] == 1.0, "Ship at (0,2) should be 1.0"
    assert array[0, 3, 0] == 0.0, "Non-ship at (0,3) should be 0.0"
    assert array[1, 0, 0] == 0.0, "Non-ship at (1,0) should be 0.0"
    print("✓ test_to_numpy_array_with_ship passed")


def test_to_numpy_array_with_attack() -> None:
    """Test that hits and misses are reflected in channels 1 and 2."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    board.place_ship(0, 0, 0, 2)
    board.attack(0, 0)  # Hit
    board.attack(1, 1)  # Miss

    array = board.to_numpy_array()
    assert array[0, 0, 1] == 1.0, "Hit at (0,0) should be 1.0 in channel 1"
    assert array[1, 1, 2] == 1.0, "Miss at (1,1) should be 1.0 in channel 2"
    assert array[0, 0, 2] == 0.0, "Hit location should not have miss"
    print("✓ test_to_numpy_array_with_attack passed")


def test_from_numpy_array_roundtrip() -> None:
    """Test that board -> array -> board preserves state."""
    board1 = BattleshipBoardV1(rows=8, cols=8, ship_sizes=[])
    board1.place_ship(2, 2, 2, 5)
    board1.place_ship(5, 0, 7, 0)
    board1.attack(2, 3)  # Hit on first ship
    board1.attack(0, 0)  # Miss

    # Convert to array and back
    array = board1.to_numpy_array()
    board2 = BattleshipBoardV1(rows=8, cols=8, ship_sizes=[])
    board2.from_numpy_array(array)

    # Convert back to array and compare
    array2 = board2.to_numpy_array()
    assert np.allclose(array, array2), "Round-trip conversion should be identical"
    print("✓ test_from_numpy_array_roundtrip passed")


def test_from_numpy_array_wrong_shape() -> None:
    """Test that from_numpy_array rejects wrong shapes."""
    board = BattleshipBoardV1(rows=10, cols=10, ship_sizes=[])
    bad_array = np.zeros((5, 5, 3), dtype=np.float32)

    try:
        board.from_numpy_array(bad_array)
        assert False, "Should raise ValueError for wrong shape"
    except ValueError as e:
        assert "Expected shape" in str(e)
        print("✓ test_from_numpy_array_wrong_shape passed")


def test_board_to_numpy_array_direct() -> None:
    """Test board.to_numpy_array() directly."""
    board = BattleshipBoardV1(rows=6, cols=6, ship_sizes=[])
    board.place_ship(0, 0, 0, 3)
    tensor = board.to_numpy_array()

    assert tensor.shape == (6, 6, 3)
    assert np.sum(tensor[:, :, 0]) == 4.0  # 4 ship cells
    print("✓ test_board_to_numpy_array_direct passed")


def test_tensor_to_board_wrapper() -> None:
    """Test wrapper function tensor_to_board."""
    board1 = BattleshipBoardV1(rows=7, cols=7, ship_sizes=[])
    board1.place_ship(0, 0, 3, 0)  # Vertical
    tensor = board1.to_numpy_array()
    board2 = tensor_to_board(tensor, rows=7, cols=7)

    tensor2 = board2.to_numpy_array()
    assert np.allclose(tensor, tensor2)
    print("✓ test_tensor_to_board_wrapper passed")


def test_batch_boards_to_tensors() -> None:
    """Test batch conversion of boards to tensors."""
    boards: List[BattleshipBoardV1] = []
    for i in range(3):
        board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
        board.place_ship(0, 0, 0, i + 1)  # Ships of different lengths
        boards.append(board)

    batch_array = batch_boards_to_tensors(boards)
    assert batch_array.shape == (
        3,
        5,
        5,
        3,
    ), f"Expected (3, 5, 5, 3), got {batch_array.shape}"
    print("✓ test_batch_boards_to_tensors passed")


def test_batch_tensors_to_boards() -> None:
    """Test batch conversion of tensors to boards."""
    boards: List[BattleshipBoardV1] = []
    for i in range(2):
        board = BattleshipBoardV1(rows=4, cols=4, ship_sizes=[])
        board.place_ship(i, 0, i, 2)
        boards.append(board)

    batch_array = batch_boards_to_tensors(boards)
    reconstructed_boards = batch_tensors_to_boards(batch_array)

    assert len(reconstructed_boards) == 2
    for i, board in enumerate(reconstructed_boards):
        assert board.rows == 4 and board.cols == 4
    print("✓ test_batch_tensors_to_boards passed")


def test_normalize_tensor() -> None:
    """Test tensor normalization."""
    array = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)
    normalized = normalize_tensor(array)

    expected = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    assert np.allclose(normalized, expected)
    print("✓ test_normalize_tensor passed")


def test_denormalize_tensor() -> None:
    """Test tensor denormalization to uint8."""
    array = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    denormalized = denormalize_tensor(array, scale=255.0)

    assert denormalized.dtype == np.uint8
    assert denormalized[0, 0, 0] == 0
    assert denormalized[0, 0, 2] == 255
    print("✓ test_denormalize_tensor passed")


def test_get_board_stats() -> None:
    """Test board statistics computation."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    board.place_ship(0, 0, 0, 2)  # 3-cell ship
    board.place_ship(3, 3, 3, 4)  # 2-cell ship
    board.attack(0, 0)  # Hit
    board.attack(0, 1)  # Hit
    board.attack(2, 2)  # Miss

    stats = get_board_stats(board)
    assert stats["num_ships"] == 5.0, f"Expected 5 occupied ship cells, got {stats['num_ships']}"
    assert stats["num_hits"] == 2, f"Expected 2 hits, got {stats['num_hits']}"
    assert stats["num_misses"] == 1, f"Expected 1 miss, got {stats['num_misses']}"
    assert (
        stats["cells_revealed"] == 3
    ), f"Expected 3 cells revealed, got {stats['cells_revealed']}"
    print("✓ test_get_board_stats passed")


def test_rectangular_board() -> None:
    """Test with non-square board dimensions."""
    board = BattleshipBoardV1(rows=5, cols=8, ship_sizes=[])
    board.place_ship(0, 0, 0, 4)
    board.attack(0, 2)

    array = board.to_numpy_array()
    assert array.shape == (5, 8, 3)

    board2 = BattleshipBoardV1(rows=5, cols=8, ship_sizes=[])
    board2.from_numpy_array(array)

    array2 = board2.to_numpy_array()
    assert np.allclose(array, array2)
    print("✓ test_rectangular_board passed")


def test_attack_out_of_bounds() -> None:
    """Test that attack() raises ValueError for out-of-bounds coordinates."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    for row, col in [(-1, 0), (0, -1), (5, 0), (0, 5), (10, 10)]:
        try:
            board.attack(row, col)
            assert False, f"Expected ValueError for ({row}, {col})"
        except ValueError as e:
            assert "bounds" in str(e).lower(), f"Error should mention bounds: {e}"
    print("✓ test_attack_out_of_bounds passed")


def test_attack_already_attacked() -> None:
    """Test that attacking the same cell twice raises ValueError."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    board.attack(2, 2)
    try:
        board.attack(2, 2)
        assert False, "Expected ValueError for re-attacking (2, 2)"
    except ValueError as e:
        assert "already" in str(e).lower(), f"Error should mention 'already': {e}"
    print("✓ test_attack_already_attacked passed")


def test_attack_miss_sets_misses() -> None:
    """Test that attacking a water cell marks it in misses."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    board.attack(3, 3)
    assert bool(board.misses[3, 3]), "misses[3,3] should be True after water attack"
    assert not bool(board.hits[3, 3]), "hits[3,3] should remain False"
    print("✓ test_attack_miss_sets_misses passed")


def test_attack_hit_sets_hits() -> None:
    """Test that attacking a ship cell marks it in hits."""
    board = BattleshipBoardV1(rows=5, cols=5, ship_sizes=[])
    board.place_ship(1, 1, 1, 3)
    board.attack(1, 2)
    assert bool(board.hits[1, 2]), "hits[1,2] should be True after ship attack"
    assert not bool(board.misses[1, 2]), "misses[1,2] should remain False"
    print("✓ test_attack_hit_sets_hits passed")


if __name__ == "__main__":
    test_to_numpy_array_shape()
    test_to_numpy_array_empty_board()
    test_to_numpy_array_with_ship()
    test_to_numpy_array_with_attack()
    test_from_numpy_array_roundtrip()
    test_from_numpy_array_wrong_shape()
    test_board_to_numpy_array_direct()
    test_tensor_to_board_wrapper()
    test_batch_boards_to_tensors()
    test_batch_tensors_to_boards()
    test_normalize_tensor()
    test_denormalize_tensor()
    test_get_board_stats()
    test_rectangular_board()
    test_attack_out_of_bounds()
    test_attack_already_attacked()
    test_attack_miss_sets_misses()
    test_attack_hit_sets_hits()

    print("\n✅ All tests passed!")

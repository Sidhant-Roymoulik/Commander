"""Utilities for batch board representation conversions for ML training."""

from typing import List, Dict, Any
import numpy as np
from numpy.typing import NDArray
from BattleshipBoardV1 import BattleshipBoardV1
from utils.constants import DEFAULT_ROWS, DEFAULT_COLS, ML_OCCUPIED, ML_EMPTY


def board_to_tensor(board: BattleshipBoardV1) -> NDArray[Any]:
    """
    Convert a single BattleshipBoard to its tensor representation.
    Wrapper around board.to_numpy_array() for consistency.

    Args:
        board: BattleshipBoardV1 instance
    Returns:
        np.ndarray of shape (rows, cols, 3) with dtype float32
    """
    return board.to_numpy_array()


def tensor_to_board(
    array: NDArray[Any], rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
) -> BattleshipBoardV1:
    """
    Reconstruct a board from its tensor representation.

    Args:
        array: np.ndarray of shape (rows, cols, 3)
        rows: number of rows (default 10)
        cols: number of columns (default 10)
    Returns:
        BattleshipBoardV1 instance
    """
    board = BattleshipBoardV1(rows=rows, cols=cols)
    board.from_numpy_array(array)
    return board


def batch_boards_to_tensors(boards: List[BattleshipBoardV1]) -> NDArray[Any]:
    """
    Convert a batch of boards to a stacked tensor (batch_size, rows, cols, 3).

    Args:
        boards: list of BattleshipBoardV1 instances
    Returns:
        np.ndarray of shape (batch_size, rows, cols, 3) with dtype float32
    """
    if not boards:
        raise ValueError("boards list is empty")

    batch_size = len(boards)
    rows, cols = boards[0].rows, boards[0].cols

    batch_array = np.zeros((batch_size, rows, cols, 3), dtype="float32")
    for i, board in enumerate(boards):
        batch_array[i] = board.to_numpy_array()

    return batch_array


def batch_tensors_to_boards(batch_array: NDArray[Any]) -> List[BattleshipBoardV1]:
    """
    Reconstruct a batch of boards from stacked tensors.

    Args:
        batch_array: np.ndarray of shape (batch_size, rows, cols, 3)
    Returns:
        list of BattleshipBoardV1 instances
    """
    if batch_array.ndim != 4 or batch_array.shape[3] != 3:
        raise ValueError(
            f"Expected shape (batch_size, rows, cols, 3), got {batch_array.shape}"
        )

    batch_size, rows, cols = (
        batch_array.shape[0],
        batch_array.shape[1],
        batch_array.shape[2],
    )
    boards: List[BattleshipBoardV1] = []

    for i in range(batch_size):
        board = BattleshipBoardV1(rows=rows, cols=cols, ship_sizes=[])
        board.from_numpy_array(batch_array[i])
        boards.append(board)

    return boards


def normalize_tensor(array: NDArray[Any]) -> NDArray[Any]:
    """
    Ensure tensor values are in [0.0, 1.0] range (idempotent).
    Clips values to normalized range.

    Args:
        array: np.ndarray
    Returns:
        np.ndarray clipped to [0.0, 1.0]
    """
    return np.clip(array, ML_EMPTY, ML_OCCUPIED).astype("float32")


def denormalize_tensor(array: NDArray[Any], scale: float = 255.0) -> NDArray[Any]:
    """
    Convert normalized [0.0, 1.0] tensor to integer range [0, scale].
    Useful for visualization or saving to uint8 images.

    Args:
        array: np.ndarray with values in [0.0, 1.0]
        scale: max value (default 255 for uint8)
    Returns:
        np.ndarray scaled to [0, scale] and cast to uint8
    """
    return (array * scale).astype(np.uint8)


def get_board_stats(board: BattleshipBoardV1) -> Dict[str, Any]:
    """
    Compute statistics about board state (useful for analysis/debugging).

    Args:
        board: BattleshipBoardV1 instance
    Returns:
        dict with keys: num_ships, num_hits, num_misses, hit_rate, cells_revealed
    """
    array = board.to_numpy_array()

    num_ships = int(np.sum(array[:, :, 0]))
    num_hits = int(np.sum(array[:, :, 1]))
    num_misses = int(np.sum(array[:, :, 2]))
    cells_revealed = num_hits + num_misses
    total_cells = board.rows * board.cols
    hit_rate = num_hits / max(cells_revealed, 1)

    return {
        "num_ships": num_ships,
        "num_hits": num_hits,
        "num_misses": num_misses,
        "cells_revealed": cells_revealed,
        "total_cells": total_cells,
        "hit_rate": hit_rate,
    }

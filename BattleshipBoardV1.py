import numpy as np
import random
from typing import List, Tuple, Optional, Any
from numpy.typing import NDArray
from utils.constants import (
    SHIP_EMPTY,
    WATER,
    HIT,
    MISS,
    DEFAULT_ROWS,
    DEFAULT_COLS,
    DEFAULT_SHIP_SIZES,
)
from utils.constants import ML_DTYPE


class BattleshipBoardV1:
    def __init__(
        self,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        ship_sizes: Optional[List[int]] = None,
    ) -> None:
        self.rows: int = rows
        self.cols: int = cols
        # Use NumPy arrays as canonical storage for performance and ML friendliness
        self.ship_board: NDArray[np.int16] = np.full(
            (rows, cols), SHIP_EMPTY, dtype=np.int16
        )
        self.hits: NDArray[np.bool_] = np.zeros((rows, cols), dtype=bool)
        self.misses: NDArray[np.bool_] = np.zeros((rows, cols), dtype=bool)
        self.ships: List[Tuple[int, Tuple[Tuple[int, int], Tuple[int, int]]]] = []

        # Default to standard Battleship ship sizes if not specified
        if ship_sizes is None:
            ship_sizes = DEFAULT_SHIP_SIZES

        # Sort ship sizes descending to place larger ships first (improves fit)
        ship_sizes = sorted(ship_sizes, reverse=True)

        # Randomly place ships on the board
        for ship_size in ship_sizes:
            self._place_ship_randomly(ship_size)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_ship_placement_valid(
        self, start_row: int, start_col: int, end_row: int, end_col: int
    ) -> bool:
        """
        Check if a ship can be placed at the given coordinates.
        Returns True if placement is valid (in bounds and no overlaps).
        """
        if not (
            self._in_bounds(start_row, start_col) and self._in_bounds(end_row, end_col)
        ):
            return False

        # Check for overlaps
        if start_row == end_row:  # Horizontal
            a, b = sorted((start_col, end_col))
            for col in range(a, b + 1):
                if int(self.ship_board[start_row, col]) != SHIP_EMPTY:
                    return False
        elif start_col == end_col:  # Vertical
            a, b = sorted((start_row, end_row))
            for row in range(a, b + 1):
                if int(self.ship_board[row, start_col]) != SHIP_EMPTY:
                    return False
        else:
            return False

        return True

    def _place_ship_randomly(self, ship_size: int) -> None:
        """
        Randomly place a ship of given size on the board.
        Tries up to 100 times to find a valid placement.
        Raises ValueError if unable to place ship after max attempts.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            # Random orientation: 0 = horizontal, 1 = vertical
            orientation = random.randint(0, 1)

            if orientation == 0:  # Horizontal
                row = random.randint(0, self.rows - 1)
                start_col = random.randint(0, self.cols - ship_size)
                end_col = start_col + ship_size - 1
                start_row = end_row = row
            else:  # Vertical
                col = random.randint(0, self.cols - 1)
                start_row = random.randint(0, self.rows - ship_size)
                end_row = start_row + ship_size - 1
                start_col = end_col = col

            if self._is_ship_placement_valid(start_row, start_col, end_row, end_col):
                self.place_ship(start_row, start_col, end_row, end_col)
                return

        raise ValueError(
            f"Could not place ship of size {ship_size} after {max_attempts} attempts"
        )

    def place_ship(
        self, start_row: int, start_col: int, end_row: int, end_col: int
    ) -> None:
        if not (
            self._in_bounds(start_row, start_col) and self._in_bounds(end_row, end_col)
        ):
            raise ValueError("Ship coordinates out of bounds")

        # assign a numeric ship id (1-based) and mark board with that id
        ship_id = len(self.ships) + 1
        if start_row == end_row:  # Horizontal
            a, b = sorted((start_col, end_col))
            for col in range(a, b + 1):
                self.ship_board[start_row, col] = ship_id
            self.ships.append((ship_id, ((start_row, a), (end_row, b))))
        elif start_col == end_col:  # Vertical
            a, b = sorted((start_row, end_row))
            for row in range(a, b + 1):
                self.ship_board[row, start_col] = ship_id
            self.ships.append((ship_id, ((a, start_col), (b, end_col))))
        else:
            raise ValueError("Ships must be placed either horizontally or vertically.")

    def attack(self, row: int, col: int) -> None:
        if not self._in_bounds(row, col):
            raise ValueError("Attack coordinates out of bounds")

        if bool(self.hits[row, col]) or bool(self.misses[row, col]):
            raise ValueError("Already attacked this position.")

        if int(self.ship_board[row, col]) == SHIP_EMPTY:
            self.misses[row, col] = True
        else:
            self.hits[row, col] = True

    def display_combined(self) -> None:
        for r in range(self.rows):
            row_cells: List[str] = []
            for c in range(self.cols):
                if bool(self.hits[r, c]):
                    row_cells.append(HIT)
                elif bool(self.misses[r, c]):
                    row_cells.append(MISS)
                else:
                    row_cells.append(WATER)
            print(" ".join(row_cells))

    def display_ships(self) -> None:
        for row in self.ship_board:
            print(
                " ".join(
                    WATER if int(cell) == SHIP_EMPTY else str(int(cell)) for cell in row
                )
            )

    def display_hits(self) -> None:
        for r in range(self.rows):
            print(
                " ".join(
                    HIT if bool(self.hits[r, c]) else WATER for c in range(self.cols)
                )
            )

    def display_misses(self) -> None:
        for r in range(self.rows):
            print(
                " ".join(
                    MISS if bool(self.misses[r, c]) else WATER for c in range(self.cols)
                )
            )

    def to_numpy_array(self) -> Any:
        """
        Convert board state to a normalized 3-channel numpy array (rows, cols, 3).
        Channels: [0]=ships_occupancy, [1]=hits, [2]=misses
        Values normalized to [0.0, 1.0] where 1.0 = occupied/hit/miss, 0.0 = empty.
        Returns: np.ndarray of shape (rows, cols, 3) with dtype float32
        """
        array = np.zeros((self.rows, self.cols, 3), dtype=ML_DTYPE)
        array[:, :, 0] = (self.ship_board != SHIP_EMPTY).astype(ML_DTYPE)
        array[:, :, 1] = self.hits.astype(ML_DTYPE)
        array[:, :, 2] = self.misses.astype(ML_DTYPE)
        return array

    def from_numpy_array(self, array: Any) -> None:
        """
        Load board state from a 3-channel numpy array.
        Overwrites current board state (ships, hits, misses).
        Note: Ship IDs are lost; all ships marked as ID 1 for simplicity.
        array: np.ndarray of shape (rows, cols, 3) with values in [0.0, 1.0]
        """
        if array.shape != (self.rows, self.cols, 3):
            raise ValueError(
                f"Expected shape {(self.rows, self.cols, 3)}, got {array.shape}"
            )

        # Reset arrays and ships list
        self.ship_board = np.where(array[:, :, 0] > 0.5, 1, SHIP_EMPTY).astype(np.int16)
        self.hits = (array[:, :, 1] > 0.5).astype(bool)
        self.misses = (array[:, :, 2] > 0.5).astype(bool)
        self.ships = []

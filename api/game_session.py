import uuid
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from BattleshipBoardV1 import BattleshipBoardV1
from utils.constants import DEFAULT_SHIP_SIZES


@dataclass
class GameSession:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    player_board: BattleshipBoardV1 = field(
        default_factory=lambda: BattleshipBoardV1(ship_sizes=[])
    )
    ai_board: BattleshipBoardV1 = field(
        default_factory=lambda: BattleshipBoardV1(ship_sizes=DEFAULT_SHIP_SIZES)
    )
    phase: str = "placement"  # "placement" | "playing" | "finished"
    winner: Optional[str] = None  # "player" | "ai" | None


def board_state(board: BattleshipBoardV1, reveal_ships: bool = False) -> dict:
    """Serialize a board to a JSON-friendly dict.

    reveal_ships=True exposes ship_board (used for player's own board and AI
    board after game over).
    """
    return {
        "rows": board.rows,
        "cols": board.cols,
        "hits": board.hits.tolist(),
        "misses": board.misses.tolist(),
        "ships": board.ship_board.tolist() if reveal_ships else None,
    }

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make repo root importable regardless of working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from BattleshipBoardV1 import BattleshipBoardV1
from models.cnn_model import load_model, predict_board
from api.game_session import GameSession, board_state
from api.targeting import compute_remaining_sizes, build_forbidden_mask, compute_valid_cells

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Commander Battleship API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------

MODEL_PATH = ROOT / "notebooks" / "models" / "cnn_3ch_best.pt"
_model = load_model(str(MODEL_PATH))
_model.eval()


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions: dict[str, GameSession] = {}


def _get_session(session_id: str) -> GameSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game session not found")
    return session


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ShipPlacement(BaseModel):
    start_row: int
    start_col: int
    end_row: int
    end_col: int


class PlaceShipsRequest(BaseModel):
    ships: List[ShipPlacement]


class AttackRequest(BaseModel):
    row: int
    col: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/game/new")
def new_game():
    session = GameSession()
    _sessions[session.id] = session
    return {
        "session_id": session.id,
        "phase": session.phase,
        "ship_sizes": [5, 4, 3, 3, 2],
    }


@app.post("/game/{session_id}/place-ships")
def place_ships(session_id: str, body: PlaceShipsRequest):
    session = _get_session(session_id)
    if session.phase != "placement":
        raise HTTPException(status_code=400, detail="Not in placement phase")

    # Fresh empty board
    session.player_board = BattleshipBoardV1(ship_sizes=[])
    try:
        for s in body.ships:
            session.player_board.place_ship(s.start_row, s.start_col, s.end_row, s.end_col)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    session.phase = "playing"
    return {
        "phase": session.phase,
        "player_board": board_state(session.player_board, reveal_ships=True),
    }


@app.post("/game/{session_id}/player-attack")
def player_attack(session_id: str, body: AttackRequest):
    session = _get_session(session_id)
    if session.phase != "playing":
        raise HTTPException(status_code=400, detail="Not in playing phase")

    try:
        session.ai_board.attack(body.row, body.col)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    hit = bool(session.ai_board.hits[body.row, body.col])
    just_sunk = session.ai_board.just_sunk_by(body.row, body.col)
    game_over = session.ai_board.all_ships_sunk()
    if game_over:
        session.phase = "finished"
        session.winner = "player"

    return {
        "row": body.row,
        "col": body.col,
        "hit": hit,
        "just_sunk": just_sunk,
        "game_over": game_over,
        "winner": session.winner,
        "ai_board": board_state(session.ai_board, reveal_ships=game_over),
    }


@app.get("/game/{session_id}/ai-attack")
def ai_attack(session_id: str):
    session = _get_session(session_id)
    if session.phase != "playing":
        raise HTTPException(status_code=400, detail="Not in playing phase")

    board = session.player_board

    sunk_ch = np.zeros((board.rows, board.cols), dtype=np.float32)
    for rc in board.get_sunk_ship_cells():
        sunk_ch[rc[0], rc[1]] = 1.0
    board_input = np.stack([board.hits.astype(np.float32), board.misses.astype(np.float32), sunk_ch], axis=-1)
    probs = predict_board(_model, board_input)  # (10, 10) [0, 1]

    # Mask attacked cells, then further restrict to cells where a remaining ship can fit
    attacked = board.hits | board.misses
    probs_masked = probs.copy()
    probs_masked[attacked] = -1.0

    remaining_sizes = compute_remaining_sizes(board)
    forbidden = build_forbidden_mask(board)
    valid_cells = compute_valid_cells(board, remaining_sizes, forbidden)
    probs_masked[~valid_cells] = -1.0

    # Fallback: if the heuristic eliminated all cells (shouldn't occur in a valid game),
    # revert to the attacked-only mask so the game can finish
    if probs_masked.max() < 0:
        probs_masked = probs.copy()
        probs_masked[attacked] = -1.0

    if probs_masked.max() < 0:
        raise HTTPException(status_code=400, detail="No cells left to attack")

    row, col = [int(x) for x in np.unravel_index(np.argmax(probs_masked), probs_masked.shape)]
    board.attack(row, col)

    hit = bool(board.hits[row, col])
    just_sunk = board.just_sunk_by(row, col)
    game_over = board.all_ships_sunk()
    if game_over:
        session.phase = "finished"
        session.winner = "ai"

    return {
        "row": row,
        "col": col,
        "hit": hit,
        "just_sunk": just_sunk,
        "game_over": game_over,
        "winner": session.winner,
        "prob_map": probs.tolist(),
        "player_board": board_state(board, reveal_ships=True),
    }


@app.get("/game/{session_id}/state")
def game_state(session_id: str):
    session = _get_session(session_id)
    game_over = session.phase == "finished"
    return {
        "session_id": session.id,
        "phase": session.phase,
        "winner": session.winner,
        "player_board": board_state(session.player_board, reveal_ships=True),
        "ai_board": board_state(session.ai_board, reveal_ships=game_over),
    }


# ---------------------------------------------------------------------------
# Static frontend (production — built by `npm run build`)
# ---------------------------------------------------------------------------

DIST = ROOT / "frontend" / "dist"

if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str = ""):
        candidate = DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(DIST / "index.html"))

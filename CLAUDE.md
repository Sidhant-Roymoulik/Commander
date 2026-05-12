# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Commander is a Battleship game where a human plays against a CNN-powered AI. The AI uses a trained neural network to predict ship locations from hits/misses/sunk-cell signals, then filters candidates with a geometric heuristic before picking the highest-probability unattacked cell each turn.

## Environment Setup

Python 3.11.9 with a local `.venv/`:

```bash
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Unix/Mac
pip install -r requirements.txt
```

Frontend (Node.js):

```bash
cd frontend && npm install
```

## Running the App

Both servers must run simultaneously. The Vite dev server proxies all `/game/*` requests to the FastAPI backend on port 8000.

```bash
# Terminal 1 — FastAPI backend (from repo root)
uvicorn api.app:app --reload

# Terminal 2 — Vite frontend
cd frontend && npm run dev
```

Then open `http://localhost:5173`.

## Running Tests

```bash
python tests/test_ship_placement.py
python tests/test_representations.py
python tests/test_cnn_model.py
python tests/test_ai_heuristic.py
python tests/test_model_behavior.py   # requires trained model weights
```

No test runner configured — tests are plain scripts called directly.

## Architecture

### Data Flow (ML)

```
BattleshipBoardV1 (random placement + simulated attacks)
    → to_numpy_array() → 3-channel float32 tensor [ships, hits, misses]
    → NumpyBoardDataset (PyTorch Dataset)
    → ShipProbCNN (input: 3 channels [hits, misses, sunk], output: 1 channel probabilities)
```

The ship channel is only used during training data generation; at inference time the input is `[hits, misses, sunk_cells]`.

### Request Flow (Web App)

```
Browser → Vite dev server (port 5173)
         → proxies /game/* → FastAPI (port 8000)
                              → GameSession (in-memory, keyed by UUID)
                              → BattleshipBoardV1 (board state)
                              → ShipProbCNN + targeting heuristic (AI)
```

### Key Files

- **`BattleshipBoardV1.py`** — Core game class. Three NumPy arrays: `ship_board` (integer ship IDs), `hits`, `misses` (booleans). `to_numpy_array()` / `from_numpy_array()` handle ML conversion. `just_sunk_by(row, col)` returns cells of the ship sunk by a specific hit. `get_sunk_ship_cells()` returns all cells of every completely sunk ship.
- **`api/app.py`** — FastAPI app. In-memory `_sessions` dict maps UUID → `GameSession`. Loads the CNN once at startup from `notebooks/models/cnn_3ch_best.pt`. Five endpoints: `POST /game/new`, `POST /game/{id}/place-ships`, `POST /game/{id}/player-attack`, `GET /game/{id}/ai-attack`, `GET /game/{id}/state`.
- **`api/game_session.py`** — `GameSession` dataclass (player board, AI board, phase, winner) and `board_state()` serializer. The AI board is initialized with ships pre-placed; the player board starts empty and is populated during placement phase.
- **`api/targeting.py`** — AI targeting heuristic. `compute_remaining_sizes()` lists unsunk ship sizes. `build_forbidden_mask()` marks misses and sunk-ship cells as off-limits. `compute_valid_cells()` returns a boolean mask of cells where at least one remaining ship could geometrically fit. Applied after CNN inference to eliminate impossible targets.
- **`models/cnn_model.py`** — `ShipProbCNN` (5-layer CNN), `train_model()`, `predict_board()` inference, `save_model()` / `load_model()`. `load_model()` auto-detects `in_channels` and `hidden` from checkpoint weights — no need to pass hyperparams at the call site.
- **`utils/constants.py`** — Board symbols, default 10×10 size, default ship sizes `[2,3,3,4,5]`, `float32` dtype.
- **`utils/representation_utils.py`** — Batch tensor conversion, normalization/denormalization, board statistics.
- **`notebooks/ship_prediction.ipynb`** — Trains on 100k synthetic boards; grid search over lr ∈ {8e-4, 1e-3, 1.2e-3} and hidden ∈ {48, 64, 96, 128}.
- **`notebooks/models/cnn_3ch_best.pt`** — Best trained weights (3-channel input: hits, misses, sunk).

### Frontend Components (`frontend/src/`)

- **`App.jsx`** — Top-level state machine: `loading → placement → playing → finished`. Orchestrates player attack → AI attack sequence with a 700ms delay.
- **`components/PlacementPhase.jsx`** — Drag-to-place ship placement UI.
- **`components/BattlePhase.jsx`** — Two-board battle view with AI probability heatmap overlay.
- **`components/EndScreen.jsx`** — Win/loss screen with board reveal and play-again button.
- **`components/Board.jsx`** — Reusable grid component used by both phases.
- **`api.js`** — Thin fetch wrappers for all five backend endpoints.

### CNN Architecture

```
Input: [batch, 3, H, W]  (hits channel, misses channel, sunk channel)
  → Conv(3→h) → Conv(h→2h) → Conv(2h→2h) → Conv(2h→h) → Conv(h→1)
Output: [batch, 1, H, W]  (logits; apply sigmoid for probabilities)
```

`h` is the `hidden_size` hyperparameter (best: 128). The sunk channel signals which hits belong to already-resolved ships so the model can suppress probability around them.

### AI Targeting Strategy

```
build 3-channel input [hits, misses, sunk_cells]
    → CNN inference → (H, W) probability map
    → mask attacked cells (hits | misses) → -1
    → compute_remaining_sizes + build_forbidden_mask + compute_valid_cells
    → mask cells where no remaining ship can geometrically fit → -1
    → fallback: if all masked, revert to attacked-only mask
    → argmax → attack that cell
```

## Design Notes

- Boards support non-square sizes (e.g., 5×15) — don't assume 10×10 in utilities.
- Tensor channels are consistently ordered: index 0 = ships, 1 = hits, 2 = misses. The model uses index 0 = hits, 1 = misses, 2 = sunk at inference time.
- `predict_board()` accepts HWC `(H, W, C)` or CHW `(C, H, W)` arrays — it transposes HWC to CHW automatically when the last dimension is ≤ 4.
- Game sessions are stored in-memory; restarting the backend invalidates all active sessions.

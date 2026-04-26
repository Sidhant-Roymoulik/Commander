# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Commander is a Battleship game simulator and ML-powered ship detection system. The core problem: given partial observations (hits/misses on a board), can a neural network predict where the remaining ships are located? The project generates synthetic game data, converts board states into tensors, and trains a CNN to output per-cell ship probability maps.

## Environment Setup

Python 3.11.9 with a local `.venv/`:

```bash
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Unix/Mac
pip install numpy torch matplotlib seaborn scikit-learn ipython jupyter
```

## Commands

```bash
# Run demo simulation
python main.py

# Run tests
python tests/test_ship_placement.py
python tests/test_representations.py

# Launch training notebook
jupyter notebook notebooks/ship_prediction.ipynb
```

There is no test runner configured (no pytest.ini / pyproject.toml) — tests are run directly as scripts.

## Architecture

### Data Flow

```
BattleshipBoardV1 (random placement + simulated attacks)
    → to_numpy_array() → 3-channel float32 tensor [ships, hits, misses]
    → NumpyBoardDataset (PyTorch Dataset)
    → ShipProbCNN (input: 2 channels [hits, misses], output: 1 channel probabilities)
```

The model never sees the ship channel at inference time — only hits and misses are inputs.

### Key Files

- **`BattleshipBoardV1.py`** — Core game class. Three NumPy arrays track state: `ship_board` (integer ship IDs), `hits`, `misses` (booleans). `to_numpy_array()` / `from_numpy_array()` handle ML conversion.
- **`utils/constants.py`** — Board symbols, default 10×10 size, default ship sizes `[2,3,3,4,5]`, `float32` dtype, normalized range `[0.0, 1.0]`.
- **`utils/representation_utils.py`** — Batch tensor conversion, normalization/denormalization, board statistics.
- **`models/cnn_model.py`** — `ShipProbCNN` (5-layer encoder-decoder CNN), `make_dataset()`, training loop, `predict_board()` inference function. Loss: `BCEWithLogitsLoss`.
- **`notebooks/ship_prediction.ipynb`** — Trains on 100k synthetic boards; runs grid search over lr ∈ {8e-4, 1e-3, 1.2e-3} and hidden units ∈ {48, 64, 96, 128}.
- **`notebooks/models/cnn_lr_0.001_hidden_128.pt`** — Best trained weights (lr=0.001, hidden=128).

### CNN Architecture

```
Input: [batch, 2, H, W]  (hits channel, misses channel)
  → Conv(2→h) → Conv(h→2h) → Conv(2h→2h) → Conv(2h→h) → Conv(h→1)
Output: [batch, 1, H, W]  (logits; apply sigmoid for probabilities)
```

`h` is the `hidden_size` hyperparameter (best: 128).

## Design Notes

- Boards support non-square sizes (e.g., 5×15) — don't assume 10×10 in utilities.
- Ship placement uses a retry loop (up to 100 attempts) to avoid collisions before raising an error.
- Tensor channels are consistently ordered: index 0 = ships, 1 = hits, 2 = misses. The model drops channel 0 for inference.
- All tensor values are normalized to `[0.0, 1.0]`.

"""
Tests for CNN model behavior on specific board states.

These tests check that the model has learned sensible probability outputs:
- Sunk ship cells should have low probability
- Cells at the ends of a sunk ship (no other ship can extend there) should
  have lower probability than open cells elsewhere on the board
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
from models.cnn_model import load_model, predict_board

MODEL_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "models" / "cnn_3ch_best.pt"


def load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH} — run the training notebook first")
    return load_model(str(MODEL_PATH))


def board_input(hits, misses, sunk):
    """Build (H, W, 3) input array from three (H, W) bool/float arrays."""
    return np.stack([
        np.array(hits, dtype=np.float32),
        np.array(misses, dtype=np.float32),
        np.array(sunk, dtype=np.float32),
    ], axis=-1)


def empty_channels():
    return (
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Sunk cells should have low probability
# ---------------------------------------------------------------------------

def test_sunk_cells_have_low_probability():
    """When a ship is fully sunk, its cells should have near-zero probability."""
    model = load()
    hits, misses, sunk = empty_channels()

    # Sink a horizontal size-3 ship at row 5, cols 3-5
    for c in range(3, 6):
        hits[5, c] = 1.0
        sunk[5, c] = 1.0

    probs = predict_board(model, board_input(hits, misses, sunk))

    for c in range(3, 6):
        assert probs[5, c] < 0.3, (
            f"Sunk cell (5,{c}) has unexpectedly high probability {probs[5,c]:.3f}; "
            "model should predict near-zero for already-resolved cells"
        )
    print("PASS test_sunk_cells_have_low_probability")


def test_sunk_cells_lower_than_open_cells():
    """Sunk ship cells should have strictly lower probability than unattacked open cells."""
    model = load()
    hits, misses, sunk = empty_channels()

    # Sink a vertical size-4 ship at col 7, rows 1-4
    for r in range(1, 5):
        hits[r, 7] = 1.0
        sunk[r, 7] = 1.0

    probs = predict_board(model, board_input(hits, misses, sunk))

    sunk_probs   = [probs[r, 7] for r in range(1, 5)]
    open_samples = [probs[8, 0], probs[8, 5], probs[9, 3]]  # far from the sunk ship

    assert max(sunk_probs) < min(open_samples), (
        f"Sunk cells (max {max(sunk_probs):.3f}) should be lower than "
        f"open cells (min {min(open_samples):.3f})"
    )
    print("PASS test_sunk_cells_lower_than_open_cells")


# ---------------------------------------------------------------------------
# Cells at the ends of a sunk ship (in the same axis) should be depressed
# ---------------------------------------------------------------------------

def test_end_cells_lower_than_far_open_cells():
    """
    After sinking a ship, the unattacked cells immediately beyond each end of
    the ship (same row/col) should have lower probability than open cells far
    away on the board.

    Setup: horizontal size-3 sunk at row 5, cols 4-6.
    Misses at (5,3) and (5,7) box in the ship so no other ship can extend there —
    those end cells are already misses.
    Check that the adjacent cells further out (5,2) and (5,8) are lower than
    a completely untouched area like (0,0).

    Without the sunk-target fix the model tends to assign high probability to
    the region around a cluster of hits regardless of the sunk channel.
    """
    model = load()
    hits, misses, sunk = empty_channels()

    # Sink ship at row 5, cols 4-6
    for c in range(4, 7):
        hits[5, c] = 1.0
        sunk[5, c] = 1.0

    # Misses immediately flanking the ship
    misses[5, 3] = 1.0
    misses[5, 7] = 1.0

    probs = predict_board(model, board_input(hits, misses, sunk))

    # Cells one step further out in the same lane — still unattacked but near a sunk ship
    near_end_probs = [probs[5, 2], probs[5, 8]]
    far_open_prob  = probs[0, 0]  # completely untouched corner

    for i, p in enumerate(near_end_probs):
        assert p < far_open_prob, (
            f"Cell near sunk ship end (prob {p:.3f}) should be lower than "
            f"a far open cell (prob {far_open_prob:.3f})"
        )
    print("PASS test_end_cells_lower_than_far_open_cells")


def test_partial_hit_raises_adjacent_probability():
    """
    A partial hit (ship not yet sunk) should raise probability for adjacent cells —
    the model should suggest continuing to hunt in that area.
    """
    model = load()

    # Baseline: totally empty board
    blank = board_input(*empty_channels())
    probs_blank = predict_board(model, blank)

    # One hit at (5,5) — ship NOT sunk (no sunk channel)
    hits, misses, sunk = empty_channels()
    hits[5, 5] = 1.0
    probs_hit = predict_board(model, board_input(hits, misses, sunk))

    # The cells immediately adjacent to the hit should have higher probability
    # than the same cells on a blank board
    neighbors = [(5, 4), (5, 6), (4, 5), (6, 5)]
    improved = sum(probs_hit[r, c] > probs_blank[r, c] for r, c in neighbors)
    assert improved >= 3, (
        f"Expected most neighbors of a hit to increase in probability, "
        f"but only {improved}/4 did"
    )
    print("PASS test_partial_hit_raises_adjacent_probability")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_sunk_cells_have_low_probability()
    test_sunk_cells_lower_than_open_cells()
    test_end_cells_lower_than_far_open_cells()
    test_partial_hit_raises_adjacent_probability()
    print("\nAll model behavior tests passed!")

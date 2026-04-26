"""Unit tests for cnn_model: ShipProbCNN, make_dataset, predict_board, save/load."""

import os
import tempfile
import numpy as np
import torch
from torch.utils.data import TensorDataset

from models.cnn_model import ShipProbCNN, make_dataset, predict_board, save_model, load_model


def test_ship_prob_cnn_forward_shape() -> None:
    """ShipProbCNN output shape should be (B, H, W)."""
    model = ShipProbCNN(in_channels=2, hidden=16)
    x = torch.randn(4, 2, 10, 10)
    out = model(x)
    assert out.shape == (4, 10, 10), f"Expected (4, 10, 10), got {out.shape}"
    print("✓ test_ship_prob_cnn_forward_shape passed")


def test_make_dataset_shapes_and_dtypes() -> None:
    """make_dataset should transpose NHWC→NCHW and return float32 tensors."""
    inputs = np.random.rand(8, 10, 10, 2).astype(np.float32)
    targets = np.random.rand(8, 10, 10).astype(np.float32)
    ds = make_dataset(inputs, targets)
    assert isinstance(ds, TensorDataset), "Expected TensorDataset"
    x, y = ds[0]
    assert x.shape == (2, 10, 10), f"Expected (2, 10, 10), got {x.shape}"
    assert y.shape == (10, 10), f"Expected (10, 10), got {y.shape}"
    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
    assert y.dtype == torch.float32, f"Expected float32, got {y.dtype}"
    print("✓ test_make_dataset_shapes_and_dtypes passed")


def test_predict_board_output() -> None:
    """predict_board should return (H, W) array with values in [0, 1]."""
    model = ShipProbCNN(in_channels=2, hidden=8)
    arr = np.random.rand(10, 10, 2).astype(np.float32)
    probs = predict_board(model, arr)
    assert probs.shape == (10, 10), f"Expected (10, 10), got {probs.shape}"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]"
    print("✓ test_predict_board_output passed")


def test_save_load_model_roundtrip() -> None:
    """save_model/load_model should preserve weights exactly."""
    model = ShipProbCNN(in_channels=2, hidden=8)
    original_weights = model.enc1.weight.data.clone()

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model.pt")
    try:
        save_model(model, path)
        loaded = load_model(path, in_channels=2, hidden=8)
        assert torch.allclose(original_weights, loaded.enc1.weight.data), (
            "Loaded model weights differ from saved weights"
        )
    finally:
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(tmpdir)
    print("✓ test_save_load_model_roundtrip passed")


if __name__ == "__main__":
    test_ship_prob_cnn_forward_shape()
    test_make_dataset_shapes_and_dtypes()
    test_predict_board_output()
    test_save_load_model_roundtrip()

    print("\n✅ All CNN model tests passed!")

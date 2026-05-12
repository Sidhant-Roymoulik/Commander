"""Lightweight CNN for predicting per-cell ship probabilities from hits/misses.

This module provides a small PyTorch model, dataset wrapper, training loop,
and prediction helpers. Input expected as numpy arrays with shape
`(rows, cols, channels)` where channels contains at least hits and misses.
"""

from typing import Optional
import numpy as np

import copy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
except Exception as e:
    raise ImportError(
        "PyTorch is required for cnn_model. Install with `pip install torch`"
    ) from e


class ShipProbCNN(nn.Module):
    def __init__(self, in_channels: int = 2, hidden: int = 32):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.mid = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.mid(x))
        x = F.relu(self.dec1(x))
        x = self.out_conv(x)
        return x.squeeze(1)  # (B, H, W)


def make_dataset(inputs: np.ndarray, targets: np.ndarray) -> "TensorDataset":
    """Wrap numpy arrays in a TensorDataset. NHWC inputs are transposed to NCHW."""
    inp = inputs.copy()
    if inp.ndim == 4 and inp.shape[-1] in (1, 2, 3):
        inp = np.transpose(inp, (0, 3, 1, 2))
    return TensorDataset(
        torch.from_numpy(inp.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )


def _run_eval(model: "nn.Module", loader: "DataLoader", criterion: "nn.Module", device: str) -> float:
    """Run one evaluation pass over loader, return average loss."""
    model.eval()
    loss_accum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.ndim == 3:
                xb = xb.unsqueeze(1)
            loss_accum += float(criterion(model(xb), yb).item()) * xb.size(0)
    return loss_accum / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
):
    """Train `model` on `train_dataset` and optionally report validation/test loss per epoch.

    Returns the trained model and a history dict with keys 'train', 'val', 'test' (lists of per-epoch losses).
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pin = device == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
        if test_dataset is not None
        else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train": [], "val": [], "test": []}
    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if xb.ndim == 3:
                xb = xb.unsqueeze(1)

            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_train_loss += float(loss.item()) * xb.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        history["train"].append(avg_train_loss)

        # validation
        avg_val_loss = None
        if val_loader is not None:
            avg_val_loss = _run_eval(model, val_loader, criterion, device)
            history["val"].append(avg_val_loss)

        # test + best-model checkpoint
        avg_test_loss = None
        if test_loader is not None:
            avg_test_loss = _run_eval(model, test_loader, criterion, device)
            history["test"].append(avg_test_loss)
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_state = copy.deepcopy(model.state_dict())

        # reporting
        report_parts = [f"Epoch {epoch}/{epochs}", f"train_loss: {avg_train_loss:.4f}"]
        if avg_val_loss is not None:
            report_parts.append(f"val_loss: {avg_val_loss:.4f}")
        if avg_test_loss is not None:
            marker = " *" if avg_test_loss == best_test_loss else ""
            report_parts.append(f"test_loss: {avg_test_loss:.4f}{marker}")

        print(" - ".join(report_parts))

    # Restore the best weights seen during training
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict_board(
    model: nn.Module, array: np.ndarray, device: Optional[str] = None
) -> np.ndarray:
    """Predict per-cell ship probability from a board tensor.

    Args:
        model: trained ShipProbCNN
        array: np.ndarray shape (H, W, C) in HWC layout, or (C, H, W) in CHW layout.
               C must match model's in_channels.
    Returns:
        probs: np.ndarray shape (H, W) with values in [0, 1]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    arr = array.copy()
    if arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    arr = arr.astype(np.float32)

    inp_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    return probs


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(
    path: str, in_channels: Optional[int] = None, hidden: Optional[int] = None, device: Optional[str] = None
) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    # Infer architecture from checkpoint weights so the call site doesn't need to know hyperparams
    detected_in_channels = in_channels or int(state["enc1.weight"].shape[1])
    detected_hidden      = hidden      or int(state["enc1.weight"].shape[0])
    model = ShipProbCNN(in_channels=detected_in_channels, hidden=detected_hidden)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

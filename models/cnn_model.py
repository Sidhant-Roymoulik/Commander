"""Lightweight CNN for predicting per-cell ship probabilities from hits/misses.

This module provides a small PyTorch model, dataset wrapper, training loop,
and prediction helpers. Input expected as numpy arrays with shape
`(rows, cols, channels)` where channels contains at least hits and misses.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
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


class NumpyBoardDataset(Dataset):
    """Dataset wrapping numpy arrays.

    inputs: np.ndarray shape (N, H, W, C) or (N, C, H, W)
    targets: np.ndarray shape (N, H, W) binary or float
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        if inputs.ndim == 4 and inputs.shape[-1] in (1, 2, 3):
            # convert NHWC -> NCHW
            inputs = np.transpose(inputs, (0, 3, 1, 2))
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if test_dataset is not None
        else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train": [], "val": [], "test": []}

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
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for xb_val, yb_val in val_loader:
                    xb_val = xb_val.to(device)
                    yb_val = yb_val.to(device)
                    if xb_val.ndim == 3:
                        xb_val = xb_val.unsqueeze(1)
                    logits_val = model(xb_val)
                    loss_val = criterion(logits_val, yb_val)
                    val_loss_accum += float(loss_val.item()) * xb_val.size(0)

            avg_val_loss = val_loss_accum / len(val_loader.dataset)
            history["val"].append(avg_val_loss)

        # test
        avg_test_loss = None
        if test_loader is not None:
            model.eval()
            test_loss_accum = 0.0
            with torch.no_grad():
                for xb_test, yb_test in test_loader:
                    xb_test = xb_test.to(device)
                    yb_test = yb_test.to(device)
                    if xb_test.ndim == 3:
                        xb_test = xb_test.unsqueeze(1)
                    logits_test = model(xb_test)
                    loss_test = criterion(logits_test, yb_test)
                    test_loss_accum += float(loss_test.item()) * xb_test.size(0)

            avg_test_loss = test_loss_accum / len(test_loader.dataset)
            history["test"].append(avg_test_loss)

        # reporting
        report_parts = [f"Epoch {epoch}/{epochs}", f"train_loss: {avg_train_loss:.4f}"]
        if avg_val_loss is not None:
            report_parts.append(f"val_loss: {avg_val_loss:.4f}")
        if avg_test_loss is not None:
            report_parts.append(f"test_loss: {avg_test_loss:.4f}")

        print(" - ".join(report_parts))

    return model, history


def predict_board(
    model: nn.Module, array: np.ndarray, device: Optional[str] = None
) -> np.ndarray:
    """Predict per-cell ship probability from a single board tensor.

    Args:
        model: trained ShipProbCNN
        array: np.ndarray shape (H, W, C) or (C, H, W)
    Returns:
        probs: np.ndarray shape (H, W) with values in [0, 1]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    arr = array.copy()
    if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3):
        arr = np.transpose(arr, (2, 0, 1))
    arr = arr.astype(np.float32)
    # take first two channels as hits/misses if present
    if arr.shape[0] >= 2:
        inp = arr[:2]
    else:
        # pad to 2 channels
        inp = np.pad(arr, ((0, 2 - arr.shape[0]), (0, 0), (0, 0)))

    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    return probs


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(
    path: str, in_channels: int = 2, hidden: int = 32, device: Optional[str] = None
) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ShipProbCNN(in_channels=in_channels, hidden=hidden)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

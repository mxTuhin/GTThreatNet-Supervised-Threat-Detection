"""
bilstm_model.py
================
PyTorch Bidirectional LSTM for temporal threat sequence classification.

Input : (batch, T=30, n_frame_features=8)  — per-frame feature sequences
Output: (batch, n_classes=4)               — class logits

Architecture:
  Bidirectional LSTM × 2 layers
  → Layer-norm on final hidden states
  → Dropout
  → Fully-connected classification head

Usage (as module):
    from src.models.bilstm_model import BiLSTMThreatClassifier, train_bilstm, predict_bilstm
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    BILSTM_HIDDEN, BILSTM_LAYERS, BILSTM_DROPOUT,
    BILSTM_EPOCHS, BILSTM_LR, BILSTM_BATCH,
    FRAME_FEATURES, CLASS_NAMES, RANDOM_SEED,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """X: (N, T, F)  y: (N,) int64"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTMThreatClassifier(nn.Module):

    def __init__(
        self,
        input_dim:  int = len(FRAME_FEATURES),
        hidden_dim: int = BILSTM_HIDDEN,
        n_layers:   int = BILSTM_LAYERS,
        n_classes:  int = len(CLASS_NAMES),
        dropout:    float = BILSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout        = dropout if n_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)   # ×2 for bidirectional
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)          # (B, T, 2H)
        # Use mean pooling over time (more robust than last hidden for short seqs)
        out = out.mean(dim=1)          # (B, 2H)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return self.fc(out)            # (B, n_classes)

    def attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns per-timestep importance as softmax of L2-norm of LSTM output.
        Shape: (T,). Used as proxy attention for xAI explanations.
        """
        self.eval()
        with torch.no_grad():
            out, _ = self.lstm(x)          # (B, T, 2H)
            norms  = out.norm(dim=-1)      # (B, T)
            weights = torch.softmax(norms, dim=-1)
        return weights.cpu().numpy()       # (B, T)


# ── Training utilities ────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_classes:   int   = len(CLASS_NAMES),
    epochs:      int   = BILSTM_EPOCHS,
    lr:          float = BILSTM_LR,
    batch_size:  int   = BILSTM_BATCH,
    save_path:   str | None = None,
) -> tuple["BiLSTMThreatClassifier", dict]:
    """
    Train BiLSTM and return (model, history).
    history = {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    torch.manual_seed(RANDOM_SEED)
    device = _get_device()
    print(f"  BiLSTM training on {device}  ({X_train.shape[0]} train / {X_val.shape[0]} val)")

    # Normalize sequences per feature
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std

    train_ds = SequenceDataset(X_train_n, y_train)
    val_ds   = SequenceDataset(X_val_n,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # Compute class weights for imbalance
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1), dtype=torch.float32).to(device)

    model     = BiLSTMThreatClassifier(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        epoch_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                correct  += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc   = correct / len(val_ds)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # Save model + normalisation stats
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "norm_mean":   mean,
            "norm_std":    std,
            "n_classes":   n_classes,
            "class_names": CLASS_NAMES,
        }, save_path)
        print(f"  Saved BiLSTM → {save_path}  (best val_acc={best_val_acc:.3f})")

    return model, history


def predict_bilstm(
    X: np.ndarray,
    checkpoint_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load saved checkpoint and predict.
    Returns (pred_labels: int array, pred_probs: float (N, n_classes)).
    """
    device = _get_device()
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_classes = ckpt["n_classes"]

    model = BiLSTMThreatClassifier(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    X_norm = (X - mean) / std
    ds = SequenceDataset(X_norm, np.zeros(len(X_norm), dtype=np.int64))
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs  = np.concatenate(all_probs, axis=0)
    labels = probs.argmax(axis=1)
    return labels, probs

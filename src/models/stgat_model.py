"""
stgat_model.py
===============
Spatio-Temporal Graph Attention Network (STGAT) for threat detection.

Architecture
------------
At each of T timesteps:
  2-layer Multi-Head GAT processes the person interaction graph.
  Target node embedding + mean-pooled global embedding → frame vector.
Across T frames:
  2-layer GRU → classification head.

Input
-----
  X     : (B, T, N, 4)    node features [cx, cy, vx, vy] (normalised at train time)
  A     : (B, T, N, N)    binary adjacency matrix (1 = edge, 0 = no edge / padding)
  valid : (B, N)  float   1 = real node, 0 = padded

XAI
---
  The GAT attention weights (B, T, N, N) are returned on every forward pass.
  attn[b, t, i, j] = how much node j influenced node i at timestep t.
  For threat detection, attn[:, :, 0, :] shows which persons the TARGET
  was attending to — average over T to get per-person importance scores.

No external graph library required (pure PyTorch).

Usage:
    from src.models.stgat_model import train_stgat, predict_stgat
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    STGAT_GAT_HIDDEN, STGAT_GAT_HEADS, STGAT_GRU_HIDDEN, STGAT_DROPOUT,
    STGAT_EPOCHS, STGAT_LR, STGAT_BATCH,
    GRAPH_NODE_DIM, GRAPH_N_MAX, CLASS_NAMES, RANDOM_SEED,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GraphDataset(Dataset):
    def __init__(self, X: np.ndarray, A: np.ndarray,
                 valid: np.ndarray, y: np.ndarray):
        self.X     = torch.from_numpy(X).float()
        self.A     = torch.from_numpy(A).float()
        self.valid = torch.from_numpy(valid.astype(np.float32))
        self.y     = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.valid[idx], self.y[idx]


# ── GAT layers ────────────────────────────────────────────────────────────────

class _GATHead(nn.Module):
    """Single-head graph attention."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W       = nn.Linear(in_dim, out_dim, bias=False)
        self.a       = nn.Linear(2 * out_dim, 1,  bias=False)
        self.leaky   = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x   : (B, N, in_dim)
        adj : (B, N, N)  — 1 = edge, 0 = no edge/padding
        Returns h (B, N, out_dim), alpha (B, N, N)
        """
        B, N, _ = x.shape
        Wh = self.W(x)                                    # (B, N, out)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)     # (B, N, N, out)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)     # (B, N, N, out)
        e    = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))  # (B,N,N)
        e    = e.masked_fill(adj < 0.5, -1e9)
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(adj < 0.5, 0.0)        # avoid NaN from -inf rows
        alpha = self.dropout(alpha)
        h = torch.bmm(alpha, Wh)                         # (B, N, out)
        return h, alpha


class _MultiHeadGAT(nn.Module):
    """Multi-head GAT: concatenates all heads, returns mean attention."""

    def __init__(self, in_dim: int, head_dim: int,
                 n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [_GATHead(in_dim, head_dim, dropout) for _ in range(n_heads)]
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        outputs, attns = zip(*[h(x, adj) for h in self.heads])
        out  = torch.cat(outputs, dim=-1)                 # (B, N, head_dim*n_heads)
        attn = torch.stack(attns, dim=0).mean(dim=0)     # (B, N, N) mean across heads
        return out, attn


# ── STGAT classifier ──────────────────────────────────────────────────────────

class STGATClassifier(nn.Module):
    """
    Spatio-Temporal Graph Attention Network.

    Forward returns (logits, attn_weights):
      logits       : (B, n_classes)
      attn_weights : (B, T, N, N)  — raw GAT attention for XAI
    """

    def __init__(
        self,
        node_dim:   int   = GRAPH_NODE_DIM,
        gat_hidden: int   = STGAT_GAT_HIDDEN,
        n_heads:    int   = STGAT_GAT_HEADS,
        gru_hidden: int   = STGAT_GRU_HIDDEN,
        n_classes:  int   = len(CLASS_NAMES),
        dropout:    float = STGAT_DROPOUT,
    ):
        super().__init__()
        gat_out = gat_hidden * n_heads   # 32 × 4 = 128

        self.gat1 = _MultiHeadGAT(node_dim, gat_hidden, n_heads, dropout)
        self.gat2 = _MultiHeadGAT(gat_out,  gat_hidden, n_heads, dropout)
        self.ln1  = nn.LayerNorm(gat_out)
        self.ln2  = nn.LayerNorm(gat_out)

        # Per-frame: target emb + global pool → 2 × gat_out
        self.gru = nn.GRU(
            gat_out * 2, gru_hidden,
            num_layers=2, batch_first=True,
            dropout=dropout,
        )
        self.ln_gru  = nn.LayerNorm(gru_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(
        self,
        x:     torch.Tensor,   # (B, T, N, F)
        adj:   torch.Tensor,   # (B, T, N, N)
        valid: torch.Tensor,   # (B, N)  float
    ):
        B, T, N, _ = x.shape
        frame_embs: list[torch.Tensor] = []
        attn_list:  list[torch.Tensor] = []

        for t in range(T):
            xt = x[:, t]     # (B, N, F)
            at = adj[:, t]   # (B, N, N)

            h1, _    = self.gat1(xt, at)
            h1 = F.elu(self.ln1(h1))
            h2, attn = self.gat2(h1, at)
            h2 = F.elu(self.ln2(h2))

            # Zero out padded nodes
            h2 = h2 * valid.unsqueeze(-1)              # (B, N, gat_out)

            target_emb = h2[:, 0, :]                   # (B, gat_out)
            n_valid    = valid.sum(1, keepdim=True).clamp(min=1)
            global_emb = h2.sum(1) / n_valid           # (B, gat_out)

            frame_embs.append(torch.cat([target_emb, global_emb], dim=-1))
            attn_list.append(attn)

        seq    = torch.stack(frame_embs, dim=1)        # (B, T, 2*gat_out)
        gru_out, _ = self.gru(seq)                     # (B, T, gru_hidden)
        last   = self.ln_gru(gru_out[:, -1])           # (B, gru_hidden)
        logits = self.fc(self.dropout(last))           # (B, n_classes)

        attn_stack = torch.stack(attn_list, dim=1)    # (B, T, N, N)
        return logits, attn_stack

    def get_attention_weights(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        valid: torch.Tensor,
    ) -> np.ndarray:
        """Return attention weights (B, T, N, N) without gradient tracking."""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, adj, valid)
        return attn.cpu().numpy()


# ── Normalisation ─────────────────────────────────────────────────────────────

def _fit_norm(X_train: np.ndarray):
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)   # (1,1,1,F)
    std  = X_train.std(axis=(0, 1, 2),  keepdims=True) + 1e-8
    return mean, std


def _apply_norm(X: np.ndarray, mean, std) -> np.ndarray:
    return (X - mean) / std


# ── Training ──────────────────────────────────────────────────────────────────

def train_stgat(
    X_train:     np.ndarray,
    A_train:     np.ndarray,
    valid_train: np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    A_val:       np.ndarray,
    valid_val:   np.ndarray,
    y_val:       np.ndarray,
    n_classes:   int   = len(CLASS_NAMES),
    epochs:      int   = STGAT_EPOCHS,
    lr:          float = STGAT_LR,
    batch_size:  int   = STGAT_BATCH,
    save_path:   str | None = None,
) -> tuple["STGATClassifier", dict]:
    """
    Train STGAT and return (model, history).
    history = {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  STGAT training on {device}  "
          f"({len(X_train)} train / {len(X_val)} val sequences)")

    mean, std = _fit_norm(X_train)
    X_tr_n = _apply_norm(X_train, mean, std)
    X_va_n = _apply_norm(X_val,   mean, std)

    train_ds = GraphDataset(X_tr_n, A_train, valid_train, y_train)
    val_ds   = GraphDataset(X_va_n, A_val,   valid_val,   y_val)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin)

    counts  = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1), dtype=torch.float32).to(device)

    model     = STGATClassifier(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history      = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, ab, vb, yb in train_loader:
            xb, ab, vb, yb = (xb.to(device), ab.to(device),
                               vb.to(device), yb.to(device))
            optimizer.zero_grad()
            logits, _ = model(xb, ab, vb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        epoch_loss /= len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, ab, vb, yb in val_loader:
                xb, ab, vb, yb = (xb.to(device), ab.to(device),
                                   vb.to(device), yb.to(device))
                logits, _ = model(xb, ab, vb)
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
                  f"train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "norm_mean":   mean,
            "norm_std":    std,
            "n_classes":   n_classes,
            "class_names": CLASS_NAMES,
        }, save_path)
        print(f"  Saved STGAT → {save_path}  (best val_acc={best_val_acc:.3f})")

    return model, history


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_stgat(
    X: np.ndarray,
    A: np.ndarray,
    valid: np.ndarray,
    checkpoint_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load checkpoint and run inference.

    Returns
    -------
    labels : (N,)          int predicted class indices
    probs  : (N, n_classes) float softmax probabilities
    attns  : (N, T, N_MAX, N_MAX) float GAT attention weights (for XAI)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_classes = ckpt["n_classes"]

    model = STGATClassifier(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    X_norm = _apply_norm(X, mean, std)

    ds     = GraphDataset(X_norm, A, valid, np.zeros(len(X_norm), dtype=np.int64))
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs, all_attns = [], []
    with torch.no_grad():
        for xb, ab, vb, _ in loader:
            logits, attn = model(xb.to(device), ab.to(device), vb.to(device))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_attns.append(attn.cpu().numpy())

    probs  = np.concatenate(all_probs)
    attns  = np.concatenate(all_attns)   # (N, T, N_MAX, N_MAX)
    labels = probs.argmax(axis=1)
    return labels, probs, attns

"""
explainability.py
==================
Explainable AI for the trained threat detection models.

  - SHAP TreeExplainer: RF and XGBoost
  - SHAP DeepExplainer / GradientExplainer: BiLSTM (optional, needs shap>=0.42)
  - Temporal attention weights for BiLSTM (proxy attribution)
  - Global + local (per-instance) explanations
  - Saves plots to data/outputs/eda/xai/

Usage:
    python src/xai/explainability.py --model rf
    python src/xai/explainability.py --model xgb
    python src/xai/explainability.py --model bilstm
    python src/xai/explainability.py --model all
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    THREAT_WINDOWS_CSV, SEQUENCE_WINDOWS_NPZ, MODELS_DIR, LABEL_ENCODER_PATH,
    SPLITS_CSV, WINDOW_FEATURES, CLASS_NAMES, EDA_OUT_DIR,
    GRAPH_SEQ_NPZ, GRAPH_N_MAX,
)

warnings.filterwarnings("ignore")
OUT = Path(EDA_OUT_DIR) / "xai"


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _load_window_data(features: list[str], split: str = "test"):
    df = pd.read_csv(THREAT_WINDOWS_CSV)
    le = joblib.load(LABEL_ENCODER_PATH)
    df = df[df["threat_type"].notna() & (df["threat_type"].str.strip() != "")].copy()
    df["threat_type"] = df["threat_type"].str.strip()
    df["y"] = le.transform(df["threat_type"])

    if Path(SPLITS_CSV).exists():
        splits = pd.read_csv(SPLITS_CSV)[["video_name", "split"]]
        df = df.merge(splits, on="video_name", how="left")
        df["split"] = df["split"].fillna("train")
    else:
        df["split"] = "train"

    df_split = df[df["split"] == split] if len(df[df["split"] == split]) > 0 else df
    feats = [f for f in features if f in df_split.columns]
    X = df_split[feats].fillna(0.0)
    y = df_split["y"].values
    return X, y, le, feats


# ── SHAP for tree-based models ────────────────────────────────────────────────

def explain_tree_model(model_name: str, model_path: Path, max_samples: int = 200):
    try:
        import shap
    except ImportError:
        print("[SKIP] shap not installed. Run: pip install shap")
        return

    if not model_path.exists():
        print(f"[SKIP] Model not found: {model_path}")
        return

    print(f"\n── SHAP for {model_name} ─────────────────────────────────")
    model = joblib.load(model_path)
    X, y, le, feats = _load_window_data(WINDOW_FEATURES)
    X_sample = X.sample(min(max_samples, len(X)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    classes     = le.classes_
    slug        = model_name.lower().replace(" ", "_")

    # ── Global: beeswarm / summary per class ──────────────────────────────────
    if isinstance(shap_values, list):
        # Multi-class: shap_values is list[n_classes] of (n_samples, n_feats)
        for i, cls in enumerate(classes):
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values[i], X_sample, feature_names=feats,
                plot_type="dot", show=False, max_display=15,
            )
            ax = plt.gca()
            ax.set_title(f"{model_name} — SHAP Summary: class '{cls}'")
            _save(plt.gcf(), f"{slug}_shap_summary_{cls}")

        # Combined mean |SHAP| bar chart
        mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        imp_df = pd.DataFrame({"feature": feats, "mean_|SHAP|": mean_shap})
        imp_df = imp_df.sort_values("mean_|SHAP|", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(5, len(feats) * 0.35)))
        ax.barh(imp_df["feature"], imp_df["mean_|SHAP|"])
        ax.set_title(f"{model_name} — Mean |SHAP| (all classes)")
        ax.set_xlabel("Mean |SHAP value|")
        _save(fig, f"{slug}_shap_mean_importance")

    else:
        # Binary: shap_values is (n_samples, n_feats)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feats,
                          plot_type="dot", show=False, max_display=15)
        plt.gca().set_title(f"{model_name} — SHAP Summary")
        _save(plt.gcf(), f"{slug}_shap_summary")

    # ── Local: waterfall for one example per class ────────────────────────────
    for i, cls in enumerate(classes):
        cls_mask = y == i
        if cls_mask.sum() == 0:
            continue
        idx = int(np.where(cls_mask)[0][0])
        if isinstance(shap_values, list):
            sv = shap_values[i][idx]
        else:
            sv = shap_values[idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = range(len(feats))
        colors = ["tomato" if s < 0 else "steelblue" for s in sv]
        ax.barh(y_pos, sv, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{model_name} — SHAP Waterfall: class '{cls}' (sample #{idx})")
        ax.set_xlabel("SHAP value")
        _save(fig, f"{slug}_shap_waterfall_{cls}")

    print(f"  SHAP analysis complete for {model_name}")


# ── BiLSTM attention visualization ───────────────────────────────────────────

def explain_bilstm(model_dir: Path, max_samples: int = 50):
    model_path = model_dir / "bilstm_model.pt"
    if not model_path.exists():
        print(f"[SKIP] BiLSTM not found: {model_path}")
        return

    if not Path(SEQUENCE_WINDOWS_NPZ).exists():
        print(f"[SKIP] Sequence data not found: {SEQUENCE_WINDOWS_NPZ}")
        return

    print("\n── BiLSTM Attention Attribution ──────────────────────────")
    import torch
    from src.models.bilstm_model import BiLSTMThreatClassifier

    data = np.load(str(SEQUENCE_WINDOWS_NPZ), allow_pickle=True)
    X, y_true = data["X"], data["y"]
    le = joblib.load(LABEL_ENCODER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(str(model_path), map_location=device)
    n_cls  = ckpt["n_classes"]
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]

    model = BiLSTMThreatClassifier(n_classes=n_cls).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_norm = (X - mean) / std
    sample = min(max_samples, len(X_norm))
    X_t = torch.from_numpy(X_norm[:sample]).float().to(device)

    attn_weights = model.attention_weights(X_t)   # (sample, T)
    classes = le.classes_

    # ── Mean attention per class ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(classes)))
    for i, cls in enumerate(classes):
        mask = y_true[:sample] == i
        if mask.sum() == 0:
            continue
        mean_attn = attn_weights[mask].mean(0)
        ax.plot(mean_attn, label=cls, color=palette[i], linewidth=2)
    ax.set_xlabel("Frame (within 30-frame window)")
    ax.set_ylabel("Attention weight")
    ax.set_title("BiLSTM — Mean Temporal Attention per Class")
    ax.legend()
    _save(fig, "bilstm_attention_per_class")

    # ── Heatmap: attention for each class ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        mask = y_true[:sample] == i
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        im = ax.imshow(attn_weights[mask], aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_title(cls)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Sample")
        plt.colorbar(im, ax=ax)
    fig.suptitle("BiLSTM — Attention Heatmaps per Class")
    fig.tight_layout()
    _save(fig, "bilstm_attention_heatmaps")

    # ── SHAP DeepExplainer (optional) ────────────────────────────────────────
    try:
        import shap
        background = X_t[:min(20, sample)]
        e = shap.DeepExplainer(model, background)
        shap_vals = e.shap_values(X_t[:10])   # (n_classes, 10, T, F)

        # Mean |SHAP| per frame feature, averaged over classes and samples
        mean_abs = np.mean([np.abs(sv).mean(axis=(0, 1)) for sv in shap_vals], axis=0)
        from config import FRAME_FEATURES
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(FRAME_FEATURES, mean_abs)
        ax.set_title("BiLSTM — Mean |SHAP| per Frame Feature")
        ax.set_ylabel("Mean |SHAP|")
        ax.tick_params(axis="x", rotation=45)
        _save(fig, "bilstm_shap_feature_importance")
    except Exception as e:
        print(f"  [INFO] SHAP DeepExplainer skipped: {e}")

    print("  BiLSTM xAI complete")


# ── STGAT GAT-attention XAI ──────────────────────────────────────────────────

def explain_stgat(model_dir: Path, max_samples: int = 100):
    """
    Visualise GAT attention weights as XAI for STGAT.

    attn[b, t, i, j] = how much node j influenced node i at timestep t.
    We focus on row 0 (target node): attn[:, :, 0, 1:] shows how much
    each nearby person influenced the target's representation.

    Produces:
      stgat_person_importance_<class>.png  — mean attention from target to each node slot
      stgat_attention_timeline_<class>.png — attention over time per node slot
    """
    model_path = model_dir / "stgat_model.pt"
    if not model_path.exists():
        print(f"[SKIP] STGAT model not found: {model_path}")
        return
    if not Path(GRAPH_SEQ_NPZ).exists():
        print(f"[SKIP] Graph sequence data not found: {GRAPH_SEQ_NPZ}")
        return

    print("\n── STGAT GAT Attention XAI ───────────────────────────────")
    import torch
    from src.models.stgat_model import predict_stgat

    data  = np.load(str(GRAPH_SEQ_NPZ), allow_pickle=True)
    X, A, valid, y_true = data["X"], data["A"], data["valid"], data["y"]
    le      = joblib.load(LABEL_ENCODER_PATH)
    classes = le.classes_

    sample  = min(max_samples, len(X))
    _, _, attns = predict_stgat(
        X[:sample], A[:sample], valid[:sample], str(model_path)
    )
    # attns: (N, T, N_MAX, N_MAX)
    # Target-to-others: attns[:, :, 0, :]  shape (N, T, N_MAX)
    target_attn = attns[:, :, 0, :]    # (N, T, N_MAX)
    y_s = y_true[:sample]
    T   = target_attn.shape[1]
    N   = GRAPH_N_MAX

    node_labels = ["target"] + [f"person {i}" for i in range(1, N)]

    # ── Mean person importance per class ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                              sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        mask = y_s == i
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        mean_imp = target_attn[mask].mean(axis=(0, 1))   # (N_MAX,)
        colors   = ["steelblue" if j == 0 else "tomato" for j in range(N)]
        ax.bar(node_labels, mean_imp, color=colors)
        ax.set_title(f"Class: {cls}")
        ax.set_xlabel("Node")
        ax.tick_params(axis="x", rotation=45)
        if i == 0:
            ax.set_ylabel("Mean GAT attention")
    fig.suptitle("STGAT — Target Node's Attention to Each Person (mean over time)")
    fig.tight_layout()
    _save(fig, "stgat_person_importance")

    # ── Attention timeline per class ─────────────────────────────────────────
    palette = plt.cm.get_cmap("tab10")(np.linspace(0, 1, N))
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                              sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        mask = y_s == i
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        mean_over_time = target_attn[mask].mean(axis=0)  # (T, N_MAX)
        for ni in range(1, N):                            # skip target self-attn
            ax.plot(mean_over_time[:, ni], label=node_labels[ni],
                    color=palette[ni], alpha=0.8)
        ax.set_title(f"Class: {cls}")
        ax.set_xlabel("Frame (within window)")
        if i == 0:
            ax.set_ylabel("Mean attention")
        if i == len(classes) - 1:
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("STGAT — Attention from Target to Each Nearby Person Over Time")
    fig.tight_layout()
    _save(fig, "stgat_attention_timeline")

    print("  STGAT xAI complete")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="xAI explanations for trained models")
    parser.add_argument("--model",       default="all",
                        choices=["xgb", "bilstm", "stgat", "all"])
    parser.add_argument("--model-dir",   default=str(MODELS_DIR))
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()
    mdir = Path(args.model_dir)

    if args.model in ("xgb", "all"):
        explain_tree_model("XGBoost",   mdir / "xgb_model.pkl", args.max_samples)
    if args.model in ("bilstm", "all"):
        explain_bilstm(mdir, args.max_samples)
    if args.model in ("stgat", "all"):
        explain_stgat(mdir, args.max_samples)


if __name__ == "__main__":
    main()

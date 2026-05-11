"""
post_train_eda.py
==================
Post-training EDA: per-class confidence, error analysis, per-video accuracy,
and prediction timeline — for XGBoost, BiLSTM, and STGAT.

Usage:
    python src/evaluation/post_train_eda.py --model xgb
    python src/evaluation/post_train_eda.py --model bilstm
    python src/evaluation/post_train_eda.py --model stgat
    python src/evaluation/post_train_eda.py --model all
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
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    THREAT_WINDOWS_CSV, SEQUENCE_WINDOWS_NPZ, GRAPH_SEQ_NPZ,
    MODELS_DIR, LABEL_ENCODER_PATH,
    SPLITS_CSV, WINDOW_FEATURES, CLASS_NAMES, EDA_OUT_DIR,
)

warnings.filterwarnings("ignore")
OUT = Path(EDA_OUT_DIR) / "post_train"

_DISPLAY_CLASS = {
    "normal":       "Normal",
    "following":    "Following",
    "surrounding":  "Surrounding",
    "fast_approach": "Fast Approach",
}


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _parse_name(name: str) -> tuple[str, int]:
    """Parse 'videoname_w3_t5' → (video_name, window_idx)."""
    parts = name.rsplit("_w", 1)
    video = parts[0]
    w_idx = 0
    if len(parts) > 1:
        w_part = parts[1].split("_")[0]
        if w_part.isdigit():
            w_idx = int(w_part)
    return video, w_idx


def _test_mask(names: list, split_key: str = "test") -> np.ndarray:
    """Return boolean mask for test-split samples based on splits.csv."""
    if Path(SPLITS_CSV).exists():
        splits_df = pd.read_csv(SPLITS_CSV).set_index("video_name")["split"].to_dict()
        split_arr = np.array([splits_df.get(_parse_name(n)[0], "train") for n in names])
    else:
        split_arr = np.array(["train"] * len(names))
    mask = split_arr == split_key
    if mask.sum() == 0:
        mask = np.ones(len(names), dtype=bool)
    return mask


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_xgb_predictions(model_path: Path):
    """Load XGBoost model and return (df, le, feature_list)."""
    df = pd.read_csv(THREAT_WINDOWS_CSV)
    le = joblib.load(LABEL_ENCODER_PATH)
    df = df[df["threat_type"].notna() & (df["threat_type"].str.strip() != "")].copy()
    df["threat_type"] = df["threat_type"].str.strip()
    df["y_true"] = le.transform(df["threat_type"])

    feats  = [f for f in WINDOW_FEATURES if f in df.columns]
    X      = df[feats].fillna(0.0).values
    model  = joblib.load(model_path)
    df["y_pred"]      = model.predict(X)
    df["y_pred_name"] = le.inverse_transform(df["y_pred"])
    proba = model.predict_proba(X)
    df["pred_confidence"] = proba.max(axis=1)
    df["correct"] = df["y_true"] == df["y_pred"]

    # Restrict to test split
    if Path(SPLITS_CSV).exists():
        splits = pd.read_csv(SPLITS_CSV)[["video_name", "split"]]
        df = df.merge(splits, on="video_name", how="left")
        df["split"] = df["split"].fillna("train")
        test_df = df[df["split"] == "test"]
        if len(test_df) > 0:
            df = test_df
    return df, le, feats


def load_bilstm_predictions(model_path: Path):
    """Load BiLSTM model and return (df, le) for test split."""
    if not model_path.exists():
        print(f"[SKIP] BiLSTM model not found: {model_path}")
        return None
    if not Path(SEQUENCE_WINDOWS_NPZ).exists():
        print(f"[SKIP] Sequence data not found: {SEQUENCE_WINDOWS_NPZ}")
        return None

    from src.models.bilstm_model import predict_bilstm

    data = np.load(str(SEQUENCE_WINDOWS_NPZ), allow_pickle=True)
    X, y_true, names = data["X"], data["y"], list(data["names"])
    le = joblib.load(LABEL_ENCODER_PATH)

    mask    = _test_mask(names)
    idxs    = np.where(mask)[0]
    X_te    = X[mask]
    y_te    = y_true[mask]
    names_te = [names[i] for i in idxs]

    y_pred, y_proba = predict_bilstm(X_te, str(model_path))

    video_names, w_idxs = zip(*[_parse_name(n) for n in names_te]) if names_te else ([], [])
    df = pd.DataFrame({
        "video_name":       list(video_names),
        "start_frame":      list(w_idxs),
        "y_true":           y_te,
        "y_pred":           y_pred,
        "y_pred_name":      [le.classes_[p] for p in y_pred],
        "threat_type":      [le.classes_[i] for i in y_te],
        "pred_confidence":  y_proba.max(axis=1),
        "correct":          y_te == y_pred,
    })
    return df, le


def load_stgat_predictions(model_path: Path):
    """Load STGAT model and return (df, le) for test split."""
    if not model_path.exists():
        print(f"[SKIP] STGAT model not found: {model_path}")
        return None
    if not Path(GRAPH_SEQ_NPZ).exists():
        print(f"[SKIP] Graph sequence data not found: {GRAPH_SEQ_NPZ}")
        return None

    from src.models.stgat_model import predict_stgat

    data  = np.load(str(GRAPH_SEQ_NPZ), allow_pickle=True)
    X, A, valid, y_true = data["X"], data["A"], data["valid"], data["y"]
    names = list(data["names"])
    le    = joblib.load(LABEL_ENCODER_PATH)

    mask    = _test_mask(names)
    idxs    = np.where(mask)[0]
    y_te    = y_true[mask]
    names_te = [names[i] for i in idxs]

    y_pred, y_proba, _ = predict_stgat(X[mask], A[mask], valid[mask], str(model_path))

    video_names, w_idxs = zip(*[_parse_name(n) for n in names_te]) if names_te else ([], [])
    df = pd.DataFrame({
        "video_name":       list(video_names),
        "start_frame":      list(w_idxs),
        "y_true":           y_te,
        "y_pred":           y_pred,
        "y_pred_name":      [le.classes_[p] for p in y_pred],
        "threat_type":      [le.classes_[i] for i in y_te],
        "pred_confidence":  y_proba.max(axis=1),
        "correct":          y_te == y_pred,
    })
    return df, le


# ── Plots ─────────────────────────────────────────────────────────────────────

def per_class_confidence(df: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("Set2", len(CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        subset = df[df["threat_type"] == cls]["pred_confidence"]
        if len(subset) == 0:
            continue
        ax.hist(subset, bins=20, alpha=0.5, color=palette[i], label=cls, density=True)
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Prediction Confidence per Class")
    ax.legend()
    _save(fig, f"{model_name}_confidence_dist")


def error_analysis(df: pd.DataFrame, feats: list, model_name: str):
    """Compare feature distributions: correct vs incorrect (XGBoost only)."""
    correct   = df[df["correct"]]
    incorrect = df[~df["correct"]]
    if len(incorrect) == 0:
        print("  No errors found — skipping error analysis")
        return

    n_feats   = min(10, len(feats))
    top_feats = feats[:n_feats]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        ax = axes[i]
        ax.hist(correct[feat].dropna(),   bins=20, alpha=0.6, label="Correct",   density=True, color="steelblue")
        ax.hist(incorrect[feat].dropna(), bins=20, alpha=0.6, label="Incorrect", density=True, color="tomato")
        ax.set_title(feat, fontsize=8)
        ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{model_name} — Feature Distributions: Correct vs Incorrect Predictions")
    fig.tight_layout()
    _save(fig, f"{model_name}_error_analysis")


def per_video_accuracy(df: pd.DataFrame, model_name: str):
    video_acc   = df.groupby("video_name")["correct"].mean().sort_values()
    video_class = df.groupby("video_name")["threat_type"].first()
    labels = [
        f"[{_DISPLAY_CLASS.get(video_class.get(v, ''), video_class.get(v, ''))}] {v}"
        for v in video_acc.index
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(video_acc) * 0.5), 5))
    colors = ["tomato" if a < 0.7 else "steelblue" for a in video_acc.values]
    ax.barh(range(len(video_acc)), video_acc.values, color=colors)
    ax.set_yticks(range(len(video_acc)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.7, label="70% threshold")
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{model_name} — Per-Video Accuracy")
    ax.legend()
    _save(fig, f"{model_name}_per_video_accuracy")


def prediction_timeline(df: pd.DataFrame, model_name: str, max_videos: int = 4):
    """Show predicted class over window index for a few videos."""
    videos  = df["video_name"].unique()[:max_videos]
    palette = {cls: c for cls, c in zip(CLASS_NAMES,
               sns.color_palette("Set2", len(CLASS_NAMES)))}
    fig, axes = plt.subplots(len(videos), 1,
                             figsize=(14, 3 * len(videos)), squeeze=False)
    for i, vid in enumerate(videos):
        vdf = df[df["video_name"] == vid].sort_values("start_frame")
        ax  = axes[i, 0]
        ax.scatter(vdf["start_frame"], vdf["y_pred"],
                   c=[palette.get(n, "gray") for n in vdf["y_pred_name"]],
                   s=15, alpha=0.7)
        ax.plot(vdf["start_frame"], vdf["y_true"], "k--", alpha=0.4,
                label="Ground truth")
        cls = vdf["threat_type"].iloc[0] if len(vdf) > 0 else ""
        cls_display = _DISPLAY_CLASS.get(cls, cls)
        ax.set_title(f"[{cls_display}] {vid}", fontsize=9)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax.set_xlabel("Window index")
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"{model_name} — Prediction Timeline per Video", fontsize=11)
    fig.tight_layout()
    _save(fig, f"{model_name}_prediction_timeline")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_xgb_eda(mdir: Path):
    mpath = mdir / "xgb_model.pkl"
    if not mpath.exists():
        print(f"[SKIP] XGBoost model not found: {mpath}")
        return
    print(f"\nPost-Training EDA for XGBoost → {OUT}")
    df, le, feats = load_xgb_predictions(mpath)
    per_class_confidence(df, "xgb")
    error_analysis(df, feats, "xgb")
    per_video_accuracy(df, "xgb")
    prediction_timeline(df, "xgb")


def run_bilstm_eda(mdir: Path):
    mpath = mdir / "bilstm_model.pt"
    result = load_bilstm_predictions(mpath)
    if result is None:
        return
    print(f"\nPost-Training EDA for BiLSTM → {OUT}")
    df, le = result
    per_class_confidence(df, "bilstm")
    per_video_accuracy(df, "bilstm")
    prediction_timeline(df, "bilstm")


def run_stgat_eda(mdir: Path):
    mpath = mdir / "stgat_model.pt"
    result = load_stgat_predictions(mpath)
    if result is None:
        return
    print(f"\nPost-Training EDA for STGAT → {OUT}")
    df, le = result
    per_class_confidence(df, "stgat")
    per_video_accuracy(df, "stgat")
    prediction_timeline(df, "stgat")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb",
                        choices=["xgb", "bilstm", "stgat", "all"])
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    mdir = Path(args.model_dir)

    if args.model in ("xgb", "all"):
        run_xgb_eda(mdir)
    if args.model in ("bilstm", "all"):
        run_bilstm_eda(mdir)
    if args.model in ("stgat", "all"):
        run_stgat_eda(mdir)

    print("Post-Training EDA complete.\n")


if __name__ == "__main__":
    main()

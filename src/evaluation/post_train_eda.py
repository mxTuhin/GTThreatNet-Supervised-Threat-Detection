"""
post_train_eda.py
==================
Post-training EDA: per-class feature value analysis conditioned on model predictions,
error analysis, per-video performance, temporal prediction patterns.

Usage:
    python src/evaluation/post_train_eda.py
    python src/evaluation/post_train_eda.py --model rf
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
    THREAT_WINDOWS_CSV, MODELS_DIR, LABEL_ENCODER_PATH,
    SPLITS_CSV, WINDOW_FEATURES, CLASS_NAMES, EDA_OUT_DIR,
)

warnings.filterwarnings("ignore")
OUT = Path(EDA_OUT_DIR) / "post_train"


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def load_predictions(model_path: Path, features: list[str]):
    df = pd.read_csv(THREAT_WINDOWS_CSV)
    le = joblib.load(LABEL_ENCODER_PATH)
    df = df[df["threat_type"].notna() & (df["threat_type"].str.strip() != "")].copy()
    df["threat_type"] = df["threat_type"].str.strip()
    df["y_true"] = le.transform(df["threat_type"])

    feats  = [f for f in features if f in df.columns]
    X      = df[feats].fillna(0.0).values
    model  = joblib.load(model_path)
    df["y_pred"]  = model.predict(X)
    df["y_pred_name"] = le.inverse_transform(df["y_pred"])
    proba = model.predict_proba(X)
    df["pred_confidence"] = proba.max(axis=1)
    df["correct"] = df["y_true"] == df["y_pred"]

    return df, le, feats


def per_class_confidence(df, model_name):
    """Distribution of prediction confidence per class."""
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


def error_analysis(df, feats, model_name):
    """Compare feature distributions: correct vs incorrect predictions."""
    correct   = df[df["correct"]]
    incorrect = df[~df["correct"]]
    if len(incorrect) == 0:
        print("  No errors found — skipping error analysis")
        return

    n_feats = min(10, len(feats))
    top_feats = feats[:n_feats]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        ax = axes[i]
        ax.hist(correct[feat].dropna(),   bins=20, alpha=0.6, label="Correct",   density=True, color="steelblue")
        ax.hist(incorrect[feat].dropna(), bins=20, alpha=0.6, label="Incorrect", density=True, color="tomato")
        ax.set_title(feat, fontsize=8)
        ax.legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{model_name} — Feature Distributions: Correct vs Incorrect Predictions")
    fig.tight_layout()
    _save(fig, f"{model_name}_error_analysis")


def per_video_accuracy(df, model_name):
    """Bar chart of per-video accuracy."""
    video_acc = df.groupby("video_name")["correct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(max(8, len(video_acc) * 0.5), 5))
    colors = ["tomato" if a < 0.7 else "steelblue" for a in video_acc.values]
    ax.barh(range(len(video_acc)), video_acc.values, color=colors)
    ax.set_yticks(range(len(video_acc)))
    ax.set_yticklabels(video_acc.index, fontsize=8)
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.7, label="70% threshold")
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{model_name} — Per-Video Accuracy")
    ax.legend()
    _save(fig, f"{model_name}_per_video_accuracy")


def prediction_timeline(df, model_name, max_videos: int = 4):
    """
    For a few videos, show the predicted class over time (window index).
    Lets you see if the model is stable or flickering.
    """
    videos = df["video_name"].unique()[:max_videos]
    palette = {cls: c for cls, c in zip(CLASS_NAMES, sns.color_palette("Set2", len(CLASS_NAMES)))}
    fig, axes = plt.subplots(len(videos), 1, figsize=(14, 3 * len(videos)), squeeze=False)
    for i, vid in enumerate(videos):
        vdf = df[df["video_name"] == vid].sort_values("start_frame")
        ax  = axes[i, 0]
        ax.scatter(vdf["start_frame"], vdf["y_pred"],
                   c=[palette[n] for n in vdf["y_pred_name"]], s=15, alpha=0.7)
        ax.plot(vdf["start_frame"], vdf["y_true"], "k--", alpha=0.4, label="Ground truth")
        ax.set_title(f"{vid}", fontsize=9)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax.set_xlabel("Start Frame")
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"{model_name} — Prediction Timeline per Video", fontsize=11)
    fig.tight_layout()
    _save(fig, f"{model_name}_prediction_timeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rf", choices=["rf", "xgb"])
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    mdir  = Path(args.model_dir)
    fname = "rf_model.pkl" if args.model == "rf" else "xgb_model.pkl"
    mpath = mdir / fname

    if not mpath.exists():
        print(f"[ERROR] Model not found: {mpath}")
        return

    print(f"\nPost-Training EDA for {args.model.upper()} → {OUT}")
    df, le, feats = load_predictions(mpath, WINDOW_FEATURES)

    per_class_confidence(df, args.model)
    error_analysis(df, feats, args.model)
    per_video_accuracy(df, args.model)
    prediction_timeline(df, args.model)
    print("Post-Training EDA complete.\n")


if __name__ == "__main__":
    main()

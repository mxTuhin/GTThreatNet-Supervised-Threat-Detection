"""
evaluate.py
============
Evaluates trained models on the held-out test split.
Generates: confusion matrix, ROC curves, PR curves, per-class metrics.

Usage:
    python src/evaluation/evaluate.py --model rf
    python src/evaluation/evaluate.py --model xgb
    python src/evaluation/evaluate.py --model bilstm
    python src/evaluation/evaluate.py --model all
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    THREAT_WINDOWS_CSV, SEQUENCE_WINDOWS_NPZ, MODELS_DIR, LABEL_ENCODER_PATH,
    SPLITS_CSV, WINDOW_FEATURES, CLASS_NAMES, EDA_OUT_DIR,
)

warnings.filterwarnings("ignore")
OUT = Path(EDA_OUT_DIR) / "evaluation"


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def load_test_windows(features: list[str]):
    df = pd.read_csv(THREAT_WINDOWS_CSV)
    le = joblib.load(LABEL_ENCODER_PATH)
    df = df[df["threat_type"].notna() & (df["threat_type"].str.strip() != "")].copy()
    df["threat_type"] = df["threat_type"].str.strip()
    df["y"] = le.transform(df["threat_type"])

    splits = pd.read_csv(SPLITS_CSV)[["video_name", "split"]] if Path(SPLITS_CSV).exists() else None
    if splits is not None:
        df = df.merge(splits, on="video_name", how="left")
        df["split"] = df["split"].fillna("train")
    else:
        df["split"] = "train"

    df_test = df[df["split"] == "test"]
    if len(df_test) == 0:
        print("[WARN] No test rows found — evaluating on all data")
        df_test = df

    feats = [f for f in features if f in df_test.columns]
    X_te  = df_test[feats].fillna(0.0).values
    y_te  = df_test["y"].values
    return X_te, y_te, le


def plot_confusion_matrix(y_true, y_pred, classes, title, fname):
    cm  = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{title} — Raw Counts")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes,
                cmap="Blues", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title(f"{title} — Normalized")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")
    fig.tight_layout()
    _save(fig, fname)


def plot_roc_curves(y_true, y_proba, classes, title, fname):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    palette = sns.color_palette("Set2", len(classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[i], label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} — ROC Curves")
    ax.legend(loc="lower right")
    _save(fig, fname)


def plot_pr_curves(y_true, y_proba, classes, title, fname):
    y_bin   = label_binarize(y_true, classes=list(range(len(classes))))
    palette = sns.color_palette("Set2", len(classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=palette[i], label=f"{cls} (AUC={pr_auc:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} — Precision-Recall Curves")
    ax.legend(loc="upper right")
    _save(fig, fname)


def evaluate_sklearn_model(model_name: str, model_path: Path, features: list[str]):
    if not model_path.exists():
        print(f"[SKIP] Model not found: {model_path}")
        return

    print(f"\n── Evaluating {model_name} ───────────────────────────────")
    model  = joblib.load(model_path)
    X_te, y_te, le = load_test_windows(features)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    classes = le.classes_

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    print(f"  Test accuracy:  {acc:.3f}")
    print(f"  Test F1 (wtd):  {f1:.3f}")
    print("\n" + classification_report(y_te, y_pred, target_names=classes, zero_division=0))

    slug = model_name.lower().replace(" ", "_")
    plot_confusion_matrix(y_te, y_pred, classes, model_name, f"{slug}_confusion")
    plot_roc_curves(y_te, y_proba, classes, model_name, f"{slug}_roc")
    plot_pr_curves(y_te, y_proba,  classes, model_name, f"{slug}_pr")


def evaluate_bilstm(model_dir: Path):
    model_path = model_dir / "bilstm_model.pt"
    if not model_path.exists():
        print(f"[SKIP] BiLSTM model not found: {model_path}")
        return

    print("\n── Evaluating BiLSTM ─────────────────────────────────────")
    if not Path(SEQUENCE_WINDOWS_NPZ).exists():
        print(f"[SKIP] Sequence data not found: {SEQUENCE_WINDOWS_NPZ}")
        return

    from src.models.bilstm_model import predict_bilstm
    data = np.load(str(SEQUENCE_WINDOWS_NPZ), allow_pickle=True)
    X, y_true, names = data["X"], data["y"], list(data["names"])

    # Get test split
    if Path(SPLITS_CSV).exists():
        splits_df = pd.read_csv(SPLITS_CSV).set_index("video_name")["split"].to_dict()
        def get_split(name):
            video = name.split("_w")[0]
            return splits_df.get(video, "train")
        split_arr = np.array([get_split(n) for n in names])
    else:
        split_arr = np.array(["train"] * len(names))

    mask   = split_arr == "test"
    if mask.sum() == 0:
        print("[WARN] No test sequences — evaluating on all data")
        mask = np.ones(len(names), dtype=bool)

    X_te, y_te = X[mask], y_true[mask]
    le = joblib.load(LABEL_ENCODER_PATH)
    classes = le.classes_

    y_pred, y_proba = predict_bilstm(X_te, str(model_path))
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    print(f"  Test accuracy:  {acc:.3f}")
    print(f"  Test F1 (wtd):  {f1:.3f}")
    print("\n" + classification_report(y_te, y_pred, target_names=classes, zero_division=0))

    plot_confusion_matrix(y_te, y_pred, classes, "BiLSTM", "bilstm_confusion")
    plot_roc_curves(y_te, y_proba,  classes, "BiLSTM", "bilstm_roc")
    plot_pr_curves(y_te, y_proba,   classes, "BiLSTM", "bilstm_pr")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained threat models")
    parser.add_argument("--model", default="all", choices=["rf", "xgb", "bilstm", "all"])
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    args  = parser.parse_args()
    mdir  = Path(args.model_dir)
    feats = WINDOW_FEATURES

    if args.model in ("rf", "all"):
        evaluate_sklearn_model("Random Forest", mdir / "rf_model.pkl",  feats)
    if args.model in ("xgb", "all"):
        evaluate_sklearn_model("XGBoost",       mdir / "xgb_model.pkl", feats)
    if args.model in ("bilstm", "all"):
        evaluate_bilstm(mdir)


if __name__ == "__main__":
    main()

"""
train_models.py
================
Trains RF, XGBoost, and BiLSTM classifiers on the extracted threat windows.
Proper train/val/test split at VIDEO level. No data leakage.

Usage:
    python src/models/train_models.py --model rf
    python src/models/train_models.py --model xgb
    python src/models/train_models.py --model bilstm
    python src/models/train_models.py --model all      # trains all three
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    THREAT_WINDOWS_CSV, SEQUENCE_WINDOWS_NPZ, MODELS_DIR, LABEL_ENCODER_PATH,
    SPLITS_CSV, WINDOW_FEATURES, CLASS_NAMES, THREAT_TYPE_MAP,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR,
    CV_FOLDS, N_JOBS, RANDOM_SEED,
    BILSTM_EPOCHS, BILSTM_LR, BILSTM_BATCH,
    GRAPH_SEQ_NPZ, STGAT_EPOCHS, STGAT_LR, STGAT_BATCH,
)

warnings.filterwarnings("ignore")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_window_data(csv_path: str, splits_csv: str | None = None):
    """
    Load feature CSV, apply video-level splits if splits.csv exists.
    Returns (df_train, df_val, df_test, label_encoder).
    """
    df = pd.read_csv(csv_path)
    df = df[df["threat_type"].notna() & (df["threat_type"].astype(str).str.strip() != "")].copy()
    df["threat_type"] = df["threat_type"].str.strip()

    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)
    df["y"] = le.transform(df["threat_type"])

    # Apply video-level split
    if splits_csv and Path(splits_csv).exists():
        splits = pd.read_csv(splits_csv)[["video_name", "split"]]
        df = df.merge(splits, on="video_name", how="left")
        df["split"] = df["split"].fillna("train")
    else:
        # Fallback: assign splits by video name hash for reproducibility
        rng = np.random.default_rng(RANDOM_SEED)
        unique_videos = df["video_name"].unique()
        rng.shuffle(unique_videos)
        n = len(unique_videos)
        n_train = max(1, int(n * 0.70))
        n_val   = max(1, int(n * 0.15))
        split_map = {}
        for i, v in enumerate(unique_videos):
            if i < n_train:
                split_map[v] = "train"
            elif i < n_train + n_val:
                split_map[v] = "val"
            else:
                split_map[v] = "test"
        df["split"] = df["video_name"].map(split_map)

    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test"].copy()

    print(f"Split sizes: train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
    return df_train, df_val, df_test, le


def _get_Xy(df: pd.DataFrame, features: list[str]):
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(0.0).values
    y = df["y"].values
    return X, y


# ── STGAT ─────────────────────────────────────────────────────────────────────

def train_stgat_model(npz_path: str, splits_csv: str, le, save_dir: Path):
    from src.models.stgat_model import train_stgat

    print("\n── STGAT (Spatio-Temporal Graph Attention Network) ───────")
    if not Path(npz_path).exists():
        print(f"[SKIP] Graph sequence file not found: {npz_path}")
        print("       Run pipeline features stage first to generate it.")
        return None

    data  = np.load(npz_path, allow_pickle=True)
    X, A, valid, y = data["X"], data["A"], data["valid"], data["y"]
    names = list(data["names"])

    # Apply video-level splits
    if Path(splits_csv).exists():
        splits_df = pd.read_csv(splits_csv).set_index("video_name")["split"].to_dict()
        def _get_split(name):
            video = name.split("_w")[0]
            return splits_df.get(video, "train")
        split_arr = np.array([_get_split(n) for n in names])
    else:
        split_arr = np.array(["train"] * len(names))

    tr = split_arr == "train"
    va = split_arr == "val"
    te = split_arr == "test"

    print(f"Graph split: train={tr.sum()}  val={va.sum()}  test={te.sum()}")

    if va.sum() == 0:
        n_va = max(1, tr.sum() // 5)
        va_idx = np.where(tr)[0][:n_va]
        va = np.zeros(len(split_arr), dtype=bool)
        va[va_idx] = True
        tr = ~va & (split_arr == "train")
        print(f"  [WARN] No val graph sequences — using first {n_va} train samples")

    save_dir.mkdir(parents=True, exist_ok=True)
    model, _ = train_stgat(
        X_train     = X[tr],     A_train     = A[tr],
        valid_train = valid[tr], y_train     = y[tr],
        X_val       = X[va],     A_val       = A[va],
        valid_val   = valid[va], y_val       = y[va],
        n_classes   = len(le.classes_),
        epochs      = STGAT_EPOCHS,
        lr          = STGAT_LR,
        batch_size  = STGAT_BATCH,
        save_path   = str(save_dir / "stgat_model.pt"),
    )

    if te.sum() > 0:
        from src.models.stgat_model import predict_stgat
        te_preds, _, _ = predict_stgat(X[te], A[te], valid[te],
                                        str(save_dir / "stgat_model.pt"))
        te_f1 = f1_score(y[te], te_preds, average="weighted", zero_division=0)
        print(f"Test F1 (weighted):  {te_f1:.3f}")
        print(classification_report(y[te], te_preds,
                                    target_names=le.classes_, zero_division=0))

    return model


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgb(df_train, df_val, df_test, le, features, save_dir: Path):
    try:
        import xgboost as xgb
    except ImportError:
        print("[SKIP] xgboost not installed. Run: pip install xgboost")
        return None

    print("\n── XGBoost ───────────────────────────────────────────────")
    X_tr, y_tr = _get_Xy(df_train, features)
    X_va, y_va = _get_Xy(df_val,   features)
    X_te, y_te = _get_Xy(df_test,  features)

    n_classes = len(le.classes_)
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

    model = xgb.XGBClassifier(
        n_estimators    = XGB_N_ESTIMATORS,
        max_depth       = XGB_MAX_DEPTH,
        learning_rate   = XGB_LR,
        objective       = objective,
        num_class       = n_classes if n_classes > 2 else None,
        n_jobs          = N_JOBS,
        random_state    = RANDOM_SEED,
        use_label_encoder = False,
        eval_metric     = "mlogloss" if n_classes > 2 else "logloss",
        verbosity       = 0,
    )

    eval_set = [(X_va, y_va)] if len(X_va) else None
    model.fit(
        X_tr, y_tr,
        eval_set        = eval_set,
        verbose         = False,
    )

    if len(X_va):
        va_preds = model.predict(X_va)
        va_f1    = f1_score(y_va, va_preds, average="weighted", zero_division=0)
        print(f"Val  F1 (weighted):  {va_f1:.3f}")
        print(classification_report(y_va, va_preds, target_names=le.classes_, zero_division=0))

    if len(X_te):
        te_preds = model.predict(X_te)
        te_f1    = f1_score(y_te, te_preds, average="weighted", zero_division=0)
        print(f"Test F1 (weighted):  {te_f1:.3f}")
        print(classification_report(y_te, te_preds, target_names=le.classes_, zero_division=0))

    imp_df = pd.DataFrame({
        "feature":   features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances (top 10):")
    print(imp_df.head(10).to_string(index=False))

    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / "xgb_model.pkl")
    print(f"Saved → {save_dir}/xgb_model.pkl")
    return model


# ── BiLSTM ────────────────────────────────────────────────────────────────────

def train_bilstm_model(npz_path: str, splits_csv: str, le, save_dir: Path):
    from src.models.bilstm_model import train_bilstm

    print("\n── BiLSTM ────────────────────────────────────────────────")
    if not Path(npz_path).exists():
        print(f"[SKIP] Sequence file not found: {npz_path}")
        print("       Run pipeline with --mode sequence to generate it first.")
        return None

    data = np.load(npz_path, allow_pickle=True)
    X, y, names = data["X"], data["y"], list(data["names"])

    # Apply video-level splits
    if Path(splits_csv).exists():
        splits_df = pd.read_csv(splits_csv).set_index("video_name")["split"].to_dict()
        def get_split(name):
            video = name.split("_w")[0]
            return splits_df.get(video, "train")
        split_labels = [get_split(n) for n in names]
    else:
        split_labels = ["train"] * len(names)

    split_arr = np.array(split_labels)
    X_tr = X[split_arr == "train"]
    y_tr = y[split_arr == "train"]
    X_va = X[split_arr == "val"]
    y_va = y[split_arr == "val"]
    X_te = X[split_arr == "test"]
    y_te = y[split_arr == "test"]

    print(f"Sequence split: train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    if len(X_va) == 0:
        X_va, y_va = X_tr[:max(1, len(X_tr)//5)], y_tr[:max(1, len(X_tr)//5)]
        print("  [WARN] No val sequences from split — using 20% of train for monitoring")

    save_dir.mkdir(parents=True, exist_ok=True)
    model, history = train_bilstm(
        X_train   = X_tr,
        y_train   = y_tr,
        X_val     = X_va,
        y_val     = y_va,
        n_classes = len(le.classes_),
        epochs    = BILSTM_EPOCHS,
        lr        = BILSTM_LR,
        batch_size= BILSTM_BATCH,
        save_path = str(save_dir / "bilstm_model.pt"),
    )

    if len(X_te):
        from src.models.bilstm_model import predict_bilstm
        te_preds, _ = predict_bilstm(X_te, str(save_dir / "bilstm_model.pt"))
        te_f1 = f1_score(y_te, te_preds, average="weighted", zero_division=0)
        print(f"Test F1 (weighted):  {te_f1:.3f}")
        print(classification_report(y_te, te_preds, target_names=le.classes_, zero_division=0))

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train threat detection models")
    parser.add_argument("--input",        default=str(THREAT_WINDOWS_CSV))
    parser.add_argument("--seq-input",    default=str(SEQUENCE_WINDOWS_NPZ))
    parser.add_argument("--graph-input",  default=str(GRAPH_SEQ_NPZ))
    parser.add_argument("--splits",       default=str(SPLITS_CSV))
    parser.add_argument("--model",        default="all",
                        choices=["xgb", "bilstm", "stgat", "all"])
    parser.add_argument("--out-dir",      default=str(MODELS_DIR))
    args = parser.parse_args()

    save_dir = Path(args.out_dir)
    features = WINDOW_FEATURES

    # Load window data (needed by XGB)
    df_train, df_val, df_test, le = load_window_data(args.input, args.splits)
    joblib.dump(le, LABEL_ENCODER_PATH)

    if args.model in ("xgb", "all"):
        train_xgb(df_train, df_val, df_test, le, features, save_dir)

    if args.model in ("bilstm", "all"):
        train_bilstm_model(args.seq_input, args.splits, le, save_dir)

    if args.model in ("stgat", "all"):
        train_stgat_model(args.graph_input, args.splits, le, save_dir)


if __name__ == "__main__":
    main()

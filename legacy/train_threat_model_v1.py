"""
train_threat_model.py
======================
Trains a supervised classifier to detect threat scenarios from group-level
trajectory features extracted by src/threat_feature_extractor.py.

Supports:
  - Binary classification: 0=normal, 1=threat
  - Multi-class: 0=normal, 1=following, 2=surrounding, 3=fast_approach

Input:  data/derived/threat_windows.csv   (labeled rows)
Output: data/derived/threat_model.pkl     (trained model)

Usage:
    python train_threat_model.py                    # binary mode
    python train_threat_model.py --mode multiclass  # multi-class mode
    python train_threat_model.py --model gbm        # use GradientBoosting
"""

import os
import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

INPUT_CSV   = "data/derived/threat_windows.csv"
OUTPUT_PKG  = "data/derived/threat_model.pkl"
LABEL_ENC_PATH = "data/derived/threat_label_encoder.pkl"

FEATURES = [
    # general proximity
    "threat_person_count",
    "min_distance",
    # closing / approach
    "num_closing_persons",
    "max_closing_speed",
    "avg_closing_rate",
    "max_velocity_toward_target",
    # surrounding-specific
    "angle_spread",
    "converging_sector_count",
    # following-specific
    "behind_person_count",
    "avg_distance_consistency",
    # group motion
    "avg_speed_of_group",
    "target_speed",
]

THREAT_TYPE_MAP = {
    "normal":        0,
    "following":     1,
    "surrounding":   2,
    "fast_approach": 3,
}


def load_binary(df: pd.DataFrame):
    """Binary: threat_label column must be 0 or 1."""
    df = df.dropna(subset=["threat_label"]).copy()
    df = df[df["threat_label"].astype(str).str.strip() != ""].copy()
    df["threat_label"] = df["threat_label"].astype(int)
    return df, df["threat_label"]


def load_multiclass(df: pd.DataFrame):
    """Multi-class: uses threat_type column string labels."""
    df = df.dropna(subset=["threat_type"]).copy()
    df = df[df["threat_type"].astype(str).str.strip() != ""].copy()
    le = LabelEncoder()
    y = le.fit_transform(df["threat_type"].str.strip())
    return df, pd.Series(y), le


def build_model(model_name: str, n_classes: int):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "gbm":
        if n_classes > 2:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                random_state=42,
            )
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=42,
        )
    elif model_name == "svm":
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose rf / gbm / svm")


def print_feature_importances(model, feature_names: list):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nFeature Importances:")
        print(imp.to_string(index=False))
    else:
        print("\n(Feature importances not available for this model type)")


def main():
    parser = argparse.ArgumentParser(description="Train threat detection model")
    parser.add_argument("--input",   default=INPUT_CSV,  help="Labeled feature CSV")
    parser.add_argument("--output",  default=OUTPUT_PKG, help="Output model .pkl path")
    parser.add_argument("--mode",    default="binary",
                        choices=["binary", "multiclass"],
                        help="Classification mode")
    parser.add_argument("--model",   default="rf",
                        choices=["rf", "gbm", "svm"],
                        help="Classifier type")
    parser.add_argument("--cv",      type=int, default=5,
                        help="Number of cross-validation folds (video-grouped)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    label_encoder = None
    if args.mode == "binary":
        df, y = load_binary(df)
        class_names = ["normal", "threat"]
        print(f"\nClass distribution:\n{y.value_counts().to_string()}")
    else:
        df, y, label_encoder = load_multiclass(df)
        class_names = list(label_encoder.classes_)
        print(f"\nClass distribution:\n{pd.Series(y).value_counts().to_string()}")

    # Drop rows with any missing feature
    df = df.dropna(subset=FEATURES).copy()
    y  = y.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

    X = df[FEATURES]
    groups = df["video_name"]  # used for group-based CV split

    n_classes = len(class_names)
    model = build_model(args.model, n_classes)

    # ── Cross-validation (video-grouped so same video never leaks across folds) ──
    print(f"\nRunning {args.cv}-fold video-grouped cross-validation...")
    cv = StratifiedGroupKFold(n_splits=min(args.cv, groups.nunique()))
    cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_weighted")
    print(f"CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Final fit on all data ──
    model.fit(X, y)

    preds = model.predict(X)
    print("\nTraining set performance (informational only):")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, target_names=class_names))

    print_feature_importances(model, FEATURES)

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(model, args.output)
    print(f"\nSaved model → {args.output}")

    if label_encoder is not None:
        joblib.dump(label_encoder, LABEL_ENC_PATH)
        print(f"Saved label encoder → {LABEL_ENC_PATH}")


if __name__ == "__main__":
    main()

"""
pre_train_eda.py
=================
Exploratory Data Analysis BEFORE training.
Outputs plots and statistics to data/outputs/eda/.

Analyses:
  - Class distribution bar chart
  - Feature distributions per class (box plots)
  - Correlation heatmap
  - PCA / UMAP 2-D scatter coloured by class
  - Missing value heatmap
  - Video-level sample count per class

Usage:
    python src/evaluation/pre_train_eda.py
    python src/evaluation/pre_train_eda.py --input data/derived/threat_windows.csv
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import THREAT_WINDOWS_CSV, EDA_OUT_DIR, WINDOW_FEATURES, CLASS_NAMES

warnings.filterwarnings("ignore")
OUT = Path(EDA_OUT_DIR) / "pre_train"


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def class_distribution(df: pd.DataFrame):
    counts = df["threat_type"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("Set2", len(counts)))
    ax.bar_label(bars)
    ax.set_title("Class Distribution (labeled windows)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Threat Type")
    _save(fig, "01_class_distribution")


def feature_boxplots(df: pd.DataFrame, features: list[str]):
    feats = [f for f in features if f in df.columns]
    n = len(feats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(CLASS_NAMES))
    for i, feat in enumerate(feats):
        ax = axes[i]
        sns.boxplot(data=df, x="threat_type", y=feat, order=CLASS_NAMES,
                    palette=palette, ax=ax, flierprops={"marker": ".", "alpha": 0.4})
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions per Class", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "02_feature_boxplots")


def correlation_heatmap(df: pd.DataFrame, features: list[str]):
    feats = [f for f in features if f in df.columns]
    corr  = df[feats].corr()
    mask  = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "03_correlation_heatmap")


def pca_scatter(df: pd.DataFrame, features: list[str]):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    feats = [f for f in features if f in df.columns]
    X = df[feats].fillna(0.0).values
    y = df["threat_type"].values

    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2, random_state=42)
    Z    = pca.fit_transform(X_sc)
    var  = pca.explained_variance_ratio_

    palette = sns.color_palette("Set2", len(CLASS_NAMES))
    color_map = {cls: palette[i] for i, cls in enumerate(CLASS_NAMES)}
    colors = [color_map.get(c, "gray") for c in y]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in CLASS_NAMES:
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], c=[color_map[cls]], label=cls,
                   alpha=0.6, s=30, edgecolors="none")
    ax.set_title(f"PCA 2-D Projection  (var explained: "
                 f"PC1={var[0]:.1%}, PC2={var[1]:.1%})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    _save(fig, "04_pca_scatter")

    # Attempt UMAP if installed
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        Z_umap  = reducer.fit_transform(X_sc)
        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in CLASS_NAMES:
            mask = y == cls
            ax.scatter(Z_umap[mask, 0], Z_umap[mask, 1], c=[color_map[cls]],
                       label=cls, alpha=0.6, s=30, edgecolors="none")
        ax.set_title("UMAP 2-D Projection")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()
        _save(fig, "04b_umap_scatter")
    except ImportError:
        pass   # umap-learn optional


def missing_heatmap(df: pd.DataFrame, features: list[str]):
    feats = [f for f in features if f in df.columns]
    miss  = df[feats].isna()
    if miss.values.sum() == 0:
        print("  No missing values — skipping missing heatmap")
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(miss.T, cbar=True, ax=ax, cmap="Reds", yticklabels=True)
    ax.set_title("Missing Values (red = missing)")
    ax.set_xlabel("Row index")
    _save(fig, "05_missing_values")


def video_sample_counts(df: pd.DataFrame):
    vc = df.groupby(["video_name", "threat_type"]).size().reset_index(name="n_windows")
    fig, ax = plt.subplots(figsize=(max(8, len(vc) * 0.4), 5))
    palette = sns.color_palette("Set2", len(CLASS_NAMES))
    type_colors = {cls: palette[i] for i, cls in enumerate(CLASS_NAMES)}
    bars = ax.bar(range(len(vc)), vc["n_windows"],
                  color=[type_colors.get(t, "gray") for t in vc["threat_type"]])
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc["video_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# Windows")
    ax.set_title("Windows per Video (coloured by threat type)")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=type_colors[c], label=c) for c in CLASS_NAMES]
    ax.legend(handles=legend_elements)
    _save(fig, "06_video_sample_counts")


def run_eda(csv_path: str):
    print(f"\nPre-Training EDA → {OUT}")
    df = pd.read_csv(csv_path)
    df = df[df["threat_type"].notna() & (df["threat_type"].str.strip() != "")].copy()
    df["threat_type"] = df["threat_type"].str.strip()

    print(f"  Loaded {len(df)} labeled rows | {df['threat_type'].nunique()} classes")
    feats = [f for f in WINDOW_FEATURES if f in df.columns]

    class_distribution(df)
    feature_boxplots(df, feats)
    correlation_heatmap(df, feats)
    pca_scatter(df, feats)
    missing_heatmap(df, feats)
    video_sample_counts(df)
    print("Pre-Training EDA complete.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(THREAT_WINDOWS_CSV))
    args = parser.parse_args()
    run_eda(args.input)


if __name__ == "__main__":
    main()

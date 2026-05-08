"""
eda_following.py

Exploratory Data Analysis for the person-following research project.
Run this after mot17_pair_extractor.py and after labeling (labeled_pairs.csv).

Produces plots in data/derived/eda/
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

EDA_DIR = "data/derived/eda"
os.makedirs(EDA_DIR, exist_ok=True)

PAIR_WINDOWS_CSV = "data/derived/pair_windows_for_labeling.csv"
LABELED_CSV = "data/derived/labeled_pairs.csv"
TRACKED_CSV = "data/outputs/csv/tracked_output_01.csv"

FEATURES = [
    "num_points",
    "direction_similarity",
    "avg_distance",
    "distance_variance",
    "speed_similarity",
    "behind_ratio",
]

FEATURE_LABELS = {
    "num_points": "Num Points",
    "direction_similarity": "Direction Similarity",
    "avg_distance": "Avg Distance (px)",
    "distance_variance": "Distance Variance",
    "speed_similarity": "Speed Similarity",
    "behind_ratio": "Behind Ratio",
}


# ─────────────────────────────────────────
# 1. Track trajectory EDA (from track_video output)
# ─────────────────────────────────────────

def eda_tracks(csv_path):
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    track_ids = df["track_id"].unique()
    print(f"Unique tracks: {len(track_ids)}, Total detections: {len(df)}")

    # 1a. Track length distribution
    lengths = df.groupby("track_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Track Lengths")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "track_length_distribution.png"), dpi=120)
    plt.close(fig)
    print(f"  Track lengths: min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}")

    # 1b. Trajectory map — all tracks overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = cm.get_cmap("tab20", len(track_ids))
    for i, tid in enumerate(sorted(track_ids)):
        sub = df[df["track_id"] == tid].sort_values("frame_idx")
        ax.plot(sub["cx"], sub["cy"], color=cmap(i), linewidth=0.8, alpha=0.7, label=f"ID {tid}")
        ax.scatter(sub["cx"].iloc[0], sub["cy"].iloc[0], color=cmap(i), s=20, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel("cx (pixels)")
    ax.set_ylabel("cy (pixels)")
    ax.set_title("Trajectory Map — All Tracks")
    if len(track_ids) <= 15:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "trajectory_map.png"), dpi=120)
    plt.close(fig)

    # 1c. Detection confidence distribution
    if "confidence" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["confidence"].dropna(), bins=40, color="darkorange", edgecolor="white")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Detection Confidence Distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "confidence_distribution.png"), dpi=120)
        plt.close(fig)

    # 1d. Speed per track (frame-to-frame displacement)
    speeds = []
    for tid in track_ids:
        sub = df[df["track_id"] == tid].sort_values("frame_idx")
        dx = sub["cx"].diff().dropna()
        dy = sub["cy"].diff().dropna()
        spd = np.sqrt(dx**2 + dy**2).mean()
        speeds.append(spd)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(speeds, bins=30, color="mediumseagreen", edgecolor="white")
    ax.set_xlabel("Mean speed (px/frame)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Track Mean Speed Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "speed_distribution.png"), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────
# 2. Pair window EDA (MOT17 unlabeled pairs)
# ─────────────────────────────────────────

def eda_pair_windows(csv_path):
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found — run mot17_pair_extractor.py first")
        return

    df = pd.read_csv(csv_path)
    print(f"\nPair windows: {len(df)}")

    # 2a. Feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURES):
        if feat in df.columns:
            axes[i].hist(df[feat].dropna(), bins=40, color="steelblue", edgecolor="white")
            axes[i].set_title(FEATURE_LABELS[feat])
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel("Count")
    fig.suptitle("Feature Distributions (Unlabeled Pair Windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "feature_distributions_unlabeled.png"), dpi=120)
    plt.close(fig)

    # 2b. Sequences — number of pair windows per sequence
    if "sequence" in df.columns:
        seq_counts = df["sequence"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        seq_counts.plot(kind="bar", ax=ax, color="slateblue", edgecolor="white")
        ax.set_xlabel("Sequence")
        ax.set_ylabel("Pair windows")
        ax.set_title("Pair Windows per MOT17 Sequence")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "pair_windows_per_sequence.png"), dpi=120)
        plt.close(fig)

    # 2c. Scatter: direction_similarity vs behind_ratio
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["direction_similarity"], df["behind_ratio"],
               alpha=0.2, s=8, color="teal")
    ax.set_xlabel("Direction Similarity")
    ax.set_ylabel("Behind Ratio")
    ax.set_title("Direction Similarity vs Behind Ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "scatter_dirsim_behindratio.png"), dpi=120)
    plt.close(fig)

    # 2d. Scatter: avg_distance vs speed_similarity
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["avg_distance"], df["speed_similarity"],
               alpha=0.2, s=8, color="coral")
    ax.set_xlabel("Avg Distance (px)")
    ax.set_ylabel("Speed Similarity")
    ax.set_title("Avg Distance vs Speed Similarity")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "scatter_dist_speedsim.png"), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────
# 3. Labeled pair EDA (after manual labeling)
# ─────────────────────────────────────────

def eda_labeled(csv_path):
    if not os.path.exists(csv_path):
        print(f"\n[skip] {csv_path} not found — label pair_windows_for_labeling.csv first")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["following_label"]).copy()
    df["following_label"] = df["following_label"].astype(int)
    print(f"\nLabeled pairs: {len(df)}, following=1: {df['following_label'].sum()}")

    # 3a. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["following_label"].value_counts().sort_index()
    ax.bar(["Not Following (0)", "Following (1)"], counts.values,
           color=["tomato", "mediumseagreen"], edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_title("Class Balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "class_balance.png"), dpi=120)
    plt.close(fig)

    # 3b. Feature distributions per class (box plots)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURES):
        if feat not in df.columns:
            continue
        groups = [
            df[df["following_label"] == 0][feat].dropna().values,
            df[df["following_label"] == 1][feat].dropna().values,
        ]
        bp = axes[i].boxplot(groups, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("tomato")
        bp["boxes"][1].set_facecolor("mediumseagreen")
        axes[i].set_xticklabels(["Not Following", "Following"])
        axes[i].set_title(FEATURE_LABELS[feat])
        axes[i].set_ylabel(feat)
    fig.suptitle("Feature Distributions by Class")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "feature_by_class_boxplots.png"), dpi=120)
    plt.close(fig)

    # 3c. Correlation heatmap
    feat_df = df[FEATURES].dropna()
    corr = feat_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURES], fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)

    # 3d. PCA scatter (2D) colored by label
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        X = feat_df.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        labels = df.loc[feat_df.index, "following_label"].values

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["tomato", "mediumseagreen"]
        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[cls], label=["Not Following", "Following"][cls],
                       alpha=0.5, s=15)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("PCA of Feature Space")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "pca_scatter.png"), dpi=120)
        plt.close(fig)
        print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    except ImportError:
        print("  [skip PCA] sklearn not available")

    # 3e. Distance over time for sample pairs
    _plot_distance_over_time(df)


def _plot_distance_over_time(df):
    """Plot avg_distance and behind_ratio trend for a few sample pairs per class."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ["Not Following (label=0)", "Following (label=1)"]

    for cls_idx, cls in enumerate([0, 1]):
        ax = axes[cls_idx]
        sample = df[df["following_label"] == cls].head(5)
        for _, row in sample.iterrows():
            label = f"L{int(row['leader_id'])}-F{int(row['follower_id'])}"
            ax.scatter([row["avg_distance"]], [row["behind_ratio"]],
                       s=80, label=label)
        ax.set_xlabel("Avg Distance (px)")
        ax.set_ylabel("Behind Ratio")
        ax.set_title(titles[cls_idx])
        ax.set_xlim(0, 260)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle("Sample Pairs: Avg Distance vs Behind Ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "sample_pairs_dist_behind.png"), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────
# 4. Model feature importance (post-training)
# ─────────────────────────────────────────

def eda_model():
    model_path = "data/derived/following_rf.pkl"
    if not os.path.exists(model_path):
        print("\n[skip] Model not found — run train_following_baseline.py first")
        return

    import joblib
    model = joblib.load(model_path)
    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = np.argsort(importances)[::-1]
    ax.bar(range(len(FEATURES)),
           [importances[i] for i in sorted_idx],
           color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS[FEATURES[i]] for i in sorted_idx], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importances")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "feature_importances.png"), dpi=120)
    plt.close(fig)
    print("\nFeature importances saved.")


def main():
    print("=== EDA: Track Video Output ===")
    eda_tracks(TRACKED_CSV)

    print("\n=== EDA: Pair Windows (Unlabeled) ===")
    eda_pair_windows(PAIR_WINDOWS_CSV)

    print("\n=== EDA: Labeled Pairs ===")
    eda_labeled(LABELED_CSV)

    print("\n=== EDA: Model Feature Importances ===")
    eda_model()

    print(f"\nAll plots saved to {EDA_DIR}/")


if __name__ == "__main__":
    main()

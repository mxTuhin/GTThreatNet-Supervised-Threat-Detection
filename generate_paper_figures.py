"""
Generate combined paper figures from individual EDA and XAI images.

Grouping logic:
- Only combine images that are thematically related.
- Pre-train EDA: class structure, feature analysis, dataset distribution.
- Post-train EDA: per-model confidence, per-video accuracy, prediction timelines, error analysis.
- XAI/SHAP: SHAP waterfalls (4-class), SHAP summary, BiLSTM attention, STGAT attention.
- Model Evaluation: confusion matrices, ROC curves, PR curves (one figure each, all 3 models).

Run:
    source venv/bin/activate
    python3 generate_paper_figures.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

BASE = "data/outputs/eda"
PRE = f"{BASE}/pre_train"
POST = f"{BASE}/post_train"
XAI = f"{BASE}/xai"
OUT = f"{BASE}/paper_figures"
os.makedirs(OUT, exist_ok=True)


def load(path):
    return Image.open(path)


def add_label(ax, label, fontsize=14):
    ax.set_title(label, fontsize=fontsize, fontweight="bold", loc="left", pad=6)


def save(fig, name, dpi=180):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PRE-TRAIN EDA
# ---------------------------------------------------------------------------

# Figure PRE-1: Correlation Heatmap + PCA Scatter
# Rationale: both reveal feature-space structure and class separability.
print("Generating pre-train figures...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].imshow(load(f"{PRE}/03_correlation_heatmap.png"))
axes[0].axis("off")
add_label(axes[0], "(a) Feature Correlation Heatmap")

axes[1].imshow(load(f"{PRE}/04_pca_scatter.png"))
axes[1].axis("off")
add_label(axes[1], "(b) PCA 2-D Projection")

fig.suptitle(
    "Pre-Train EDA — Feature Space Structure",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_pre1_corr_pca.png")

# Figure PRE-2: Class Distribution (standalone — compact & critical)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(load(f"{PRE}/01_class_distribution.png"))
ax.axis("off")
fig.suptitle(
    "Pre-Train EDA — Class Distribution (Labeled Windows)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
save(fig, "fig_pre2_class_distribution.png")

# Figure PRE-3: Feature Boxplots (standalone — too dense to pair)
fig, ax = plt.subplots(figsize=(18, 14))
ax.imshow(load(f"{PRE}/02_feature_boxplots.png"))
ax.axis("off")
fig.suptitle(
    "Pre-Train EDA — Feature Distributions per Class",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
save(fig, "fig_pre3_feature_boxplots.png")

# Figure PRE-4: Video Sample Counts (standalone — aspect ratio too wide to pair)
fig, ax = plt.subplots(figsize=(22, 5))
ax.imshow(load(f"{PRE}/06_video_sample_counts.png"))
ax.axis("off")
fig.suptitle(
    "Pre-Train EDA — Windows per Video (coloured by Threat Type)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
save(fig, "fig_pre4_video_sample_counts.png")


# ---------------------------------------------------------------------------
# POST-TRAIN EDA
# ---------------------------------------------------------------------------
print("Generating post-train figures...")

# Figure POST-1: Prediction Confidence Distribution — all 3 models
# Rationale: identical metric, 3 models — natural side-by-side comparison.
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, (model, img) in zip(axes, [
    ("BiLSTM",   f"{POST}/bilstm_confidence_dist.png"),
    ("STGAT",    f"{POST}/stgat_confidence_dist.png"),
    ("XGBoost",  f"{POST}/xgb_confidence_dist.png"),
]):
    ax.imshow(load(img))
    ax.axis("off")
    add_label(ax, f"({chr(97 + list(axes).index(ax))}) {model}")

fig.suptitle(
    "Post-Train EDA — Prediction Confidence Distribution per Class",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_post1_confidence_dist_3models.png")

# Figure POST-2: Per-Video Accuracy — all 3 models
# Rationale: identical metric, 3 models — direct cross-model comparison.
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax, (model, img) in zip(axes, [
    ("BiLSTM",  f"{POST}/bilstm_per_video_accuracy.png"),
    ("STGAT",   f"{POST}/stgat_per_video_accuracy.png"),
    ("XGBoost", f"{POST}/xgb_per_video_accuracy.png"),
]):
    ax.imshow(load(img))
    ax.axis("off")
    add_label(ax, f"({chr(97 + list(axes).index(ax))}) {model}")

fig.suptitle(
    "Post-Train EDA — Per-Video Accuracy (70 % threshold dashed)",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_post2_per_video_acc_3models.png")

# Figure POST-3: Prediction Timelines — STGAT + XGBoost
# Rationale: same visualization type; BiLSTM timeline is too compressed to be legible.
fig, axes = plt.subplots(1, 2, figsize=(26, 14))
axes[0].imshow(load(f"{POST}/stgat_prediction_timeline.png"))
axes[0].axis("off")
add_label(axes[0], "(a) STGAT")

axes[1].imshow(load(f"{POST}/xgb_prediction_timeline.png"))
axes[1].axis("off")
add_label(axes[1], "(b) XGBoost")

fig.suptitle(
    "Post-Train EDA — Prediction Timeline per Video (ground truth = dashed)",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_post3_timeline_stgat_xgb.png")

# Figure POST-4: XGBoost Error Analysis (standalone — already multi-panel)
fig, ax = plt.subplots(figsize=(22, 10))
ax.imshow(load(f"{POST}/xgb_error_analysis.png"))
ax.axis("off")
fig.suptitle(
    "Post-Train EDA — XGBoost: Feature Distributions (Correct vs Incorrect Predictions)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
save(fig, "fig_post4_xgb_error_analysis.png")


# ---------------------------------------------------------------------------
# XAI / SHAP
# ---------------------------------------------------------------------------
print("Generating XAI/SHAP figures...")

# Figure XAI-1: SHAP Waterfalls — 4 classes in 2×2 grid
# Rationale: one waterfall per class, natural grouping of 4.
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
pairs = [
    ("(a) Normal",        f"{XAI}/xgboost_shap_waterfall_normal.png"),
    ("(b) Following",     f"{XAI}/xgboost_shap_waterfall_following.png"),
    ("(c) Surrounding",   f"{XAI}/xgboost_shap_waterfall_surrounding.png"),
    ("(d) Fast Approach", f"{XAI}/xgboost_shap_waterfall_fast_approach.png"),
]
for ax, (label, img) in zip(axes.flat, pairs):
    ax.imshow(load(img))
    ax.axis("off")
    add_label(ax, label)

fig.suptitle(
    "XGBoost SHAP Waterfall Plots — Per-Class Feature Attribution",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_xai1_shap_waterfalls_4class.png")

# Figure XAI-2: SHAP Summary / Interaction Plot (standalone)
fig, ax = plt.subplots(figsize=(10, 14))
ax.imshow(load(f"{XAI}/xgboost_shap_summary.png"))
ax.axis("off")
fig.suptitle(
    "XGBoost SHAP Interaction Summary — Top Feature Pairs",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
save(fig, "fig_xai2_shap_summary.png")

# Figure XAI-3: BiLSTM Attention — heatmap + mean temporal curve
# Rationale: both visualise BiLSTM attention weights over the 30-frame window.
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].imshow(load(f"{XAI}/bilstm_attention_heatmaps.png"))
axes[0].axis("off")
add_label(axes[0], "(a) Attention Heatmap (Normal samples)")

axes[1].imshow(load(f"{XAI}/bilstm_attention_per_class.png"))
axes[1].axis("off")
add_label(axes[1], "(b) Mean Temporal Attention (Normal)")

fig.suptitle(
    "BiLSTM XAI — Attention Weight Analysis (Normal class only)",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_xai3_bilstm_attention.png")

# Figure XAI-4: STGAT Attention — timeline + per-person importance
# Rationale: both visualise STGAT graph-attention weights over persons and frames.
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].imshow(load(f"{XAI}/stgat_attention_timeline.png"))
axes[0].axis("off")
add_label(axes[0], "(a) Attention to Nearby Persons Over Time")

axes[1].imshow(load(f"{XAI}/stgat_person_importance.png"))
axes[1].axis("off")
add_label(axes[1], "(b) Mean Attention per Graph Node")

fig.suptitle(
    "STGAT XAI — Graph Attention Analysis (Normal class only)",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_xai4_stgat_attention.png")

# ---------------------------------------------------------------------------
# MODEL EVALUATION  (separate category — not EDA)
# ---------------------------------------------------------------------------
print("Generating model evaluation figures...")

EVAL = f"{BASE}/evaluation"

# Figure EVAL-1: Confusion Matrices — all 3 models stacked vertically
# Each source image already contains raw + normalised side by side.
# Stack 3 rows (one per model) so each panel stays readable.
fig, axes = plt.subplots(3, 1, figsize=(20, 24))
for ax, (model, img) in zip(axes, [
    ("BiLSTM",   f"{EVAL}/bilstm_confusion.png"),
    ("STGAT",    f"{EVAL}/stgat_confusion.png"),
    ("XGBoost",  f"{EVAL}/xgboost_confusion.png"),
]):
    ax.imshow(load(img))
    ax.axis("off")
    add_label(ax, f"({chr(97 + list(axes).index(ax))}) {model}")

fig.suptitle(
    "Model Evaluation — Confusion Matrices (Raw & Normalised)",
    fontsize=16, fontweight="bold", y=1.005
)
plt.tight_layout()
save(fig, "fig_eval1_confusion_matrices.png")

# Figure EVAL-2: ROC Curves — all 3 models side by side
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, (model, img) in zip(axes, [
    ("BiLSTM",   f"{EVAL}/bilstm_roc.png"),
    ("STGAT",    f"{EVAL}/stgat_roc.png"),
    ("XGBoost",  f"{EVAL}/xgboost_roc.png"),
]):
    ax.imshow(load(img))
    ax.axis("off")
    add_label(ax, f"({chr(97 + list(axes).index(ax))}) {model}")

fig.suptitle(
    "Model Evaluation — ROC Curves per Class",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_eval2_roc_curves.png")

# Figure EVAL-3: Precision-Recall Curves — all 3 models side by side
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, (model, img) in zip(axes, [
    ("BiLSTM",   f"{EVAL}/bilstm_pr.png"),
    ("STGAT",    f"{EVAL}/stgat_pr.png"),
    ("XGBoost",  f"{EVAL}/xgboost_pr.png"),
]):
    ax.imshow(load(img))
    ax.axis("off")
    add_label(ax, f"({chr(97 + list(axes).index(ax))}) {model}")

fig.suptitle(
    "Model Evaluation — Precision-Recall Curves per Class",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
save(fig, "fig_eval3_pr_curves.png")

print("\nAll paper figures written to:", OUT)
print("Run:  source venv/bin/activate && python3 generate_paper_figures.py")

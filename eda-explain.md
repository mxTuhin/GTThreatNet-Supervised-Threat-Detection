# EDA Figures — Research Paper Guide

> **Scope:** This document covers only the EDA (Exploratory Data Analysis) pipeline —
> pre-train feature analysis, post-train behavioural analysis, and XAI/SHAP interpretability.
> Model training curves, hyperparameter tuning, and evaluation metrics (ROC, PR, confusion
> matrix) are intentionally excluded here.

---

## How to Generate Combined Figures

```bash
source venv/bin/activate
python3 generate_paper_figures.py
# Output → data/outputs/eda/paper_figures/
```

---

## 1. Pre-Train EDA

These figures describe the **raw dataset** before any model sees it.
They belong in the **Dataset** or **Methodology** section of the paper.

---

### Figure PRE-1 — Feature Space Structure
**File:** `fig_pre1_corr_pca.png`
**Combines:** `03_correlation_heatmap.png` + `04_pca_scatter.png`
**Why paired:** Both answer the same question — *how are features structured relative to each other and to class boundaries?*

#### (a) Feature Correlation Heatmap
- **What it shows:** Pearson correlation between all 20 engineered features.
- **Key findings:**
  - `max_closing_speed` ↔ `avg_closing_rate` are highly correlated (expected — both measure closing velocity).
  - `encirclement_ratio` ↔ `converging_sector_count` cluster together — they capture the same surround geometry from different angles.
  - `threat_person_count` ↔ `num_closing_persons` are moderately correlated but carry distinct information (count vs. speed-filtered count).
- **Paper use:** Justify feature selection / dimensionality reduction decisions. Cite collinear pairs as candidates for pruning if needed.

#### (b) PCA 2-D Projection
- **What it shows:** All labeled windows projected onto the first two principal components (PC1 = 28.2 %, PC2 = 17.9 % — total 46.1 % variance explained).
- **Key findings:**
  - `normal` samples spread widely across the full PC space, including many outliers in the high-PC1 region — indicating high intra-class variance.
  - `following`, `surrounding`, and `fast_approach` samples cluster tightly near the origin, overlapping each other.
  - The overlap between threat classes is a direct visual argument for why this is a **hard multi-class problem**, not a simple binary threat/no-threat task.
- **Paper use:** Motivate the use of more expressive sequence models (BiLSTM, STGAT) over simple distance-based rules.

---

### Figure PRE-2 — Class Distribution
**File:** `fig_pre2_class_distribution.png`
**Source:** `01_class_distribution.png` (standalone — compact and self-contained)

- **What it shows:** Count of labeled sliding-window samples per threat class.
  | Class | Count |
  |---|---|
  | normal | 6 425 |
  | surrounding | 3 172 |
  | following | 1 398 |
  | fast_approach | 1 032 |

- **Key findings:**
  - Severe class imbalance: `normal` is ~6× more frequent than `fast_approach`.
  - `following` and `fast_approach` are underrepresented — these are exactly the classes with the highest real-world safety relevance.
- **Paper use:** Mandatory figure in any dataset description. Justifies the use of class-weighted loss functions or over-sampling strategies. If you use SMOTE or class weights in training, reference this figure.

---

### Figure PRE-3 — Feature Distributions per Class
**File:** `fig_pre3_feature_boxplots.png`
**Source:** `02_feature_boxplots.png` (standalone — 20 subplots, too dense to pair)

- **What it shows:** Box-and-whisker plots for all 20 features, broken down by class.
- **Key findings:**
  - `threat_person_count`: dramatically higher median for `surrounding` than other classes — the strongest visually discriminative feature.
  - `min_distance`: tightest inter-quartile range for `fast_approach` — these events happen at close range with low variance.
  - `avg_distance_consistency`: markedly different distribution for `surrounding` (persistent close proximity) vs. `normal`.
  - `approach_persistence` and `avg_closing_rate` show clear upward shifts for threat classes.
  - `target_speed` shows the widest spread for `normal` — normal movement is the most varied.
- **Paper use:** Feature importance narrative. Pick 4–6 of the most discriminative features for a condensed version if journal space is tight. Reference this figure when explaining why those features were included.

---

### Figure PRE-4 — Windows per Video
**File:** `fig_pre4_video_sample_counts.png`
**Source:** `06_video_sample_counts.png` (standalone — aspect ratio too wide to pair)

- **What it shows:** Number of labeled windows extracted from each video clip, coloured by threat type.
- **Key findings:**
  - MOT17-02-DPM and MOT17-04-DPM dominate the `normal` sample count — they are long crowd sequences with no threat events.
  - Threat-labelled windows (`following`, `surrounding`, `fast_approach`) are concentrated in the custom "Movie" clips — these were purpose-recorded threat scenarios.
  - Several Movie clips contain only a single threat type, creating per-video class imbalance that is distinct from the global imbalance.
- **Paper use:** Data provenance section. Explains why per-video accuracy varies so strongly — MOT17 clips give the model "easy" normal contexts while threat clips are shorter and noisier.

---

## 2. Post-Train EDA

These figures analyse **model behaviour on held-out data** after training.
They belong in the **Experiments / Results** section under a sub-heading like
*"Model Behavioural Analysis"* or *"Post-Training EDA"*.

> These are EDA of model outputs — not training metrics. They reveal *where* and *why*
> models succeed or fail, independently of accuracy numbers.

---

### Figure POST-1 — Prediction Confidence Distribution
**File:** `fig_post1_confidence_dist_3models.png`
**Combines:** `bilstm_confidence_dist.png` + `stgat_confidence_dist.png` + `xgb_confidence_dist.png`
**Why paired:** All three are the same plot type measuring the same quantity across models.

- **What it shows:** Histogram of softmax/probability confidence scores per class for each model's predictions on the test set.
- **Key findings:**
  - **BiLSTM:** Strongly bimodal — most predictions are either near-random (≈0.40–0.45) or very confident (≈0.97–1.0). This suggests the model is either certain or confused; it lacks a calibrated middle ground.
  - **STGAT:** Confidence is widely spread (0.3–1.0) and lacks a sharp high-confidence peak for threat classes. STGAT is less decisive on minority classes.
  - **XGBoost:** Similar to STGAT but with a sharper `normal` peak near 1.0 — XGBoost is most confident on the dominant class. Threat class confidence is still spread.
- **Paper use:** Calibration discussion. BiLSTM's bimodal pattern is a red flag for overconfidence. Recommend temperature scaling or Platt scaling if deployment requires probability outputs.

---

### Figure POST-2 — Per-Video Accuracy
**File:** `fig_post2_per_video_acc_3models.png`
**Combines:** `bilstm_per_video_accuracy.png` + `stgat_per_video_accuracy.png` + `xgb_per_video_accuracy.png`
**Why paired:** Same metric (accuracy with 70 % threshold) across all three models.

- **What it shows:** Horizontal bar chart of accuracy per video clip, sorted descending. Blue bars exceed 70 %; red bars fall below.
- **Key findings:**
  - **BiLSTM:** Only 1 video (MOT17-02-DPM) exceeds the 70 % threshold. The model generalises poorly across diverse clips.
  - **STGAT:** 2 videos (both MOT17 clips) exceed threshold. Better on crowd data but still fails on all custom threat clips.
  - **XGBoost:** 5 videos exceed threshold — the only model that passes on any custom threat clips. Strongest overall generalisation.
  - All models fail on `[Following] Movie (3)_003` and `[Surrounding] Movie (4)_010` — these are the hardest clips across the board.
- **Paper use:** Generalisation analysis. Highlights that sequence-based models (BiLSTM, STGAT) overfit to MOT17-style crowd scenes while XGBoost generalises better to diverse threat scenarios. Motivates further data collection for those two hard clips.

---

### Figure POST-3 — Prediction Timelines
**File:** `fig_post3_timeline_stgat_xgb.png`
**Combines:** `stgat_prediction_timeline.png` + `xgb_prediction_timeline.png`
**Why paired:** Same visualisation type across the two most legible models (BiLSTM timeline is too compressed).
**Excluded:** `bilstm_prediction_timeline.png` — dot size too small, panels too compressed to be legible at print resolution.

- **What it shows:** For 4 representative videos, each predicted class label (dot colour) is plotted against window index. The dashed line shows the ground-truth class.
- **Key findings:**
  - On `[Normal] MOT17-02-DPM`, both models correctly track the ground-truth `normal` class almost perfectly.
  - On `[Normal] MOT17-04-DPM`, STGAT spuriously predicts `surrounding` and `fast_approach` throughout — it conflates crowd proximity with threat. XGBoost makes far fewer spurious predictions.
  - On `[Following] Movie (3)_003`, both models largely miss the `following` ground truth — they predict `surrounding` instead. This is the key failure mode: the models cannot distinguish a persistent follower from a surrounding group.
- **Paper use:** Qualitative error analysis. Pair this with the per-video accuracy figure to give a temporal narrative to the accuracy numbers.

---

### Figure POST-4 — XGBoost Error Analysis
**File:** `fig_post4_xgb_error_analysis.png`
**Source:** `xgb_error_analysis.png` (standalone — already a 2×5 multi-panel figure)

- **What it shows:** For each of 10 features, overlaid histograms of the feature distribution for **correctly** vs **incorrectly** classified XGBoost samples.
- **Key findings:**
  - `threat_person_count = 1`: errors are heavily concentrated here. When only one nearby person is detected, the model is most confused — the feature is ambiguous between `following` (one follower) and `normal` (one nearby stranger).
  - `min_distance` errors cluster in the 50–150 pixel range — intermediate distances are the hardest to classify, not very close (obvious threat) nor very far (obvious normal).
  - `avg_closing_rate ≈ 0` causes errors: very slow closing rates are ambiguous across all classes.
  - `max_velocity_toward_target ≈ 0` is disproportionately represented in incorrect predictions — low approach velocity makes `fast_approach` vs `normal` almost indistinguishable.
- **Paper use:** Feature engineering discussion. Each highlighted failure region is a direct recommendation for a new derived feature or threshold. Only XGBoost has this figure because its feature-space is directly interpretable; neural models require SHAP (see XAI section).

---

## 3. XAI / SHAP Analysis

These figures explain **why** the XGBoost model makes specific predictions, and **what** the
BiLSTM and STGAT models attend to. Place in an **Explainability** or **Discussion** section.

> Note: BiLSTM and STGAT XAI figures currently only cover the `normal` class due to a
> generation limitation. Treat them as illustrative rather than definitive.

---

### Figure XAI-1 — SHAP Waterfall Plots (All 4 Classes)
**File:** `fig_xai1_shap_waterfalls_4class.png`
**Combines:** 4 waterfall PNGs in a 2×2 grid
**Why grouped:** One per class — natural 2×2 arrangement showing the full multi-class story.

- **What it shows:** For one representative sample per class, the SHAP value of each feature is shown as a horizontal bar. Blue = pushes toward this class; red = pushes away.
- **Key findings per class:**
  - **(a) Normal (sample #0):** `group_speed_std` is the dominant positive signal — high variance in group movement speed is the clearest marker of a non-threatening scene. `encirclement_ratio` and `threat_person_count` push *against* the normal class.
  - **(b) Following (sample #17):** `threat_person_count` is the single largest positive driver — one person persistently in the "threat zone" is the strongest signal. `avg_distance_consistency` is second — the follower maintains a stable distance. `max_closing_acceleration` *opposes* the following prediction (followers don't accelerate sharply).
  - **(c) Surrounding (sample #5):** `avg_distance_consistency` dominates positively — surrounding is defined by multiple people maintaining stable close distances. `approach_persistence` opposes it — surrounding people don't necessarily keep approaching.
  - **(d) Fast Approach (sample #24):** `threat_person_count` is the dominant **negative** driver — fast approach is a lone-actor pattern, so multiple nearby persons actually argue *against* this class. `target_speed` and `avg_speed_of_group` are the only positive contributors (high motion = fast approach).
- **Paper use:** Core XAI figure. This 2×2 is the most important single figure in the explainability section. Shows the model has learned semantically meaningful cues for each threat type.

---

### Figure XAI-2 — SHAP Interaction Summary
**File:** `fig_xai2_shap_summary.png`
**Source:** `xgboost_shap_summary.png` (standalone — vertical interaction plot format)

- **What it shows:** SHAP interaction values between the top 4 features (`max_closing_speed`, `min_distance`, `threat_person_count`, `num_closing_persons`) across all test samples. Each column shows how one feature's interaction effect is distributed (blue = high feature value, magenta = low).
- **Key findings:**
  - `max_closing_speed` × `threat_person_count`: large spread — this interaction is class-sensitive. Fast closing by many persons is a different signal than fast closing by one.
  - `min_distance` interactions are generally smaller in magnitude — distance alone is not strongly decisive once other features are considered.
- **Paper use:** Secondary XAI figure. Use it to argue that the model captures interaction effects, not just independent feature importances. If space is limited, this can move to the supplementary.

---

### Figure XAI-3 — BiLSTM Attention Weights
**File:** `fig_xai3_bilstm_attention.png`
**Combines:** `bilstm_attention_heatmaps.png` + `bilstm_attention_per_class.png`
**Why paired:** Both visualise the same attention mechanism over the 30-frame temporal window.

> **Limitation:** Only the `normal` class is shown. Other class attention maps were not generated.

- **What it shows:**
  - **(a) Heatmap:** Each row = one test sample; each column = a frame (0–29). Colour = attention weight. Shows how attention is distributed within individual samples.
  - **(b) Mean temporal curve:** Average attention weight per frame across all `normal` samples.
- **Key findings:**
  - Attention follows a **bell curve** peaking at frames 14–18 (mid-window). The BiLSTM model focuses on the middle of the 30-frame window, ignoring the first and last few frames.
  - This is consistent with a sequence model that needs context on both sides (bi-directional) before committing to the central frames.
  - Some samples show irregular attention spikes (high-attention stripes in the heatmap) — these may correspond to motion events worth investigating.
- **Paper use:** Temporal interpretability for BiLSTM. Demonstrates that the model is not just using the most recent frame; it integrates context over the full window with a centre-weighted bias.

---

### Figure XAI-4 — STGAT Graph Attention Weights
**File:** `fig_xai4_stgat_attention.png`
**Combines:** `stgat_attention_timeline.png` + `stgat_person_importance.png`
**Why paired:** Both visualise STGAT's graph-attention mechanism — one temporally, one spatially.

> **Limitation:** Only the `normal` class is shown. Other class attention maps were not generated.

- **What it shows:**
  - **(a) Attention timeline:** Mean attention from the target node to each of its up to 9 nearest-neighbour graph nodes, plotted over the 30-frame window.
  - **(b) Per-node importance bar chart:** Mean attention weight for each person node averaged over all frames.
- **Key findings:**
  - The **closest person (person 1)** receives ~26 % of the target's attention on average; the target node attends to itself at ~30 %. Attention drops off steeply — person 5+ is nearly invisible to the model.
  - Attention weights are **stable over time** — the model does not dynamically re-weight neighbours as the scene evolves. This is a potential architectural limitation for fast-changing threat scenarios.
  - For `normal` scenes, the self-attention dominance is expected (the target person is most predictive of their own "normal" movement).
- **Paper use:** Graph-level interpretability for STGAT. The proximity-decay attention pattern validates that the model has learned a spatially local representation. The temporal stability finding motivates exploring dynamic attention in future work.

---

## Summary Table — Figures for Paper

| Figure File | Section | Type | Standalone/Combined |
|---|---|---|---|
| `fig_pre1_corr_pca.png` | Dataset | Pre-train EDA | Combined (1×2) |
| `fig_pre2_class_distribution.png` | Dataset | Pre-train EDA | Standalone |
| `fig_pre3_feature_boxplots.png` | Dataset | Pre-train EDA | Standalone |
| `fig_pre4_video_sample_counts.png` | Dataset | Pre-train EDA | Standalone |
| `fig_post1_confidence_dist_3models.png` | Results | Post-train EDA | Combined (1×3) |
| `fig_post2_per_video_acc_3models.png` | Results | Post-train EDA | Combined (1×3) |
| `fig_post3_timeline_stgat_xgb.png` | Results | Post-train EDA | Combined (1×2) |
| `fig_post4_xgb_error_analysis.png` | Results | Post-train EDA | Standalone |
| `fig_xai1_shap_waterfalls_4class.png` | Explainability | XAI | Combined (2×2) |
| `fig_xai2_shap_summary.png` | Explainability | XAI | Standalone |
| `fig_xai3_bilstm_attention.png` | Explainability | XAI | Combined (1×2) |
| `fig_xai4_stgat_attention.png` | Explainability | XAI | Combined (1×2) |

**Total figures: 12**

---

## What Was Excluded and Why

| Original File(s) | Reason Excluded |
|---|---|
| `evaluation/bilstm_confusion.png` etc. | Model evaluation metrics, not EDA |
| `evaluation/*_roc.png`, `*_pr.png` | Model evaluation metrics, not EDA |
| `bilstm_prediction_timeline.png` | Too compressed to read at print resolution |
| `xgb_error_analysis.png` (other models) | Only XGBoost has this; neural models use SHAP instead |

---

## Recommended Figure Order in Paper

1. **PRE-2** (class distribution) — opens the dataset section
2. **PRE-4** (video sample counts) — data provenance
3. **PRE-1** (correlation + PCA) — feature structure motivation
4. **PRE-3** (feature boxplots) — detailed feature narrative (can go to appendix)
5. **POST-1** (confidence distributions) — opens results analysis
6. **POST-2** (per-video accuracy) — generalisation comparison
7. **POST-3** (prediction timelines) — qualitative failure mode analysis
8. **POST-4** (XGBoost error analysis) — feature-level failure diagnosis
9. **XAI-1** (SHAP waterfalls 2×2) — core explainability result
10. **XAI-2** (SHAP interaction summary) — supplementary or discussion
11. **XAI-3** (BiLSTM attention) — temporal interpretability
12. **XAI-4** (STGAT attention) — spatial-graph interpretability

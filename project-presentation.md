# Person-Following Threat Detection
### Research Presentation

---

## 1. Problem Statement

Detect **who is threatening a target person** in surveillance video using only trajectory data (no appearance, no re-ID).

**4-class classification:**

| Class | Behaviour |
|---|---|
| `normal` | Random pedestrian movement |
| `following` | One or more persons persistently trailing the target |
| `surrounding` | Group encircling target from multiple directions |
| `fast_approach` | High-speed convergence toward the target |

**Input:** Raw video → YOLO tracking → motion features → classifier → alert

---

## 2. System Architecture

```
Raw Video
    │
    ▼
YOLOv8m + BotSORT  (confidence=0.35, person class only)
    │  per-person (frame, cx, cy) tracks
    ├──► 20 aggregated features / window  ──► XGBoost
    ├──► (30 frames × 8 features) sequences  ──► BiLSTM
    └──► (30 frames × N persons × 4 node features) graphs  ──► STGAT
              │
              ▼
    70 / 15 / 15 video-level split  (no window leakage)
              │
              ▼
    Evaluate → Confusion Matrix, ROC, PR
    XAI      → SHAP (XGBoost), Attention (BiLSTM, STGAT)
```

> 📷 `system_architecture_diagram.png` — block diagram of pipeline *(create in draw.io)*

---

## 3. Dataset

### Sources

| Source | Class | Videos |
|---|---|---|
| MOT17 benchmark (DPM, FRCNN, SDP) | normal | 4 |
| Synthetic Movie clips (Movie 1–5) | normal | 18 |
| Synthetic Movie clips | following | 18 |
| Synthetic Movie clips | surrounding | 18 |
| Synthetic Movie clips | fast_approach | 18 |
| **Total** | | **76 videos** |

### Train / Val / Test Split (video-level, stratified)

| Split | normal | following | surrounding | fast_approach | Total |
|---|---|---|---|---|---|
| Train | 16 | 12 | 12 | 12 | **52** |
| Val | 3 | 3 | 3 | 3 | **12** |
| Test | 3 | 3 | 3 | 3 | **12** |

Split is at the **video level** — all windows from one video go to the same partition, preventing data leakage.

> 📷 `data/outputs/eda/pre_train/01_class_distribution.png` — class balance chart

### Synthetic Data — How It Was Made

The threat classes (following, surrounding, fast_approach) are **actor-performed clips** recorded indoors. Clips segmented into 10–15 s, tracked with YOLO+BotSORT. Labels come from the folder name (`raw_videos/following/`, etc.) — no hand annotation.

**Trajectory augmentation (train split only):**

| Type | Variants | Effect |
|---|---|---|
| Horizontal flip | 1 | Mirrors scene left↔right |
| Rotation | ±5°, ±10° (4 total) | Small viewpoint shift |
| Scale | 0.9×, 1.1× (2 total) | Zoom in/out |

Each training video → **+7 augmented copies** before feature extraction.

### Why Synthetic Data Hurts the Models

- All 5 "Movie" sources are from the **same indoor scene** — models may learn room geometry rather than threat patterns.
- MOT17 (normal class) is real outdoor crowd footage — **domain mismatch** with synthetic negatives.
- Actor-performed threats are clean and symmetric; real threats are partial and messy.
- Augmentation is geometric only; motion style is unchanged.

### How to Make Better Synthetic Data

| Problem | Fix |
|---|---|
| Single scene | Record 3+ environments (indoor, outdoor plaza, parking lot) |
| Correlated actors | Different groups per scene; vary crowd density (2 / 4 / 8 bystanders) |
| Clean scripted motion | Add partial behaviours (follower drops off mid-clip), introduce distractors |
| No appearance variation | Augment **raw videos** with brightness/blur/perspective before tracking |
| Fixed camera | Multiple tilt angles and heights |
| No real validation set | Label 5–10 clips from VIRAT/UCF-Crime as domain-transfer test |

---

## 4. Feature Engineering

### Window Features → XGBoost (20 features per 30-frame window)

| Group | Key Features |
|---|---|
| Proximity | `threat_person_count`, `min_distance` |
| Closing | `num_closing_persons`, `max_closing_speed`, `avg_closing_rate`, `max_velocity_toward_target` |
| Surrounding | `angle_spread`, `converging_sector_count`, `encirclement_ratio` |
| Following | `behind_person_count`, `avg_distance_consistency`, `approach_persistence` |
| Group motion | `avg_speed_of_group`, `group_speed_std`, `target_speed` |
| Physics | `max_closing_acceleration`, `distance_trend_slope`, `group_centroid_closing_rate`, `target_direction_changes`, `synchronized_closing_ratio` |

### Frame Sequences → BiLSTM (30 frames × 8 features)

`n_nearby` · `min_distance_frame` · `mean_distance_frame` · `max_velocity_toward_frame` · `angle_to_nearest` · `behind_count_frame` · `target_vx` · `target_vy`

### Graph Nodes → STGAT (30 frames × N≤10 persons × 4 features)

`[cx, cy, vx, vy]` per node. Node 0 = target. Adjacency = 1 if two persons within 300 px. Self-loops for valid nodes.

> 📷 `data/outputs/eda/pre_train/03_correlation_heatmap.png` — feature correlation heatmap
> 📷 `data/outputs/eda/pre_train/04_pca_scatter.png` — PCA class separability

---

## 5. Models

### 5.1 XGBoost

**Input:** 20 aggregated window features (no temporal order).

| Hyperparameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| objective | multi:softprob |
| class weights | inverse frequency |

The 20 features were hand-crafted to directly target each class:
`encirclement_ratio` → surrounding · `behind_person_count` → following · `synchronized_closing_ratio` → fast_approach

---

### 5.2 BiLSTM

**Input:** (30, 8) per-frame sequence.

```
Input (B, 30, 8)
→ BiLSTM ×2 layers  [hidden=128, bidirectional → 256-dim]
→ Mean pooling over T=30  →  (B, 256)
→ LayerNorm → Dropout(0.3) → Linear(256 → 4)
```

| Hyperparameter | Value |
|---|---|
| Hidden dim | 128 (×2 bidirectional = 256) |
| Layers | 2 |
| Dropout | 0.3 |
| Epochs | 50 |
| LR schedule | Cosine Annealing from 1e-3 |
| Gradient clip | norm = 1.0 |

**XAI proxy attention:** softmax of L2-norm of hidden state per timestep → per-frame importance.

---

### 5.3 STGAT

**Input:** (30, N, 4) node features + (30, N, N) adjacency.

```
Per frame:
  2-layer Multi-Head GAT  [4 heads × 32 → 128-dim, ELU + LayerNorm]
  Target embedding + mean-pooled global embedding → frame vector (256-dim)
Across 30 frames:
  2-layer GRU  [hidden=128]
  LayerNorm → Dropout → Linear(128 → 4)
```

| Hyperparameter | Value |
|---|---|
| GAT hidden | 32 per head × 4 heads = 128 |
| GRU hidden | 128 |
| Dropout | 0.3 |
| Epochs | 25 + early stopping (patience=10) |
| Max persons | 10 (target + 9 nearest, zero-padded) |

**XAI:** `attn[:, :, 0, :]` — how much each surrounding person influenced the target node, per frame.

---

## 6. Evaluation Results

### Model Comparison

| Model | Test Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| **XGBoost** | **0.766** | **0.782** | 0.52 |
| STGAT | 0.696 | 0.721 | 0.43 |
| BiLSTM | 0.656 | 0.699 | 0.46 |

### XGBoost — Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.95 | 0.86 | **0.91** | 2479 |
| following | 0.34 | 0.47 | 0.39 | 266 |
| surrounding | 0.40 | 0.60 | 0.48 | 372 |
| fast_approach | 0.36 | 0.26 | 0.30 | 188 |
| **weighted avg** | 0.81 | 0.77 | **0.78** | 3305 |

> 📷 `data/outputs/eda/evaluation/xgboost_confusion.png`
> 📷 `data/outputs/eda/evaluation/xgboost_roc.png`

### BiLSTM — Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.95 | 0.73 | **0.83** | 2807 |
| following | 0.33 | 0.33 | 0.33 | 369 |
| surrounding | 0.33 | 0.51 | 0.40 | 407 |
| fast_approach | 0.19 | 0.54 | 0.28 | 231 |
| **weighted avg** | 0.78 | 0.66 | **0.70** | 3814 |

> 📷 `data/outputs/eda/evaluation/bilstm_confusion.png`
> 📷 `data/outputs/eda/evaluation/bilstm_roc.png`

### STGAT — Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.92 | 0.82 | **0.86** | 2479 |
| following | 0.47 | 0.35 | 0.40 | 266 |
| surrounding | 0.20 | 0.40 | 0.27 | 372 |
| fast_approach | 0.21 | 0.17 | 0.19 | 188 |
| **weighted avg** | 0.76 | 0.70 | **0.72** | 3305 |

> 📷 `data/outputs/eda/evaluation/stgat_confusion.png`
> 📷 `data/outputs/eda/evaluation/stgat_roc.png`

### Reading the Numbers

**All three models classify `normal` well** (F1 0.83–0.91) — the class is heavily dominant (75%+ of windows) and the cleanest.

**`fast_approach` is the hardest class for all models** — fewest samples (188 test windows), easily confused with surrounding (both involve people closing in). XGBoost scores highest here (F1=0.30) via `synchronized_closing_ratio`.

**XGBoost wins** because its 20 features were explicitly engineered for these four classes. The neural models see less information: BiLSTM gets only 8 instantaneous per-frame scalars; STGAT gets only `[cx, cy, vx, vy]` per node.

**STGAT beats BiLSTM** despite a simpler per-frame representation, because it explicitly models pairwise interactions between persons at each timestep — which is the core of threat detection.

**BiLSTM's `max_velocity_toward_frame` is always zero** (bug: `feature_extractor.py:383–386` computes the proxy but always returns 0.0 because the other person's previous position is not passed through). This silently degrades BiLSTM.

### Dataset Effect on Results

| Effect | Evidence |
|---|---|
| Single synthetic scene | All models get high normal precision but struggle on threat classes — they memorise MOT17 vs. Movie-clip distinction instead of motion patterns |
| 12 test videos → high variance | 3 test videos per class means one misclassification = ~8% accuracy swing |
| Surrounding vs. fast_approach confusion | Precision very low for both across all models — training clips are too similar in motion style |
| STGAT data-starved | 25 epochs + 52 train videos is too little for a 4-layer graph model; F1 on surrounding (0.27) is worse than BiLSTM |

---

## 7. Explainability (XAI)

### XGBoost — SHAP

> 📷 `data/outputs/eda/xai/xgboost_shap_summary.png` — global feature importance across all classes

**Key SHAP findings to check:**
- Does `encirclement_ratio` dominate surrounding? (correct signal)
- Does `behind_person_count` dominate following? (correct signal)
- If `min_distance` dominates everything → model is using a scene-specific proxy, not class-specific motion

> 📷 `data/outputs/eda/xai/xgboost_shap_waterfall_following.png`
> 📷 `data/outputs/eda/xai/xgboost_shap_waterfall_surrounding.png`

### BiLSTM — Temporal Attention

> 📷 `data/outputs/eda/xai/bilstm_attention_per_class.png` — avg attention profile per class across 30 timesteps

**What to look for:** following should peak at later frames (persistence builds); fast_approach should spike mid-window (speed burst).

### STGAT — Graph Attention

> 📷 `data/outputs/eda/xai/stgat_person_importance.png` — mean attention per person index toward target

**What to look for:** surrounding → roughly equal attention on all persons; following → spike on the rear person (node furthest back from target's direction).

---

## 8. Limitations & Priority Fixes

| Issue | Impact | Fix |
|---|---|---|
| BiLSTM velocity feature always zero (`feature_extractor.py:383–386`) | Reduces BiLSTM discriminative power | Pass previous-frame position of each person into `compute_frame_features` |
| Single synthetic scene for all threat clips | Models learn scene geometry, not threat motion | Record 3+ scenes; vary environment and crowd density |
| 76 videos, 12 test | Results not statistically reliable | 5-fold video cross-validation for confidence intervals |
| STGAT only 25 epochs | Graph model underfits | Train 100 epochs with patience=20 |
| Geometric augmentation only | Doesn't diversify motion style | Augment raw videos with brightness/perspective before tracking |
| BiLSTM mean pooling | Temporal attention is crude | Replace with learnable self-attention head |
| No real threat data | Domain gap unknown | Annotate 5–10 VIRAT/UCF-Crime clips for domain-transfer test |

---

## 9. Conclusion

| | XGBoost | BiLSTM | STGAT |
|---|---|---|---|
| Test Accuracy | **0.766** | 0.656 | 0.696 |
| Weighted F1 | **0.782** | 0.699 | 0.721 |
| Normal F1 | **0.91** | 0.83 | 0.86 |
| Following F1 | 0.39 | 0.33 | **0.40** |
| Surrounding F1 | 0.48 | 0.40 | 0.27 |
| Fast-approach F1 | **0.30** | 0.28 | 0.19 |
| Interpretable | SHAP | Proxy attn | GAT attn |
| Data need | Low | Medium | High |

**XGBoost is the current best model** — its hand-crafted features match the task perfectly and it is the most data-efficient. Neural models need more diverse data to outperform it.

**Highest-priority next steps:**
1. Fix BiLSTM velocity feature bug (30 min effort, measurable uplift)
2. Record 2 more scenes per threat class (biggest expected accuracy gain)
3. Train STGAT for 100 epochs (free gain from current data)

---

## Appendix — Images Used in This Presentation

| Image | Path | Status |
|---|---|---|
| Class distribution | `data/outputs/eda/pre_train/01_class_distribution.png` | ✅ |
| Feature correlation | `data/outputs/eda/pre_train/03_correlation_heatmap.png` | ✅ |
| PCA class scatter | `data/outputs/eda/pre_train/04_pca_scatter.png` | ✅ |
| XGBoost confusion matrix | `data/outputs/eda/evaluation/xgboost_confusion.png` | ✅ |
| XGBoost ROC | `data/outputs/eda/evaluation/xgboost_roc.png` | ✅ |
| BiLSTM confusion matrix | `data/outputs/eda/evaluation/bilstm_confusion.png` | ✅ |
| BiLSTM ROC | `data/outputs/eda/evaluation/bilstm_roc.png` | ✅ |
| STGAT confusion matrix | `data/outputs/eda/evaluation/stgat_confusion.png` | ✅ |
| STGAT ROC | `data/outputs/eda/evaluation/stgat_roc.png` | ✅ |
| SHAP summary (XGBoost) | `data/outputs/eda/xai/xgboost_shap_summary.png` | ✅ |
| SHAP waterfall — following | `data/outputs/eda/xai/xgboost_shap_waterfall_following.png` | ✅ |
| SHAP waterfall — surrounding | `data/outputs/eda/xai/xgboost_shap_waterfall_surrounding.png` | ✅ |
| BiLSTM attention per class | `data/outputs/eda/xai/bilstm_attention_per_class.png` | ✅ |
| STGAT person importance | `data/outputs/eda/xai/stgat_person_importance.png` | ✅ |
| System architecture diagram | — | ❌ create manually |

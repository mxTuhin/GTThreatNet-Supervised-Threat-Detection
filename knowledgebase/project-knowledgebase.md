# Person-Follow & Threat Detection — Project Knowledgebase

> Complete technical reference for the project: architecture, data flow, models,
> pattern recognition, temporal analysis, EDA, and explainability.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Structure](#2-project-structure)
3. [Two-Pipeline Architecture](#3-two-pipeline-architecture)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Tracking — How We See People](#5-tracking--how-we-see-people)
6. [Feature Engineering — How Behaviour Becomes Numbers](#6-feature-engineering--how-behaviour-becomes-numbers)
7. [Pattern Recognition — What the System Learns to See](#7-pattern-recognition--what-the-system-learns-to-see)
8. [Temporal Modelling — How Time is Handled](#8-temporal-modelling--how-time-is-handled)
9. [Three Models and Why](#9-three-models-and-why)
10. [Data Augmentation](#10-data-augmentation)
11. [Exploratory Data Analysis (EDA)](#11-exploratory-data-analysis-eda)
12. [Explainable AI (XAI)](#12-explainable-ai-xai)
13. [Training & Evaluation](#13-training--evaluation)
14. [Inference Pipeline](#14-inference-pipeline)
15. [Configuration Reference](#15-configuration-reference)
16. [What Has Been Achieved](#16-what-has-been-achieved)

---

## 1. Problem Statement

CCTV cameras watch public spaces continuously, but detecting **pre-crime threat behaviour** in real time requires a human operator — which does not scale. The goal of this project is to **automatically classify the behavioural relationship between people in a scene** from raw video, identifying four situations:

| Class | Description |
|---|---|
| `normal` | People moving independently, no directed interaction toward a target |
| `following` | One or more persons persistently trail a target, mirroring direction changes |
| `surrounding` | A group converges and encircles a target from multiple angles |
| `fast_approach` | One or more persons close rapidly and aggressively toward a target |

The system does **not** use face recognition, object identity, or audio. It works purely from **movement trajectories** — bounding box positions over time. This makes it privacy-preserving and applicable to any camera angle or resolution.

---

## 2. Project Structure

```
person-follow-research/
│
├── prepare_dataset.py          ← Pipeline 1: videos → image frames
├── pipeline.py                 ← Pipeline 2: frames → trained models (entry point)
├── config.py                   ← All tuneable constants in one place
├── requirements.txt
│
├── data/
│   ├── raw_videos/             ← Original MP4/AVI source videos (input to prepare_dataset.py)
│   │   ├── normal/
│   │   └── threat/
│   │       ├── following/
│   │       ├── surrounding/
│   │       └── fast_approach/
│   │
│   ├── frames/                 ← Extracted image frames (input to pipeline.py)
│   │   ├── normal/
│   │   │   └── <video_name>/   ← One folder per video clip
│   │   │       ├── 000001.jpg
│   │   │       ├── 000002.jpg
│   │   │       └── fps.txt     ← Actual frame rate stored here
│   │   └── threat/
│   │       ├── following/
│   │       ├── surrounding/
│   │       └── fast_approach/
│   │
│   ├── splits.csv              ← Auto-generated: train/val/test assignment per video
│   │
│   ├── outputs/
│   │   ├── csv/                ← Tracked trajectory CSVs (one per video)
│   │   │   └── augmented/      ← Geometrically augmented trajectory CSVs
│   │   ├── video/              ← Annotated tracking videos
│   │   └── eda/                ← All EDA and evaluation plots
│   │       ├── pre_train/
│   │       ├── post_train/
│   │       ├── evaluation/
│   │       └── xai/
│   │
│   └── derived/
│       ├── threat_windows.csv      ← 20-feature rows for XGBoost
│       ├── sequence_windows.npz    ← Per-frame sequences for BiLSTM
│       ├── graph_sequences.npz     ← Graph-structured sequences for STGAT
│       └── models/
│           ├── xgb_model.pkl
│           ├── bilstm_model.pt
│           ├── stgat_model.pt
│           └── threat_label_encoder.pkl
│
└── src/
    ├── data/
    │   ├── dataset_loader.py   ← Discovers videos/frames, assigns splits
    │   ├── feature_extractor.py← All three extraction modes (window/sequence/graph)
    │   ├── augmentation.py     ← Geometric trajectory augmentation
    │   └── data_validator.py   ← Pre-training data quality checks
    │
    ├── tracking/
    │   └── track_video.py      ← YOLO + BotSORT tracker, outputs trajectory CSV
    │
    ├── models/
    │   ├── train_models.py     ← Trains XGBoost, BiLSTM, STGAT
    │   ├── bilstm_model.py     ← BiLSTM architecture + train/predict
    │   └── stgat_model.py      ← STGAT architecture + train/predict
    │
    ├── evaluation/
    │   ├── pre_train_eda.py    ← EDA before training
    │   ├── post_train_eda.py   ← EDA after training (error analysis)
    │   └── evaluate.py         ← Held-out test evaluation
    │
    ├── xai/
    │   └── explainability.py   ← SHAP (XGBoost), attention (BiLSTM), GAT weights (STGAT)
    │
    └── inference/
        └── infer_threat.py     ← Runtime inference on new videos
```

---

## 3. Two-Pipeline Architecture

The project is intentionally split into two separate pipelines with a clean boundary between them.

### Pipeline 1 — `prepare_dataset.py` (run once)

```
raw_videos/<type>/<clip>.mp4
        ↓  cv2.VideoCapture frame-by-frame
        ↓  optional: downsample FPS, resize resolution
        ↓
frames/<type>/<clip>/000001.jpg … fps.txt
```

**Why separate?** YOLO tracking is expensive. If training is re-run (hyperparameter search, new model), we should not re-read and re-decode every video. Extracting frames once to JPEG creates a stable, reusable ML dataset. The `fps.txt` sidecar tells the tracker the exact frame rate so timing calculations remain accurate.

**Key options:**
- `--fps 15` — halves the frame count (still enough for 30fps source, effective 15fps)
- `--resize 1280 720` — standardises resolution before tracking
- `--overwrite` — re-extract clips that already have frames

### Pipeline 2 — `pipeline.py` (the ML pipeline)

```
python pipeline.py          ← no arguments needed; runs everything end-to-end
```

Auto-detects `data/frames/` first; falls back to `data/raw_videos/` if frames have not been extracted yet. The 10 stages run in order:

```
split → track → augment → features → validate → eda-pre → train → evaluate → eda-post → explain
```

Individual stages can be run in isolation:
```
python pipeline.py --stage train --model stgat
python pipeline.py --stage evaluate --model all
```

---

## 4. End-to-End Data Flow

```
Video clips (15–25s, 30fps)
        │
        ▼  prepare_dataset.py
Image frames on disk  (data/frames/)
        │
        ▼  stage: split
splits.csv  — video-level stratified train/val/test assignment
        │
        ▼  stage: track  [YOLO v8m + BotSORT]
Tracked CSVs  (frame_idx, track_id, x1, y1, x2, y2, cx, cy, confidence, direction)
        │
        ▼  stage: augment  [trajectory-level geometric transforms]
Augmented tracked CSVs  (flip / rotate ±5° ±10° / scale 0.9× 1.1×)
splits.csv updated with augmented entries (train split only)
        │
        ▼  stage: features  [parallel via joblib]
        ├── threat_windows.csv       → 20 aggregated features/window  → XGBoost
        ├── sequence_windows.npz     → (T=30, 8) per-frame features   → BiLSTM
        └── graph_sequences.npz      → (T=30, N=10, 4) node features  → STGAT
                                        + (T=30, N×N) adjacency
        │
        ▼  stage: validate
Data quality report (class balance, outliers, missing values, correlation)
        │
        ▼  stage: eda-pre
Visualisations: class distribution, feature boxplots, PCA/UMAP, correlation heatmap
        │
        ▼  stage: train  [XGBoost + BiLSTM (GPU) + STGAT (GPU)]
Trained model files  (data/derived/models/)
        │
        ▼  stage: evaluate
Test-split metrics: accuracy, weighted F1, confusion matrices, ROC/PR curves
        │
        ▼  stage: eda-post
Error analysis, per-video accuracy, prediction timeline, confidence distributions
        │
        ▼  stage: explain
XAI outputs: SHAP plots, BiLSTM temporal attention, STGAT person-attention heatmaps
```

---

## 5. Tracking — How We See People

**Model:** YOLOv8m (person class only, confidence threshold 0.35)
**Tracker:** BotSORT with ReID — maintains consistent person IDs across frames even when someone temporarily leaves the frame (5-second track buffer)

**Output per detection (one CSV row):**

| Field | Description |
|---|---|
| `frame_idx` | Absolute frame number in the video |
| `track_id` | Persistent integer identity for this person |
| `x1, y1, x2, y2` | Bounding box corners in pixels |
| `cx, cy` | Bounding box centre — the only position used by features |
| `confidence` | YOLO detection confidence |
| `direction` | Smoothed movement direction (right / down-right / …) |
| `entry_side` | Which edge the person first appeared from |

From this point forward, the pipeline works **only on (cx, cy) coordinates**. The actual pixel content of the frames is never used again. This is why pixel-level augmentation (brightness, blur, colour jitter) has no effect — it only matters if it changes where YOLO places the bounding box, which is minor.

---

## 6. Feature Engineering — How Behaviour Becomes Numbers

Trajectories are processed using a **sliding window** over time:

```
WINDOW_SIZE   = 30 frames = 1.0 second at 30fps
WINDOW_STRIDE = 10 frames = new window every 0.33 seconds  (67% overlap)
```

For every window, every person in frame becomes a potential **target**. All other persons within `PROXIMITY_RADIUS = 300px` become the **threat group** for that target.

Three distinct feature representations are extracted — one for each model:

---

### 6.1 Window Features (20 scalars) → XGBoost

Each window × target pair is summarised into 20 handcrafted features that capture the aggregate geometry and dynamics of the threat group relative to the target.

**Proximity:**
- `threat_person_count` — how many persons are within 300px
- `min_distance` — distance of the closest person

**Closing / approach:**
- `num_closing_persons` — count with positive average closing rate
- `max_closing_speed` — fastest approach speed (px/frame)
- `avg_closing_rate` — mean closing rate across the group
- `max_velocity_toward_target` — maximum instantaneous velocity component directed at target
- `max_closing_acceleration` — peak rate of change of closing speed (detects sudden lunges)
- `distance_trend_slope` — linear slope of minimum distance over the window (negative = closing)

**Surrounding geometry:**
- `angle_spread` — circular standard deviation of angles from target to each person (high = spread around all sides)
- `converging_sector_count` — how many of the 4 quadrants have at least one closing person (0–4)
- `encirclement_ratio` — fraction of the 360° arc covered, computed as (360° − largest_gap) / 360°

**Following / trailing:**
- `behind_person_count` — persons positioned behind the target's direction of movement
- `avg_distance_consistency` — inverse of the distance coefficient of variation (high = stable trailing, signature of following)
- `approach_persistence` — fraction of frames in the window where at least one person is closing

**Group motion:**
- `avg_speed_of_group` — mean speed of all nearby persons
- `group_speed_std` — how coordinated or dispersed their speeds are
- `target_speed` — target person's own speed
- `group_centroid_closing_rate` — rate at which the group's centroid closes on the target

**Temporal coordination:**
- `target_direction_changes` — how many times the target changed direction (>45°) — a reactive target is more suspicious
- `synchronized_closing_ratio` — fraction of frames where the majority of persons close simultaneously (hallmark of coordinated approach)

---

### 6.2 Per-Frame Sequence Features (8 values/frame) → BiLSTM

For each frame in the window, 8 **instantaneous** features are computed. The result is a `(T=30, 8)` sequence fed to the BiLSTM.

| Feature | Meaning |
|---|---|
| `n_nearby` | Count of persons within proximity radius at this frame |
| `min_distance_frame` | Distance to closest person at this frame |
| `mean_distance_frame` | Average distance to all nearby persons |
| `max_velocity_toward_frame` | Peak velocity component directed at target |
| `angle_to_nearest` | Compass angle to the nearest person |
| `behind_count_frame` | Number of persons positioned behind target |
| `target_vx, target_vy` | Target's frame-to-frame displacement vector |

This representation feeds time-steps to the BiLSTM in order, allowing it to learn **how the scene evolves over 1 second** — not just a summary of it.

---

### 6.3 Graph Node Features (4 values/node/frame) → STGAT

For each frame in the window, every tracked person is a **node** in a graph. Edges connect any two persons within the proximity radius.

```
Nodes (up to N_MAX = 10):
  Node 0   = target person  (always at index 0)
  Node 1-9 = nearest persons sorted by distance to target, zero-padded if fewer

Node features per frame: [cx, cy, vx, vy]
  cx, cy  = pixel position (normalised to zero-mean unit-variance at train time)
  vx, vy  = frame-to-frame displacement (velocity)

Adjacency matrix (10×10):
  A[i][j] = 1.0  if euclidean(i, j) ≤ 300px  AND  both nodes are real (not padding)
  A[i][i] = 1.0  (self-loops for all real nodes)
  A[i][j] = 0.0  otherwise (including padded nodes)
```

The result per window is an `(T=30, N=10, 4)` tensor `X` and an `(T=30, N=10, N=10)` tensor `A`, both saved in `graph_sequences.npz`.

---

## 7. Pattern Recognition — What the System Learns to See

### Following
The system detects following through:
- High `avg_distance_consistency` — a follower maintains a steady gap, unlike a random passer-by
- High `behind_person_count` — the follower stays behind the target's movement direction
- Low `approach_persistence` combined with low `max_closing_speed` — they approach but maintain distance, not closing in aggressively
- STGAT learns this as a temporal graph pattern where Node 1 (nearest person) consistently stays in a low-angle sector behind Node 0 (target) across all 30 timesteps

### Surrounding
The system detects surrounding through:
- High `encirclement_ratio` (e.g. > 0.7) — the 360° arc around the target is well-covered
- High `angle_spread` — persons are not all on one side
- High `converging_sector_count` (e.g. 3–4) — multiple quadrants are closing simultaneously
- High `synchronized_closing_ratio` — everyone closes at the same time, indicating coordination
- STGAT naturally captures this: the attention pattern shows the target node (Node 0) distributing attention across 4–6 nodes positioned at very different compass angles

### Fast Approach
The system detects fast approach through:
- High `max_closing_speed` and `max_closing_acceleration`
- Steep negative `distance_trend_slope` — distance falling rapidly
- High `max_velocity_toward_target`
- Low `avg_distance_consistency` — the distance is not stable, it's dropping fast
- In the sequence models (BiLSTM, STGAT), this shows as a sudden spike in velocity features in the final frames of the window

### Normal
Normal walking patterns are characterised by:
- Low `num_closing_persons`
- Low `encirclement_ratio` (people pass through, not围绕)
- Low `behind_person_count` — no persistent trailer
- High `target_direction_changes` only if random (not reactive)
- The graph for a normal scene: edges exist but the target node's attention is diffuse and scattered — no person is consistently close or closing

---

## 8. Temporal Modelling — How Time is Handled

### The Sliding Window Strategy

Rather than treating a video as one long sequence (impractical, computationally heavy), the pipeline uses **overlapping sliding windows**:

```
Window 0: frames  0–29   (seconds 0.00–1.00)
Window 1: frames 10–39   (seconds 0.33–1.33)
Window 2: frames 20–49   (seconds 0.67–1.67)
...
```

Each window is an independent training sample. The 67% overlap means:
1. A single threat event spanning 3–4 seconds generates multiple training samples, not just one — amplifying the dataset
2. A model that predicts `following` on 8 consecutive overlapping windows is much more reliable than a single prediction

**For a 20-second video at 30fps:** 58 windows × ~3 target-group pairs per window ≈ 174 training samples per video. After augmentation (×7): ~1,218 samples per video.

### XGBoost — No Temporal Memory

XGBoost receives the 20 aggregated window features as a flat vector. It has **no memory across windows**. It classifies each 1-second snapshot independently. Time is encoded implicitly through the aggregated statistics (e.g., `distance_trend_slope` is a temporal feature computed within the window, but the model itself is not recurrent).

### BiLSTM — Sequential Memory Within a Window

The BiLSTM receives the `(T=30, 8)` sequence and processes it step-by-step:

```
Frame 1 → h₁
Frame 2 → h₂ (conditioned on h₁)
...
Frame 30 → h₃₀

Bidirectional: also processes frames 30 → 1
Final representation: mean pooling of all [h₁ … h₃₀, h₃₀_back … h₁_back]
Classification head → 4 class logits
```

The bidirectional design means the model sees each frame in the context of both what came before **and** what came after — useful for recognising patterns like "group was far, now they are close" (fast approach) or "distance stayed constant throughout" (following).

**Limitation:** The BiLSTM receives only **aggregate statistics** at each timestep (8 numbers that squash the entire nearby group into a single observation). It cannot distinguish between "2 people to the left" and "2 people on opposite sides".

### STGAT — Graph-Based Spatial-Temporal Memory

STGAT is architecturally superior for this problem because it preserves the **full spatial structure** of the scene at each timestep.

**Spatial processing (GAT) at each frame:**
```
Each person = a node with features [cx, cy, vx, vy]
Attention: Node i learns how much to attend to Node j based on
           their feature similarity and relative position

α(i,j) = softmax_j [ LeakyReLU( a^T · [W·hᵢ || W·hⱼ] ) ]

The target node (Node 0) aggregates information from all nearby
persons weighted by how "relevant" each is to the current threat context.
```

**Temporal processing (GRU) across frames:**
```
GAT outputs at t=0,1,...,29 → sequence of frame-level embeddings
GRU processes this sequence → captures how the scene evolves
Final GRU hidden state → classification head
```

This means STGAT genuinely learns: "at each moment, which persons are significant, and how does that significance evolve over the 1-second window?"

---

## 9. Three Models and Why

| | XGBoost | BiLSTM | STGAT |
|---|---|---|---|
| **Type** | Classical ML (gradient boosted trees) | Deep Learning (recurrent NN) | Deep Learning (graph NN) |
| **Input** | 20 handcrafted scalars | Sequence of 8 aggregate values per frame | Raw (cx,cy,vx,vy) per person per frame + adjacency |
| **Temporal handling** | None (within-window aggregates only) | Recurrent, within-window | Recurrent (GRU), within-window |
| **Spatial handling** | Implicit in features | Collapsed to scalars | Explicit graph — each person is a node |
| **Training time (RTX 3060)** | ~1–3 min (CPU) | ~5–15 min (GPU) | ~25–40 min (GPU) |
| **Strengths** | Interpretable, fast, robust on small data | Captures temporal evolution | Full spatial-temporal structure |
| **Weaknesses** | Hand-engineering required, no spatial structure | Loses who-is-where | More data/training needed |
| **XAI method** | SHAP values | Temporal attention proxy | GAT attention weights |

**Why keep all three:**
- XGBoost is the fast, interpretable baseline. Its SHAP values validate that the engineered features are meaningful.
- BiLSTM is the temporal recurrent baseline. Comparing it to STGAT isolates the benefit of preserving spatial structure.
- STGAT is the strongest model architecturally. Its attention weights provide the richest XAI.

Having all three allows fair comparison on identical held-out test splits.

---

## 10. Data Augmentation

Because recording threat scenarios is time-consuming, the pipeline augments the **training split only** at the trajectory level. Val and test splits are never augmented (no data leakage into evaluation).

**Stage:** `stage_augment` in `pipeline.py` — runs after tracking, before feature extraction.

**Input:** Tracked CSV files from `data/outputs/csv/`
**Output:** 6 new CSVs per training video in `data/outputs/csv/augmented/`

| Augmentation | Transform | Invariance gained |
|---|---|---|
| `flip` | Mirror all `cx` around scene x-midpoint: `cx_new = 2·cx_center − cx` | Left-right handedness |
| `rot+5`, `rot-5` | Rotate all (cx, cy) ±5° around scene centroid | Small camera angle variation |
| `rot+10`, `rot-10` | Rotate all (cx, cy) ±10° around scene centroid | Larger camera angle variation |
| `scale09`, `scale11` | Scale coordinates 0.9× / 1.1× around centroid | Camera zoom / distance variation |

**Multiplication factor:** Each training video generates 6 additional augmented trajectories → **7× training data** before feature extraction.

**What augmentation does NOT do:**
- Pixel-level augmentation (brightness, contrast, blur, colour jitter) has no effect because by the time augmentation runs, we are working with trajectory coordinates `(cx, cy)`, not pixels. Pixel augmentation would only help if applied to frames before running YOLO (expensive and minimal gain).

---

## 11. Exploratory Data Analysis (EDA)

EDA is split into two phases, both automated as pipeline stages.

### Pre-Training EDA (`stage: eda-pre`)

Run on `threat_windows.csv` before any model is trained. Output: `data/outputs/eda/pre_train/`

| Plot | What it reveals |
|---|---|
| `01_class_distribution.png` | Whether classes are balanced; how many samples we have per class |
| `02_feature_boxplots.png` | Whether each feature separates the classes visually; which features have high within-class variance |
| `03_correlation_heatmap.png` | Whether any features are redundant (>0.95 correlation = one is likely derivable from the other) |
| `04_pca_scatter.png` | Whether the four classes are linearly separable in 2D; proximity of clusters indicates confusion risk |
| `04b_umap_scatter.png` | Non-linear structure in the data (if `umap-learn` is installed) |
| `05_missing_values.png` | Where NaNs appear, which videos have sparse tracking |
| `06_video_sample_counts.png` | Per-video window count — identifies videos that are too short or contribute too few samples |

**Key questions the pre-train EDA answers:**
- "Do we have enough data?" → class distribution + sample counts
- "Will features actually help?" → boxplots showing clear class separation
- "Are there any obvious data problems?" → missing heatmap + validator output
- "Are our 20 features capturing something real?" → PCA/UMAP showing cluster structure

### Post-Training EDA (`stage: eda-post`)

Run after training on the XGBoost model's predictions. Output: `data/outputs/eda/post_train/`

| Plot | What it reveals |
|---|---|
| `xgb_confidence_dist.png` | How confident the model is when it predicts each class; low confidence = ambiguous cases |
| `xgb_error_analysis.png` | Compares feature value distributions between correctly and incorrectly classified windows |
| `xgb_per_video_accuracy.png` | Which specific videos the model struggles on; videos below 70% accuracy are flagged red |
| `xgb_prediction_timeline.png` | Whether predictions are stable over time within a video or flip erratically ("flickering") |

**Key questions post-train EDA answers:**
- "Where is the model failing?" → error analysis shows which feature ranges cause mistakes
- "Is there a bad video in our dataset?" → per-video accuracy flags outliers
- "Does the model behave consistently at runtime?" → prediction timeline shows if the output would be stable for a real operator

### Data Validation (`stage: validate`)

Automated quality checks run between feature extraction and EDA:
- Label completeness (any rows without threat_type)
- Minimum 30 samples per class
- Class imbalance ratio (warns if > 5×)
- Missing feature values per column
- Outlier detection using 3×IQR rule
- Feature pairs with correlation > 0.95 (potential redundancy)
- Verifies all video names in `threat_windows.csv` appear in `splits.csv`

---

## 12. Explainable AI (XAI)

Three different XAI methods are used, one per model. All outputs go to `data/outputs/eda/xai/`.

### 12.1 SHAP for XGBoost

**Library:** `shap.TreeExplainer` (exact Shapley values for tree-based models, fast)

**Global explanation** — answers "which features matter most overall?":
- Beeswarm / summary plot per class: each dot is one sample; x-axis = SHAP value (impact on prediction), colour = feature value magnitude
- Mean |SHAP| bar chart: aggregated importance across all classes — tells you definitively which of the 20 features the model leans on

**Local explanation** — answers "why did this specific window get predicted as `following`?":
- Waterfall chart: shows the top contributing features and their direction for one particular sample
- Features pushing toward the predicted class shown in blue; features pushing away in red

**What you typically see:**
- For `following`: `avg_distance_consistency` and `behind_person_count` are the top SHAP drivers
- For `surrounding`: `encirclement_ratio` and `converging_sector_count` dominate
- For `fast_approach`: `max_closing_speed` and `max_closing_acceleration` are decisive
- For `normal`: all the above are low; `target_speed` varies freely

### 12.2 Temporal Attention Weights for BiLSTM

The BiLSTM does not use a trainable attention mechanism. Instead, a **proxy attention** is computed from the L2 norms of LSTM output states:

```
attn_t = softmax( ‖h_t‖₂ )  for t = 0 … 29
```

A high attention weight at time t means the LSTM hidden state was "active" — information was flowing strongly at that timestep.

**Plots generated:**
- Mean temporal attention per class (line plot, T=30): which frames within a 1-second window matter most
- Attention heatmaps: one row per sample, columns = frames; colour = attention weight

**What you typically see:**
- `fast_approach`: attention spikes in later frames (the approach accelerates toward the window end)
- `following`: attention is relatively uniform (consistent behaviour throughout)
- `surrounding`: attention distributed, with peaks when the encirclement completes

Optional: `shap.DeepExplainer` computes feature importance at the frame level for BiLSTM if SHAP is installed — shows which of the 8 per-frame features are most influential.

### 12.3 GAT Attention Weights for STGAT

This is the richest XAI in the project because the attention weights are **a direct product of the model's computation** — not a post-hoc approximation.

At every timestep t, the GAT computes:

```
α[b, t, i, j] = how much node j influenced node i's representation at timestep t
```

For threat detection, we focus on `α[:, :, 0, :]` — how much the **target node (Node 0)** attended to each nearby person across all timesteps.

**Plots generated:**
- `stgat_person_importance.png`: For each class, a bar chart of mean attention from target to each node slot (Person 1, Person 2, ...). Shows which persons the model found most relevant when making its prediction.
- `stgat_attention_timeline.png`: For each class, attention weight of each nearby person over the 30 timesteps. Shows when each person became relevant during the window.

**What you typically see:**
- `following`: Person 1 (nearest, behind target) dominates attention throughout all timesteps — steady high attention
- `surrounding`: Attention distributed across Persons 1–5 at different timesteps — no single dominant person
- `fast_approach`: Attention on Person 1 spikes in final 5–10 frames as they close in
- `normal`: Attention scattered and low-magnitude across all persons — no clear pattern

**This is why GAT attention IS the XAI:** It directly answers "which person was the model watching, and when?" — no separate explainer needed.

---

## 13. Training & Evaluation

### Train/Val/Test Split Strategy

Splitting is done at the **video level**, not the window level. This is critical to prevent data leakage:

```
If a 20-second video produces 58 windows, and those windows are randomly
split into train/val/test, then overlapping windows from the same video
appear in both train and test — the model memorises the video, not the behaviour.

The correct approach: assign each ENTIRE VIDEO to exactly one split.
  Training videos → all their windows go to train
  Validation videos → all their windows go to val
  Test videos → all their windows go to test
```

Splits are stratified by `threat_type` so each class is proportionally represented in all three splits.

**Ratios:** 70% train / 15% val / 15% test (configurable in `config.py`)

### Evaluation Metrics

For every model, the evaluate stage produces:
- **Weighted F1 score** — primary metric (handles class imbalance correctly)
- **Accuracy** — overall correctness
- **Per-class precision, recall, F1** — where exactly the model is strong or weak
- **Confusion matrix** — raw counts and normalised (shows which classes get confused)
- **ROC curves** — AUC per class, one-vs-rest
- **Precision-Recall curves** — important for imbalanced classes

### Hardware

- GPU: NVIDIA RTX 3060 (12GB VRAM, CUDA 12.4)
- CPU: used for XGBoost training and joblib-parallel feature extraction
- Estimated full pipeline runtime: 40–90 minutes depending on dataset size

---

## 14. Inference Pipeline

```
python pipeline.py --stage infer --raw-video new_clip.mp4
```

The inference flow:
1. If `--raw-video` is given, runs `track_video.py` on it first → produces tracked CSV
2. Loads the tracked CSV, runs `extract_windows()` → 20-feature rows
3. Loads the trained model (auto-selects XGBoost if available)
4. Predicts class and threat_score for each window
5. Saves scored CSV to `data/derived/predictions.csv`
6. If `--output-video` is given, writes an annotated video with red alert overlays

**Threat score:** The sum of probabilities assigned to non-normal classes. `threat_score ≥ 0.5` (configurable) triggers an alert.

**Annotated video output:**
- Red border + "THREAT: FOLLOWING (0.87)" label overlaid on frames within any threat window
- Green "NORMAL (0.12)" label otherwise

---

## 15. Configuration Reference

All constants are centralised in `config.py`. Key parameters:

| Constant | Default | Description |
|---|---|---|
| `WINDOW_SIZE` | 30 | Frames per sliding window (1 second at 30fps) |
| `WINDOW_STRIDE` | 10 | Frames between window starts (0.33s step, 67% overlap) |
| `PROXIMITY_RADIUS` | 300 px | Max distance to be counted as a nearby threat person |
| `MIN_NEARBY` | 1 | Minimum nearby persons required to emit a feature row |
| `MIN_TRACK_POINTS` | 10 | Minimum detections per person per window to be included |
| `CONFIDENCE_THR` | 0.35 | YOLO detection confidence threshold |
| `GRAPH_N_MAX` | 10 | Max persons per graph (target + 9 nearest) |
| `GRAPH_NODE_DIM` | 4 | Node feature dimension [cx, cy, vx, vy] |
| `STGAT_GAT_HIDDEN` | 32 | Per-head GAT output dimension |
| `STGAT_GAT_HEADS` | 4 | Number of GAT attention heads (total output = 128) |
| `STGAT_GRU_HIDDEN` | 128 | GRU hidden state dimension |
| `STGAT_EPOCHS` | 100 | Training epochs for STGAT |
| `BILSTM_HIDDEN` | 128 | BiLSTM hidden size |
| `BILSTM_LAYERS` | 2 | Number of stacked BiLSTM layers |
| `BILSTM_EPOCHS` | 50 | Training epochs for BiLSTM |
| `XGB_N_ESTIMATORS` | 300 | XGBoost trees |
| `XGB_LR` | 0.05 | XGBoost learning rate |
| `CV_FOLDS` | 5 | Cross-validation folds (video-grouped, no leakage) |
| `THREAT_SCORE_THRESHOLD` | 0.5 | Inference alert threshold |
| `TRAIN_RATIO` | 0.70 | Video-level train split |
| `VAL_RATIO` | 0.15 | Video-level val split |
| `TEST_RATIO` | 0.15 | Video-level test split |

---

## 16. What Has Been Achieved

### Core system
- **A fully automated end-to-end pipeline** that takes raw CCTV-style video clips and produces trained, evaluated, and explained threat detection models. A researcher with new videos runs two commands: `python prepare_dataset.py` then `python pipeline.py`.

### Novel problem framing
- The problem is framed as **multi-agent spatial-temporal classification** rather than single-person action recognition. The unit of analysis is the relationship between the group and the target — not any individual's pose or action.

### Three-model comparative framework
- **XGBoost** as the interpretable classical ML baseline (fast, low data requirement, SHAP-explainable)
- **BiLSTM** as the recurrent sequence baseline (processes temporal evolution, attention proxy XAI)
- **STGAT** as the graph-neural-network approach (explicit person-graph, GAT attention as built-in XAI — the architecturally appropriate model for multi-agent interaction)

### Rich feature engineering
- **20 window features** that are physically meaningful: they capture proximity, approach dynamics, encirclement geometry, following signatures, group coordination, and temporal physics (acceleration, synchronisation ratio)
- These features translate domain knowledge directly into numbers, making the XGBoost model interpretable by design

### Built-in XAI at three levels
- **Feature level (SHAP):** which of the 20 behaviours drove the prediction
- **Time level (BiLSTM attention):** which frames within the 1-second window mattered
- **Person level (GAT attention):** which specific nearby person the model was "watching" — the most direct answer to "who is the threat?"

### Data efficiency measures
- Video-level splits with stratification — no data leakage
- 6× trajectory augmentation (flip, 4 rotations, 2 scales) — multiplies training data without re-running expensive tracking
- Parallel feature extraction via joblib — all CPU cores used for the compute-heavy extraction step

### Clean separation of concerns
- Two-pipeline architecture separates the expensive one-time frame extraction from the iterative ML pipeline
- All business logic lives in `src/` modules; `pipeline.py` is a pure orchestrator with no logic
- A single `config.py` governs all hyperparameters — no magic numbers anywhere in the codebase

### Production-ready inference
- New video → threat alert in a single command
- Annotated output video with overlay labels and confidence scores
- Threshold-configurable alert sensitivity

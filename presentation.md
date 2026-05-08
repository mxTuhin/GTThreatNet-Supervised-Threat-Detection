# Pre-Attack Threat Detection from Surveillance Video
### Project Presentation

---

## Slide 1 — Title

**Pre-Attack Threat Detection in Surveillance Video Using Trajectory-Based Anomaly Classification**

> Detecting person-level threat scenarios — following, fast approach, and encirclement — from CCTV footage using automated tracking and supervised machine learning.

- **Author:** [Your Name]
- **Supervisor:** [Professor Name]
- **Domain:** Computer Vision · Intelligent Surveillance · Public Safety
- **Date:** April 2026

---

## Slide 2 — Motivation

### Why does this matter?

- **CCTV cameras are everywhere** — cities, campuses, transit hubs — yet surveillance is still mostly *reactive*: reviewed after an incident, not during.
- **Human operators fatigue fast.** A single operator cannot monitor 20+ feeds simultaneously. Studies show attention drops sharply after 20 minutes of passive watching.
- **Pre-attack behavior is visible.** Criminological research shows that targeted violence follows predictable spatial patterns:
  - A perpetrator *trails* a target before striking
  - An attacker group *closes in from behind* rapidly
  - Multiple assailants *fan out and surround* a victim
- **There is a detection window** — seconds to minutes before physical contact — during which an automated alert could enable intervention.
- **Current AI surveillance tools** mostly detect post-event activities (fighting, fallen persons) rather than *pre-attack spatial dynamics*.

---

## Slide 3 — Problem Statement

### What exactly are we solving?

> **Given a surveillance video clip, automatically determine whether one or more persons in the scene are exhibiting pre-attack behavior toward a designated target person.**

### Threat Scenarios Targeted

| Scenario | Description | Key Signal |
|---|---|---|
| `normal` | Regular pedestrian movement | Baseline class |
| `following` | 1 person trailing a target at consistent distance/speed | Stable distance, mirrored direction |
| `fast_approach` | 1–3 persons rapidly closing from behind | High closing velocity, rear approach angle |
| `surrounding` | 2+ persons converging from multiple directions | High angle spread, multi-sector convergence |

### Why it is hard

- All three threat types look like ordinary walking motion in early frames
- Crowded scenes generate many false co-proximity events (e.g., a crowded street)
- Standard object detection gives bounding boxes — spatial relationships across time require additional modeling
- Labeled threat video data is scarce; most datasets cover post-event violence, not pre-attack behavior

---

## Slide 4 — Proposed Idea (High-Level Approach)

### Core Insight

**Threat is a relational, temporal signal** — not a property of a single person's pose or appearance, but of *how a group of persons move relative to a target over time*.

### Approach: Trajectory-Feature Classification

Instead of processing raw pixels through a deep network (data-hungry, black-box), we:

1. **Track every person** in the scene frame-by-frame using a detection + tracking model
2. **Extract group-level geometric features** within sliding time windows around each potential target
3. **Classify each window** as normal vs. one of three threat types using an interpretable ML classifier

### Two-Stage Design (Hybrid Architecture)

```
Stage 1 — Anomaly Filter (Unsupervised)
    Isolation Forest trained on normal scenes only
    → Flags: "something unusual here"
    → Can be bootstrapped with ZERO labeled threat data

Stage 2 — Threat Classifier (Supervised)
    Random Forest on 12 hand-crafted group features
    → Classifies: normal / following / surrounding / fast_approach
    → Only activated when Stage 1 triggers (reduces false positives)
```

### Why this design?

- Stage 1 catches novel threats not seen in training
- Stage 2 provides *actionable* output — security teams need to know the threat *type*, not just that something is unusual
- Stage 1 needs only normal footage — labeling effort is minimized
- Interpretable features allow debugging and explainability

---

## Slide 5 — System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│              CCTV Video Feed / Recorded MP4                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Detection & Tracking (GPU)                    │
│   YOLOv8 / YOLO11  →  Person Bounding Boxes (per frame)    │
│   BoT-SORT (ReID)  →  Persistent Track IDs across frames   │
│   Output: CSV  [frame_idx, track_id, x1,y1,x2,y2, cx,cy]  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Feature Extraction (CPU)                         │
│   Sliding window (30 frames, stride 15)                    │
│   For each target T: compute 12 group-level features       │
│   covering proximity, velocity, angle spread, closing rate  │
│   Output: threat_windows.csv  [feature row per window]     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────┐   ┌──────────────────────────┐
│  Stage 1: Anomaly Filter     │   │  Stage 2: Threat Typing  │
│  Isolation Forest            │──▶│  Random Forest / GBM     │
│  (normal scenes only)        │   │  (labeled windows)       │
│  Output: anomaly score       │   │  Output: threat class    │
└──────────────────────────────┘   └──────────────┬───────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────┐
                                   │  Alert / Visualization   │
                                   │  Annotated video overlay │
                                   │  Per-target threat label │
                                   │  Confidence score        │
                                   └──────────────────────────┘
```

### Data Flow Summary

| Stage | Input | Output |
|---|---|---|
| Tracking | Raw MP4 | Per-frame CSVs with track IDs |
| Feature Extraction | Track CSV | 12-feature rows (one per window per target) |
| Anomaly Filter | Feature rows | Anomaly flag (binary) |
| Threat Classifier | Flagged rows | Threat type + confidence |
| Visualization | Classifier output | Annotated MP4 with alert overlay |

---

## Slide 6 — Technology Stack

### Models

| Component | Model | Reason |
|---|---|---|
| Person Detection | YOLOv8m / YOLO11l | Real-time speed, strong pedestrian detection |
| Multi-object Tracking | BoT-SORT with ReID | Robust ID persistence across occlusion |
| Anomaly Filter (Stage 1) | Isolation Forest | Works with normal-only data, no threat labels needed |
| Threat Classifier (Stage 2) | Random Forest / Gradient Boosting | Small-data robust, interpretable feature importances |
| (Future) Sequence Modeling | Shallow LSTM | When 500+ samples/class become available |

### Feature Engineering — 12 Group-Level Features

Computed per target T in a 30-frame window over all persons within 300px proximity:

| Feature | Threat Signal |
|---|---|
| `threat_person_count` | High in surrounding, low in following |
| `num_closing_persons` | High in approach and surrounding |
| `max_closing_speed` | Spike in fast_approach |
| `avg_closing_rate` | Near-zero in following (stable distance) |
| `angle_spread` | High in surrounding (multiple directions) |
| `converging_sector_count` | 3–4 sectors = surrounding; 1 sector = approach |
| `min_distance` | Minimum safe distance breached |
| `avg_speed_of_group` | Fast in approach, moderate in following |
| `behind_person_count` | High in following and approach |
| `avg_distance_consistency` | High (stable) in following, low in approach |
| `max_velocity_toward_target` | Spike in fast_approach |
| `target_speed` | Slow target = more vulnerable context |

### Dataset Creation Strategy

**Real videos:**
- Publicly available pedestrian surveillance datasets (MOT, UCF-Crime)
- Self-recorded controlled scenarios with consenting participants

**Synthetic videos:**
- Procedurally generated top-down trajectory simulations (configurable patterns)
- Rendered crowd simulations with scripted threat behavior
- Augmentation via window stride variation, speed scaling, mirror/flip of trajectories

### Data Structure

```
data/raw_videos/
    normal/              ← regular pedestrian scenes
    threat/
        following/
        surrounding/
        fast_approach/
data/annotations/video_labels.csv
data/derived/threat_windows.csv   ← ML training input (auto-labeled from folder)
data/derived/threat_model.pkl     ← trained classifier
```

### Hardware

- GPU: NVIDIA RTX 3060, CUDA 13.2 (tracking inference)
- CPU: sklearn models (fast enough for tabular feature classification)
- Framework: PyTorch (YOLO), scikit-learn (RF/GBM), Python 3.12

---

## Slide 7 — Validation (Current Work Progress)

### What is implemented and working

| Component | Status |
|---|---|
| `src/track_video.py` | Done — YOLO + BoT-SORT tracking, CSV output, CLI args for batch use |
| `src/threat_feature_extractor.py` | Done — 12-feature sliding window extraction, auto-labels from folder |
| `train_threat_model.py` | Done — RF + GBM + SVM, video-grouped cross-validation |
| `infer_threat.py` | Done — scores new tracked CSV, optional annotated MP4 output |
| `batch_process_videos.py` | Done — end-to-end batch pipeline across all raw videos |
| `src/utilities/trajectory_utils.py` | Done — shared geometry math (angle, velocity, spread) |
| `src/following_logic.py` | Done (legacy) — heuristic pairwise follower scorer, kept as reference |

### Feature Discrimination Validated (Design Phase)

The 12 features were designed to be discriminative across all three threat types:

| Discriminator | Pair Separated |
|---|---|
| `avg_closing_rate ≈ 0` + high `avg_distance_consistency` | following vs. fast_approach |
| `converging_sector_count = 3–4` + high `angle_spread` | surrounding vs. fast_approach |
| `threat_person_count` high | surrounding vs. following |
| `max_velocity_toward_target` spike | fast_approach vs. all others |

### Supervised vs. Unsupervised Decision

Justified and documented the choice of supervised approach over unsupervised alternatives:
- Unsupervised methods (Isolation Forest, LSTM Autoencoder) cannot distinguish threat *types*
- High false positive rate in real scenes for pure anomaly detection
- Hand-crafted features + RF consistently outperform deep learning at < 200 samples/class
- Recommended hybrid (Stage 1 unsupervised + Stage 2 supervised) as future extension

### Current Limitation

- No real labeled dataset yet — dataset collection and annotation is the **next active step**
- Model training and evaluation pending data acquisition

---

## Slide 8 — Benchmark & EDA Plan

### Expected Feature Distribution (Design Priors)

Based on scenario definitions, expected statistical behavior of key features per class:

| Feature | `normal` | `following` | `fast_approach` | `surrounding` |
|---|---|---|---|---|
| `threat_person_count` | 0–2 | 1 | 1–3 | 3–6 |
| `avg_closing_rate` | ~0 | ~0 | **HIGH** | MEDIUM |
| `angle_spread` (deg) | random | LOW (~20°) | LOW (~30°) | **HIGH (>90°)** |
| `avg_distance_consistency` | variable | **HIGH** | LOW | LOW |
| `max_closing_speed` (px/fr) | ~0 | ~0 | **HIGH (>5)** | MEDIUM |
| `converging_sector_count` | 0 | 1 | 1 | **3–4** |

### EDA Plan (On First Real Data Batch)

1. **Class balance check** — histogram of samples per class, target 1.5:1 (normal:threat)
2. **Feature correlation matrix** — identify redundant features before training
3. **Per-feature box plots** stratified by threat type — validate design priors above
4. **PCA / t-SNE 2D projection** of feature space — visual separation between classes
5. **Feature importance from trained RF** — confirm top discriminating features match design intent
6. **Confusion matrix** — identify which threat pairs are most confused (expected: following vs. fast_approach)

### Baseline Performance Targets

| Metric | Target (small dataset) | Note |
|---|---|---|
| Overall Accuracy | > 80% | Video-grouped CV |
| Threat Recall (binary) | > 85% | False negatives are costly |
| Normal Precision | > 90% | Control false alarms |
| Following F1 | > 75% | Hardest class (subtle signal) |

### Comparison Points

- **Heuristic baseline** (`src/following_logic.py` pairwise scorer) — rule-based threshold, no ML
- **Stage 1 only (Isolation Forest)** — anomaly detection without typing
- **Stage 2 RF / GBM / SVM** — full supervised classifier comparison

---

## Slide 9 — Applications

### Direct Applications

| Domain | Use Case |
|---|---|
| **Smart CCTV Systems** | Automated flagging of pre-attack patterns for security operators, reducing monitoring fatigue |
| **Campus / Transit Security** | Real-time alerts in high-footfall areas: university campuses, metro stations, airports |
| **Event Security** | Crowd monitoring at concerts, sports venues — detecting converging threat groups |
| **VIP Protection** | Detecting follower or encirclement behavior toward a designated target person |
| **Post-Event Forensics** | Rapid review of footage to identify threat onset timing for incident reports |

### Research Extensions

- **Novel threat discovery** — Stage 1 (Isolation Forest) can surface previously unobserved behavioral patterns that do not fit any known class
- **Transfer to other domains** — same trajectory feature pipeline applies to vehicle convoys, drone swarm detection, or robotic crowd navigation safety
- **Privacy-preserving surveillance** — works on anonymized skeleton/centroid tracks, no face recognition or biometric data required
- **Edge deployment** — lightweight feature extraction runs on embedded hardware (Jetson Nano, RPi + Coral) after tracking inference

### Societal Considerations

- System outputs an alert for human review — not autonomous enforcement
- No face recognition or identity matching; operates only on spatial movement patterns
- Explicit false positive rate control through the two-stage design
- Transparent feature set allows auditing of any flagged decision

---

## Slide 10 — Conclusion

### Summary

- We are building an **interpretable, lightweight threat detection pipeline** for surveillance video focused on **pre-attack spatial behavior patterns**
- The system targets three distinct scenarios: **following, fast approach, and surrounding**
- A **two-stage hybrid architecture** (unsupervised anomaly filter + supervised threat classifier) balances sensitivity with actionable output
- **12 hand-crafted group-level trajectory features** are designed to discriminate between all three threat types and normal pedestrian movement
- The full end-to-end pipeline — from raw video to threat alert — is **implemented and ready for data**

### Key Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Supervised over pure unsupervised | Threat type matters for actionable alerts; unsupervised cannot distinguish |
| Hand-crafted features over deep learning | Robust at < 200 samples/class; interpretable for debugging and audit |
| Video-grouped cross-validation | Prevents data leakage; realistic evaluation on unseen scenes |
| Hybrid two-stage architecture | Stage 1 catches novel threats; Stage 2 provides typed, actionable output |
| Both real + synthetic data | Real data for generalization; synthetic for controlled scenario coverage |

### Next Steps

1. **Dataset collection** — real controlled footage + synthetic trajectory generation
2. **Annotation** — window-level labels from folder structure (auto) + manual review
3. **First model training** — Random Forest baseline with video-grouped CV
4. **EDA and feature validation** — confirm design priors on actual data distributions
5. **Stage 1 integration** — train Isolation Forest on normal scenes; build full two-stage pipeline
6. **Evaluation** — confusion matrix, per-class F1, comparison against heuristic baseline

### Takeaway

> Automated pre-attack threat detection is tractable with modest data if the right relational, temporal features are extracted. The pipeline here is privacy-preserving, interpretable, and designed to scale from a research prototype to a deployable system.

---

*End of Presentation*

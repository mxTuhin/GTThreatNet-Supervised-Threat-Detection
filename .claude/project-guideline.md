# Project Guideline — Threat Detection from Surveillance Video

## 1. Project Goal

Detect **pre-attack threat scenarios** involving one target person and one or more threat persons. Scenarios include:

| Scenario | Description |
|---|---|
| `following` | 1 person trailing a target at consistent distance/speed |
| `fast_approach` | 1–3 persons rapidly closing distance toward a target, typically from behind |
| `surrounding` | 2+ persons converging from different directions to encircle a target |
| `normal` | Regular pedestrian movement, no threat |

This is a **supervised binary or multi-class classification** problem on trajectory feature windows extracted from tracked video.

---

## 2. Dataset Folder Structure (Recommended)

```
data/
├── raw_videos/
│   ├── normal/              ← regular pedestrian scenes
│   │   ├── street_001.mp4
│   │   └── park_001.mp4
│   └── threat/              ← anomaly scenes
│       ├── following/
│       │   └── follow_001.mp4
│       ├── surrounding/
│       │   └── surround_001.mp4
│       └── fast_approach/
│           └── approach_001.mp4
├── annotations/
│   └── video_labels.csv     ← video_name, label, threat_type, notes
├── derived/
│   ├── threat_windows.csv   ← extracted + labeled feature windows (ML input)
│   ├── labeled_pairs.csv    ← (legacy: pairwise following labels)
│   └── threat_model.pkl     ← trained classifier
└── outputs/
    ├── csv/                 ← per-video tracked outputs (track_video.py output)
    └── video/               ← annotated output videos
```

### Why separate subfolders?

- **Data leakage prevention**: ML splits must be at the video level, not frame level. If all clips are in one folder, it is easy to accidentally train/test on frames from the same scene.
- **Class balance visibility**: Immediately see how many normal vs threat samples you have.
- **Scalability**: When you add a new scenario type, add a new subfolder — no CSV schema changes.

### annotations/video_labels.csv format

```csv
video_name,label,threat_type,notes
street_001.mp4,0,normal,crowded plaza
follow_001.mp4,1,following,single follower on sidewalk
surround_001.mp4,1,surrounding,3-person encirclement at intersection
approach_001.mp4,1,fast_approach,2 persons run toward target from behind
```

---

## 3. ML Pipeline Overview

```
raw_video
    │
    ▼
src/track_video.py          ← YOLO + BoT-SORT → per-frame detections + tracks
    │
    ▼
outputs/csv/<video>.csv     ← frame_idx, track_id, x1,y1,x2,y2, cx,cy, direction...
    │
    ▼
src/threat_feature_extractor.py   ← sliding window → group-level threat features per target
    │
    ▼
data/derived/threat_windows.csv   ← feature rows, one per (video, window, target_id)
    │                               [manual/auto label added here]
    ▼
train_threat_model.py       ← train Random Forest / Gradient Boosting classifier
    │
    ▼
data/derived/threat_model.pkl
    │
    ▼
infer_threat.py             ← score new video → per-target threat predictions
```

---

## 4. Feature Design

### Old (pairwise, following-only)
6 features computed for a (leader, follower) pair:
`num_points, direction_similarity, avg_distance, distance_variance, speed_similarity, behind_ratio`

### New (group-level, threat detection)
For each **target person T** in a time window, the **threat group** = all other persons within a proximity radius. Features:

| Feature | Description |
|---|---|
| `threat_person_count` | Number of persons within `PROXIMITY_RADIUS` pixels of T |
| `num_closing_persons` | Count of persons whose distance to T is actively decreasing |
| `max_closing_speed` | Highest rate of distance decrease (px/frame) toward T |
| `avg_closing_rate` | Mean closing rate across all nearby persons (positive = closing) |
| `angle_spread` | Std dev of angles from T to each nearby person (high → surrounding) |
| `min_distance` | Minimum distance from T to any nearby person in window |
| `avg_speed_of_group` | Average movement speed of nearby persons |
| `behind_person_count` | Persons in the rear half-plane of T's movement direction |
| `max_velocity_toward_target` | Highest velocity component directed at T |
| `target_speed` | T's own movement speed (slow target = more vulnerable) |

These 10 features replace the 6 pairwise features for threat scenarios.

---

## 5. Labeling Strategy (Small Dataset)

With a small dataset:

1. **Run** `src/track_video.py` on each raw video → produces `outputs/csv/<video>.csv`
2. **Run** `src/threat_feature_extractor.py` on each tracked CSV → produces rows in `threat_windows.csv`
3. **Label** each row: `0=normal`, `1=threat` (binary) or use `threat_type` for multi-class
4. **Train** with `train_threat_model.py`

### Labeling tips for small datasets
- Aim for at least **30–50 threat windows** per threat type
- **Augment** by extracting multiple windows from the same video with different strides
- Keep normal windows slightly more than threat windows (1.5:1 ratio is fine)
- **Do NOT mix train/test from the same video** — split at video level

### Train/val/test split
```
normal videos:  → 70% train / 15% val / 15% test
threat videos:  → 70% train / 15% val / 15% test
```

Keep a `data/splits.csv` with video names and split assignments.

---

## 6. Model Choice for Small Datasets

| Model | Why |
|---|---|
| **Random Forest** | Robust to small data, no scaling needed, interpretable feature importances |
| **Gradient Boosting (XGBoost/LightGBM)** | Slightly better performance, still small-data friendly |
| **SVM (RBF kernel)** | Good with few features + small data, needs feature scaling |

**Avoid**: deep learning (LSTM, transformer) until you have 500+ labeled windows per class.

**Recommended**: Start with `RandomForestClassifier` (already used), add `class_weight='balanced'` for class imbalance.

---

## 7. Key Script Reference

| Script | Role |
|---|---|
| `src/track_video.py` | Detect + track persons in a video, output CSV |
| `src/threat_feature_extractor.py` | Extract group-level threat features per target per window |
| `src/following_logic.py` | (Legacy) heuristic pairwise following scorer — kept for reference |
| `train_threat_model.py` | Train threat classifier from labeled feature windows |
| `infer_threat.py` | Apply trained model to new tracked CSV, output threat annotations |
| `src/utilities/trajectory_utils.py` | Shared trajectory math utilities |

---

## 8. Naming Conventions

- Raw videos: `<scene_descriptor>_<NNN>.mp4` (e.g., `street_corner_001.mp4`)
- Tracked CSVs: same stem as video (e.g., `street_corner_001.csv`)
- Annotation labels: integer `0=normal`, `1=threat`
- Threat types: `normal`, `following`, `surrounding`, `fast_approach`

---

## 9. Hardware Notes

- GPU: RTX 3060, CUDA 13.2 — YOLO tracking uses GPU automatically
- sklearn models are CPU-only (fast enough for tabular features)
- Recommended tracker: `custom_botsort.yaml` (ReID + 5s buffer)

---

## 10. What NOT to Do

- Do not combine all videos into one flat folder — makes video-level splits impossible
- Do not split train/test at the frame or window level from the same video — causes data leakage
- Do not use deep learning until you have 500+ labeled windows per class
- Do not label individual frames — label at the **window** level (30-frame clips)
- Do not use pairwise features alone for surrounding detection — group features are required

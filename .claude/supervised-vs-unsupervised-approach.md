# Supervised vs Unsupervised Approach — Threat Detection

## Current Feature Coverage (12 features, 3 scenarios)

Features are **not surrounding-only**. They are group-level features computed per target person T
across all nearby persons in a 30-frame sliding window. Each feature fires differently per scenario.

### Feature → Scenario Signal Map

| Feature | `surrounding` | `fast_approach` | `following` |
|---|---|---|---|
| `threat_person_count` | HIGH (many) | LOW (1–3) | LOW (1) |
| `min_distance` | SMALL (crowded) | DECREASING fast | STABLE, moderate |
| `num_closing_persons` | HIGH | HIGH | ~0 (consistent distance) |
| `max_closing_speed` | MEDIUM | **HIGH** | LOW |
| `avg_closing_rate` | POSITIVE | **STRONGLY POSITIVE** | ~0 |
| `max_velocity_toward_target` | MEDIUM | **HIGH** | LOW |
| `angle_spread` | **HIGH** (all directions) | LOW (all from behind/side) | LOW (single direction) |
| `converging_sector_count` | **3–4 sectors** | 1 sector | 1 sector |
| `behind_person_count` | MIXED | HIGH (from behind) | HIGH (1 person behind) |
| `avg_distance_consistency` | LOW (erratic) | LOW (rapidly changing) | **HIGH** (stable trailing) |
| `avg_speed_of_group` | MEDIUM | **HIGH** | SIMILAR to target |
| `target_speed` | ANY | ANY | SIMILAR to threat person |

### Why some features exist

- `converging_sector_count` — `angle_spread` alone is weak for surrounding. Two people
  standing on opposite sides of a target can score high on angle_spread without actually
  converging. `converging_sector_count` only counts sectors where someone is actively
  closing in, making it a much stronger surrounding signal.

- `avg_distance_consistency` — `following` produces a person who maintains a consistent
  distance (not closing, not leaving). The old pairwise `distance_variance` feature captured
  this. `avg_distance_consistency` is the equivalent for the group-level pipeline.
  Without it, following and fast_approach both show high `behind_person_count` and are
  hard to separate.

### Key discriminators between easily-confused pairs

**following vs fast_approach** (both: person from behind)
- `avg_closing_rate ≈ 0` + high `avg_distance_consistency` → following
- `avg_closing_rate >> 0` + high `max_velocity_toward_target` → fast_approach

**surrounding vs fast_approach** (both: persons closing in)
- `converging_sector_count = 3–4` + high `angle_spread` → surrounding
- `converging_sector_count = 1` + high `max_closing_speed` → fast_approach

---

## Supervised vs Unsupervised — Full Comparison

### Supervised (current approach)

- Labels required: yes (auto-populated from folder structure, manual review optional)
- Feature engineering: hand-crafted 12 features
- Models: Random Forest, Gradient Boosting, SVM
- Output: exact threat type (normal / following / surrounding / fast_approach)
- Minimum data: ~30–50 labeled windows per class is workable
- Interpretable: yes — feature importances show what the model learned

**Works well for this project because:**
- Dataset is small (hand-crafted features outperform deep models under ~500 samples/class)
- You need to know the threat *type*, not just that something is unusual
- Video-grouped cross-validation prevents leakage even with few videos

### Unsupervised Options

#### Isolation Forest / One-Class SVM
- Train on normal scenes only. No threat labels needed.
- Anything the model scores as "out of distribution" = potential anomaly.
- **Limitation**: cannot tell you *what kind* of threat it is.
- **Limitation**: high false positives (unusual-but-harmless events also trigger it).
- **Good for**: anomaly detection when you have zero labeled threat data yet.

#### LSTM Autoencoder
- Learns to reconstruct "normal" trajectory sequences.
- High reconstruction error on a new sequence = anomaly.
- **Limitation**: needs hundreds of normal sequences to learn a reliable baseline.
- **Limitation**: 30-frame windows are short — not much temporal structure to exploit.
- **Good for**: when threat patterns are rare and hard to enumerate in advance.

#### Clustering (K-Means, GMM, DBSCAN)
- Groups motion patterns. Outlier clusters = suspicious.
- **Good for**: EDA — discovering what patterns exist in your data before labeling.
- **Not good for**: production detector (clusters don't align cleanly with threat types).

### Why Unsupervised Alone Won't Replace the Current Approach

1. You cannot distinguish threat types. An Isolation Forest says "anomalous" but cannot
   say "surrounding vs fast approach". For actionable alerts, the distinction matters.

2. Normal-but-unusual ≠ threat. A person running, a crowd for an event, someone stopping
   suddenly — all score as anomalies. High false positive rate in real scenes.

3. Small dataset cuts both ways. LSTM autoencoders need hundreds of "normal" sequences
   to learn a reliable reconstruction baseline. You don't have that, and neither does
   the supervised path — but at least hand-crafted features work with 30–50 samples.

---

## Recommended Hybrid Architecture

For this project, a two-stage pipeline makes the most sense:

```
Stage 1 — Unsupervised (Isolation Forest, trained on normal scenes only)
    ↓
    Flags: "something unusual here"
    Can be trained with ZERO labeled threat data

Stage 2 — Supervised (Random Forest, trained on labeled windows)
    ↓
    Classifies: following / surrounding / fast_approach / normal
    Only runs when Stage 1 triggers
```

**Benefits:**
- Stage 1 catches anomalies you haven't seen before (novel threats)
- Stage 2 tells you which kind of threat it is (actionable output)
- Stage 1 needs only normal videos — no labeling required for that half
- False positives from Stage 1 get filtered/typed by Stage 2

**When to use this hybrid**: once you have enough normal video footage to train Stage 1.
For now, with very little data, the supervised-only path is the right starting point.

---

## Auto Feature Extraction (Deep Learning) — When Does It Help?

Deep models (LSTM → embedding → classifier, or CNN on heatmap frames) learn features
automatically from raw trajectories without manual engineering.

| Condition | Recommendation |
|---|---|
| < 200 labeled sequences per class | Hand-crafted features + RF (current approach) |
| 200–500 labeled sequences per class | Try GBM on hand-crafted features; attempt shallow LSTM |
| 500+ labeled sequences per class | LSTM autoencoder (Stage 1) + deep classifier (Stage 2) |

**For this project now**: hand-crafted features + RF is correct. The 12 features are also
interpretable — you know *why* a specific window was flagged, which matters for debugging
and for explaining results.

---

## Model Recommendations by Dataset Size

| Model | Min samples/class | Notes |
|---|---|---|
| Random Forest (`class_weight='balanced'`) | ~30 | Current default, robust to small data |
| Gradient Boosting (XGBoost / LightGBM) | ~50 | Slightly better than RF on tabular features |
| SVM (RBF kernel) | ~30 | Good with few features, needs feature scaling |
| LSTM Classifier | ~500 | Raw trajectories as input, no feature engineering |
| LSTM Autoencoder + classifier | ~500 normal + 200/class | Two-stage, best long-term option |
| Transformer | ~1000+ | Overkill for this problem at any realistic scale |

**Avoid deep learning** until you have 500+ labeled windows per class. Below that,
interpretable models on hand-crafted features consistently win on small surveillance datasets.

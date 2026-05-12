"""
config.py — Central configuration for the threat detection pipeline.
All tuneable constants live here. Import from here instead of hardcoding.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_VIDEOS_DIR  = DATA_DIR / "raw_videos"   # original video files (input to prepare_dataset.py)
FRAMES_DIR      = DATA_DIR / "frames"       # extracted image frames (input to pipeline.py)
DERIVED_DIR     = DATA_DIR / "derived"
OUTPUTS_DIR     = DATA_DIR / "outputs"
CSV_OUT_DIR     = OUTPUTS_DIR / "csv"
VIDEO_OUT_DIR   = OUTPUTS_DIR / "video"
EDA_OUT_DIR     = OUTPUTS_DIR / "eda"
SPLITS_CSV      = DATA_DIR / "splits.csv"

THREAT_WINDOWS_CSV   = DERIVED_DIR / "threat_windows.csv"
SEQUENCE_WINDOWS_NPZ = DERIVED_DIR / "sequence_windows.npz"    # BiLSTM sequences
MODELS_DIR           = DERIVED_DIR / "models"
LABEL_ENCODER_PATH   = DERIVED_DIR / "threat_label_encoder.pkl"

# ── Tracking ──────────────────────────────────────────────────────────────────
YOLO_MODEL      = str(ROOT / "src" / "yolov8m.pt")
TRACKER_CONFIG  = str(ROOT / "src" / "custom_botsort.yaml")
CONFIDENCE_THR  = 0.35
PERSON_CLASS_ID = 0                 # COCO class for person

# ── Feature extraction ────────────────────────────────────────────────────────
WINDOW_SIZE       = 30              # frames per sliding window
WINDOW_STRIDE     = 10             # stride between consecutive windows
MIN_TRACK_POINTS  = 10             # min detections per track inside a window
PROXIMITY_RADIUS  = 300.0          # px — max dist to be counted as "nearby"
MIN_NEARBY        = 1              # min nearby persons to emit a row
DIRECTION_SMOOTH  = 12             # frames used for direction smoothing

# ── Feature names (must match feature_extractor.py output) ───────────────────
WINDOW_FEATURES = [
    # proximity
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
    "encirclement_ratio",
    # following-specific
    "behind_person_count",
    "avg_distance_consistency",
    "approach_persistence",
    # group motion
    "avg_speed_of_group",
    "group_speed_std",
    "target_speed",
    # temporal / physics
    "max_closing_acceleration",
    "distance_trend_slope",
    "group_centroid_closing_rate",
    "target_direction_changes",
    "synchronized_closing_ratio",
]

# Per-frame features fed into BiLSTM (simpler, instantaneous)
FRAME_FEATURES = [
    "n_nearby",
    "min_distance_frame",
    "mean_distance_frame",
    "max_velocity_toward_frame",
    "angle_to_nearest",
    "behind_count_frame",
    "target_vx",
    "target_vy",
]

# ── Label map ─────────────────────────────────────────────────────────────────
THREAT_TYPE_MAP = {
    "normal":        0,
    "following":     1,
    "surrounding":   2,
    "fast_approach": 3,
}
THREAT_TYPE_NAMES = {v: k for k, v in THREAT_TYPE_MAP.items()}
CLASS_NAMES = list(THREAT_TYPE_MAP.keys())      # ["normal", "following", ...]

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── Training ──────────────────────────────────────────────────────────────────
CV_FOLDS    = 5
N_JOBS      = -1            # -1 = use all CPU cores

# RF hyperparameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH    = 10

# XGBoost hyperparameters
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH    = 6
XGB_LR           = 0.05

# BiLSTM hyperparameters
BILSTM_HIDDEN    = 128
BILSTM_LAYERS    = 2
BILSTM_DROPOUT   = 0.3
BILSTM_EPOCHS    = 50
BILSTM_LR        = 1e-3
BILSTM_BATCH     = 32
BILSTM_SEQ_LEN   = WINDOW_SIZE    # = 30 frames

# ── Graph / STGAT ─────────────────────────────────────────────────────────────
GRAPH_N_MAX       = 10       # target node + up to 9 nearby persons (padded if fewer)
GRAPH_NODE_DIM    = 4        # node feature dim: [cx, cy, vx, vy]
GRAPH_SEQ_NPZ     = DERIVED_DIR / "graph_sequences.npz"

STGAT_GAT_HIDDEN  = 32      # output dim per attention head
STGAT_GAT_HEADS   = 4       # heads → total GAT output = 128
STGAT_GRU_HIDDEN  = 128
STGAT_DROPOUT     = 0.3
STGAT_EPOCHS      = 25
STGAT_PATIENCE    = 10
STGAT_LR          = 1e-3
STGAT_BATCH       = 32

# ── Inference ─────────────────────────────────────────────────────────────────
THREAT_SCORE_THRESHOLD = 0.5      # above this = alert

# ── Folder-to-label mapping ───────────────────────────────────────────────────
# Maps a folder name found under raw_videos/ to (binary_label, threat_type)
FOLDER_LABEL_MAP = {
    "normal":       (0, "normal"),
    "following":    (1, "following"),
    "surrounding":  (1, "surrounding"),
    "fast_approach": (1, "fast_approach"),
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

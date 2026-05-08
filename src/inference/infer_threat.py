"""
infer_threat.py
================
Apply a trained threat model to a newly tracked CSV (or raw video).
Outputs a scored CSV and optionally an annotated video.

Usage:
    # From tracked CSV:
    python src/inference/infer_threat.py \
        --input data/outputs/csv/myvideo.csv \
        --video-name myvideo

    # From raw video (runs tracking first):
    python src/inference/infer_threat.py \
        --raw-video path/to/video.mp4

    # With annotated video output:
    python src/inference/infer_threat.py \
        --input data/outputs/csv/myvideo.csv \
        --video path/to/original.mp4 \
        --output-video data/outputs/video/myvideo_threat.mp4
"""

import sys
import os
import csv
import argparse
import subprocess
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    MODELS_DIR, LABEL_ENCODER_PATH, CSV_OUT_DIR,
    WINDOW_FEATURES, WINDOW_SIZE, WINDOW_STRIDE, PROXIMITY_RADIUS,
    THREAT_SCORE_THRESHOLD, CLASS_NAMES, THREAT_TYPE_MAP,
)
from src.data.feature_extractor import extract_windows


def load_model(model_type: str = "rf"):
    """Load the best available model. Preference: rf > xgb > bilstm."""
    model_dir = Path(MODELS_DIR)
    candidates = {
        "rf":     model_dir / "rf_model.pkl",
        "xgb":    model_dir / "xgb_model.pkl",
        "bilstm": model_dir / "bilstm_model.pt",
    }

    if model_type != "auto":
        path = candidates.get(model_type)
        if path and path.exists():
            return model_type, _load_sklearn(path) if model_type != "bilstm" else None, path
        print(f"[WARN] {model_type} model not found at {path}")

    # Auto: pick first available
    for mtype, mpath in candidates.items():
        if mpath.exists():
            print(f"Using model: {mtype} ({mpath})")
            return mtype, _load_sklearn(mpath) if mtype != "bilstm" else None, mpath

    raise FileNotFoundError(f"No trained model found in {model_dir}. Run train_models.py first.")


def _load_sklearn(path: Path):
    return joblib.load(path)


def predict_windows(rows: list[dict], model_type: str, model, model_path: Path) -> list[dict]:
    """Add prediction columns to each row."""
    if not rows:
        return rows

    le = joblib.load(LABEL_ENCODER_PATH) if Path(LABEL_ENCODER_PATH).exists() else None

    feats = [f for f in WINDOW_FEATURES if f in rows[0]]
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], dtype=np.float32)

    if model_type in ("rf", "xgb"):
        preds  = model.predict(X)
        probas = model.predict_proba(X)
        for i, row in enumerate(rows):
            row["pred_class_idx"]   = int(preds[i])
            row["pred_class_name"]  = le.classes_[preds[i]] if le else str(preds[i])
            row["pred_confidence"]  = float(probas[i].max())
            threat_idx = [j for j, c in enumerate(le.classes_ if le else CLASS_NAMES)
                          if c != "normal"]
            row["threat_score"] = float(probas[i][threat_idx].sum()) if threat_idx else row["pred_confidence"]

    elif model_type == "bilstm":
        # BiLSTM requires sequence format — skip for window-mode inference
        from src.models.bilstm_model import predict_bilstm
        # Use a synthetic repeated sequence from window features as proxy
        # (proper BiLSTM inference needs raw per-frame data)
        preds, probas = predict_bilstm(
            np.repeat(X[:, np.newaxis, :], 30, axis=1).astype(np.float32),
            str(model_path),
        )
        for i, row in enumerate(rows):
            row["pred_class_idx"]   = int(preds[i])
            row["pred_class_name"]  = le.classes_[preds[i]] if le else str(preds[i])
            row["pred_confidence"]  = float(probas[i].max())
            threat_idx = [j for j, c in enumerate(le.classes_ if le else CLASS_NAMES)
                          if c != "normal"]
            row["threat_score"] = float(probas[i][threat_idx].sum()) if threat_idx else row["pred_confidence"]

    return rows


def write_predictions(rows: list[dict], out_path: str):
    if not rows:
        print("No predictions to write.")
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["video_name", "start_frame", "end_frame", "target_id"]
    pred_fields = ["pred_class_idx", "pred_class_name", "pred_confidence", "threat_score"]
    feat_fields = [f for f in WINDOW_FEATURES if f in rows[0]]
    all_fields  = base_fields + feat_fields + pred_fields
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved predictions → {out_path}")


def annotate_video(
    video_path: str,
    rows: list[dict],
    output_path: str,
    threshold: float = THREAT_SCORE_THRESHOLD,
):
    """Overlay threat alerts on video frames."""
    try:
        import cv2
    except ImportError:
        print("[SKIP] opencv not available for video annotation")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Build per-frame threat info
    frame_threats: dict[int, list[dict]] = {}
    for row in rows:
        for f in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            frame_threats.setdefault(f, []).append(row)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        threats = frame_threats.get(frame_idx, [])
        if threats:
            max_threat = max(r.get("threat_score", 0.0) for r in threats)
            best = max(threats, key=lambda r: r.get("threat_score", 0.0))
            label_text = best.get("pred_class_name", "normal")
            if max_threat >= threshold and label_text != "normal":
                cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 200), 6)
                cv2.putText(frame, f"THREAT: {label_text.upper()} ({max_threat:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"NORMAL ({max_threat:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video → {output_path}")


def run_tracking(video_path: str, out_csv: str) -> bool:
    """Run track_video.py on a raw video and return True on success."""
    cmd = [
        sys.executable, "src/tracking/track_video.py",
        "--input", video_path,
        "--out-csv", out_csv,
        "--no-display",
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Inference: detect threats in video")
    parser.add_argument("--input",       help="Tracked CSV (track_video.py output)")
    parser.add_argument("--raw-video",   help="Raw video path (runs tracking automatically)")
    parser.add_argument("--video-name",  help="Label for this video (default: inferred from path)")
    parser.add_argument("--model",       default="auto", choices=["auto", "rf", "xgb", "bilstm"])
    parser.add_argument("--output",      default="data/derived/predictions.csv")
    parser.add_argument("--video",       help="Original video path for annotation overlay")
    parser.add_argument("--output-video", help="Annotated output video path")
    parser.add_argument("--window-size", type=int,   default=WINDOW_SIZE)
    parser.add_argument("--stride",      type=int,   default=WINDOW_STRIDE)
    parser.add_argument("--proximity",   type=float, default=PROXIMITY_RADIUS)
    parser.add_argument("--threshold",   type=float, default=THREAT_SCORE_THRESHOLD)
    args = parser.parse_args()

    # ── Determine tracked CSV ──────────────────────────────────────────────────
    if args.raw_video and not args.input:
        video_stem = Path(args.raw_video).stem
        csv_out = str(CSV_OUT_DIR / f"{video_stem}.csv")
        print(f"Running tracking on {args.raw_video} ...")
        if not run_tracking(args.raw_video, csv_out):
            print("[ERROR] Tracking failed.")
            return
        args.input = csv_out
        if not args.video:
            args.video = args.raw_video

    if not args.input:
        parser.error("Provide --input (tracked CSV) or --raw-video")

    video_name = args.video_name or Path(args.input).stem
    print(f"Extracting features from: {args.input}")
    rows = extract_windows(
        csv_path     = args.input,
        video_name   = video_name,
        window_size  = args.window_size,
        stride       = args.stride,
        proximity    = args.proximity,
    )
    print(f"Extracted {len(rows)} windows")

    model_type, model, model_path = load_model(args.model)
    rows = predict_windows(rows, model_type, model, model_path)
    write_predictions(rows, args.output)

    # Summary
    if rows:
        threat_rows = [r for r in rows if r.get("pred_class_name", "normal") != "normal"]
        print(f"\nThreat windows: {len(threat_rows)} / {len(rows)} "
              f"({100*len(threat_rows)/len(rows):.1f}%)")
        by_type: dict[str, int] = {}
        for r in threat_rows:
            by_type[r.get("pred_class_name", "?")] = by_type.get(r.get("pred_class_name","?"), 0) + 1
        for ttype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {ttype}: {count}")

    if args.output_video and args.video:
        annotate_video(args.video, rows, args.output_video, args.threshold)


if __name__ == "__main__":
    main()

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


def load_model(model_type: str = "auto"):
    """
    Load the best available trained model.
    Priority for auto: stgat > xgb > bilstm
    """
    model_dir = Path(MODELS_DIR)
    candidates = {
        "xgb":    model_dir / "xgb_model.pkl",
        "bilstm": model_dir / "bilstm_model.pt",
        "stgat":  model_dir / "stgat_model.pt",
    }

    if model_type != "auto":
        path = candidates.get(model_type)
        if path and path.exists():
            model = joblib.load(path) if model_type == "xgb" else None
            return model_type, model, path
        print(f"[WARN] {model_type} model not found at {path}")

    # Auto: prefer stgat > xgb > bilstm
    for mtype in ("stgat", "xgb", "bilstm"):
        mpath = candidates[mtype]
        if mpath.exists():
            print(f"Using model: {mtype} ({mpath})")
            model = joblib.load(mpath) if mtype == "xgb" else None
            return mtype, model, mpath

    raise FileNotFoundError(
        f"No trained model found in {model_dir}. Run train_models.py first."
    )


# ── Window-based prediction (XGBoost / BiLSTM) ───────────────────────────────

def predict_windows(rows: list[dict], model_type: str, model, model_path: Path) -> list[dict]:
    """Add prediction columns to window-feature rows."""
    if not rows:
        return rows

    le = joblib.load(LABEL_ENCODER_PATH) if Path(LABEL_ENCODER_PATH).exists() else None
    feats = [f for f in WINDOW_FEATURES if f in rows[0]]
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], dtype=np.float32)

    if model_type == "xgb":
        preds  = model.predict(X)
        probas = model.predict_proba(X)
        for i, row in enumerate(rows):
            _fill_pred(row, int(preds[i]), probas[i], le)

    elif model_type == "bilstm":
        from src.models.bilstm_model import predict_bilstm
        # Repeat each window-feature vector across T timesteps as proxy input
        preds, probas = predict_bilstm(
            np.repeat(X[:, np.newaxis, :], 30, axis=1).astype(np.float32),
            str(model_path),
        )
        for i, row in enumerate(rows):
            _fill_pred(row, int(preds[i]), probas[i], le)

    return rows


def _fill_pred(row: dict, pred_idx: int, prob_vec: np.ndarray, le):
    row["pred_class_idx"]  = pred_idx
    row["pred_class_name"] = le.classes_[pred_idx] if le else str(pred_idx)
    row["pred_confidence"] = float(prob_vec.max())
    threat_idx = [j for j, c in enumerate(le.classes_ if le else CLASS_NAMES)
                  if c != "normal"]
    row["threat_score"] = float(prob_vec[threat_idx].sum()) if threat_idx else float(prob_vec.max())


# ── Graph-based prediction (STGAT) ───────────────────────────────────────────

def predict_stgat_windows(
    csv_path:   str,
    video_name: str,
    model_path: Path,
    window_size: int   = WINDOW_SIZE,
    stride:      int   = WINDOW_STRIDE,
    proximity:   float = PROXIMITY_RADIUS,
) -> list[dict]:
    """
    Run STGAT inference on a tracked CSV.
    Returns list of dicts with video_name, start_frame, end_frame, target_id,
    pred_class_idx, pred_class_name, pred_confidence, threat_score.
    """
    from src.data.feature_extractor import extract_graph_sequences, load_tracked_csv
    from src.models.stgat_model import predict_stgat

    frame_map  = load_tracked_csv(csv_path)
    all_frames = sorted(frame_map.keys())

    X, A, valid, _, names = extract_graph_sequences(
        csv_path    = csv_path,
        video_name  = video_name,
        window_size = window_size,
        stride      = stride,
        proximity   = proximity,
    )

    if len(X) == 0:
        print("[WARN] No graph sequences extracted — check tracking CSV and proximity setting")
        return []

    le = joblib.load(LABEL_ENCODER_PATH) if Path(LABEL_ENCODER_PATH).exists() else None
    labels, probs, _ = predict_stgat(X, A, valid, str(model_path))

    rows = []
    for idx, name in enumerate(names):
        # name = "videoname_w{w_idx}_t{target_id}"
        t_parts   = name.rsplit("_t", 1)
        target_id = int(t_parts[1]) if len(t_parts) == 2 and t_parts[1].isdigit() else -1
        w_parts   = t_parts[0].rsplit("_w", 1)
        w_idx     = int(w_parts[1]) if len(w_parts) == 2 and w_parts[1].isdigit() else idx

        f_start_i  = w_idx * stride
        f_end_i    = min(f_start_i + window_size - 1, len(all_frames) - 1)
        start_frame = all_frames[f_start_i] if f_start_i < len(all_frames) else w_idx
        end_frame   = all_frames[f_end_i]   if f_end_i   < len(all_frames) else start_frame

        prob_vec  = probs[idx]
        pred_idx  = int(labels[idx])
        pred_name = le.classes_[pred_idx] if le else str(pred_idx)
        threat_idx = [j for j, c in enumerate(le.classes_ if le else CLASS_NAMES)
                      if c != "normal"]
        threat_score = float(prob_vec[threat_idx].sum()) if threat_idx else float(prob_vec.max())

        rows.append({
            "video_name":      video_name,
            "start_frame":     start_frame,
            "end_frame":       end_frame,
            "target_id":       target_id,
            "pred_class_idx":  pred_idx,
            "pred_class_name": pred_name,
            "pred_confidence": float(prob_vec.max()),
            "threat_score":    threat_score,
        })

    return rows


# ── Output ────────────────────────────────────────────────────────────────────

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
    cmd = [
        sys.executable, "src/tracking/track_video.py",
        "--input", video_path,
        "--out-csv", out_csv,
        "--no-display",
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference: detect threats in video")
    parser.add_argument("--input",        help="Tracked CSV (track_video.py output)")
    parser.add_argument("--raw-video",    help="Raw video path (runs tracking automatically)")
    parser.add_argument("--video-name",   help="Label for this video (default: inferred from path)")
    parser.add_argument("--model",        default="auto",
                        choices=["auto", "xgb", "bilstm", "stgat"])
    parser.add_argument("--output",       default="data/derived/predictions.csv")
    parser.add_argument("--video",        help="Original video path for annotation overlay")
    parser.add_argument("--output-video", help="Annotated output video path")
    parser.add_argument("--window-size",  type=int,   default=WINDOW_SIZE)
    parser.add_argument("--stride",       type=int,   default=WINDOW_STRIDE)
    parser.add_argument("--proximity",    type=float, default=PROXIMITY_RADIUS)
    parser.add_argument("--threshold",    type=float, default=THREAT_SCORE_THRESHOLD)
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

    video_name   = args.video_name or Path(args.input).stem
    model_type, model, model_path = load_model(args.model)

    # ── Run inference ──────────────────────────────────────────────────────────
    if model_type == "stgat":
        print(f"Extracting graph sequences from: {args.input}")
        rows = predict_stgat_windows(
            csv_path    = args.input,
            video_name  = video_name,
            model_path  = model_path,
            window_size = args.window_size,
            stride      = args.stride,
            proximity   = args.proximity,
        )
    else:
        print(f"Extracting features from: {args.input}")
        rows = extract_windows(
            csv_path    = args.input,
            video_name  = video_name,
            window_size = args.window_size,
            stride      = args.stride,
            proximity   = args.proximity,
        )
        rows = predict_windows(rows, model_type, model, model_path)

    print(f"Scored {len(rows)} windows")
    write_predictions(rows, args.output)

    # ── Summary ────────────────────────────────────────────────────────────────
    if rows:
        threat_rows = [r for r in rows if r.get("pred_class_name", "normal") != "normal"]
        print(f"\nThreat windows: {len(threat_rows)} / {len(rows)} "
              f"({100*len(threat_rows)/len(rows):.1f}%)")
        by_type: dict[str, int] = {}
        for r in threat_rows:
            t = r.get("pred_class_name", "?")
            by_type[t] = by_type.get(t, 0) + 1
        for ttype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {ttype}: {count}")

    if args.output_video and args.video:
        annotate_video(args.video, rows, args.output_video, args.threshold)


if __name__ == "__main__":
    main()

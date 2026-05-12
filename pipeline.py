
"""
pipeline.py — Master pipeline caller for threat detection.
Calls individual stage scripts via subprocess or their main() entry points.
Does NOT contain business logic — all logic lives in src/ modules.

Usage:
    python pipeline.py --stage all                         # full pipeline
    python pipeline.py --stage track                       # YOLO tracking on all raw videos
    python pipeline.py --stage features                    # extract window + sequence features
    python pipeline.py --stage split                       # compute train/val/test video splits
    python pipeline.py --stage validate                    # data quality checks
    python pipeline.py --stage eda-pre                     # pre-training EDA
    python pipeline.py --stage train --model rf            # train RF (or xgb / bilstm / all)
    python pipeline.py --stage evaluate --model rf         # evaluate on test split
    python pipeline.py --stage eda-post --model rf         # post-training EDA
    python pipeline.py --stage explain --model rf          # SHAP / attention xAI
    python pipeline.py --stage infer --raw-video vid.mp4   # inference on new video
    python pipeline.py --stage batch-extract               # parallel feature extraction (joblib)
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], step_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  STAGE: {step_name}")
    print(f"  CMD:   {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  [{status}] {step_name} finished in {elapsed:.1f}s")
    return result.returncode


def _py(script: str, *extra_args: str) -> list[str]:
    return [sys.executable, script, *extra_args]


# ── Stage implementations ─────────────────────────────────────────────────────

def stage_prepare(args):
    """Extract frames from raw_videos/ into frames/ (skips already-done clips)."""
    cmd = _py("prepare_dataset.py")
    if getattr(args, "fps", None):
        cmd += ["--fps", str(args.fps)]
    if getattr(args, "resize", None):
        cmd += ["--resize"] + [str(v) for v in args.resize]
    return _run(cmd, "Extract frames from raw videos")


def stage_split(args):
    # No --raw-dir: dataset_loader auto-detects data/frames/ first, then data/raw_videos/
    return _run(_py("src/data/dataset_loader.py", "--out", "data/splits.csv"),
                "Discover frame dirs / videos + assign train/val/test splits")


def stage_track(args):
    """Track all raw videos found in splits.csv using joblib parallel."""
    from config import RAW_VIDEOS_DIR, CSV_OUT_DIR, VIDEO_OUT_DIR, VIDEO_EXTENSIONS
    import joblib, os

    splits_csv = Path("data/splits.csv")
    if not splits_csv.exists():
        print("[WARN] splits.csv not found — run --stage split first")
        return 1

    import csv
    with open(splits_csv) as f:
        records = list(csv.DictReader(f))

    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _track_one(rec):
        vpath = Path(rec["video_path"])
        if not vpath.exists():
            print(f"  [SKIP] Video not found: {vpath}")
            return
        stem     = vpath.stem
        out_csv  = str(CSV_OUT_DIR / f"{stem}.csv")
        out_vid  = str(VIDEO_OUT_DIR / f"{stem}_tracked.mp4")
        out_traj = str(VIDEO_OUT_DIR / f"{stem}_trajectory.mp4")
        out_sum  = str(CSV_OUT_DIR / f"{stem}_summary.csv")
        if Path(out_csv).exists():
            print(f"  [SKIP] Already tracked: {stem}")
            return
        cmd = _py("src/tracking/track_video.py",
                  "--input",       str(vpath),
                  "--out-csv",     out_csv,
                  "--out-video",   out_vid,
                  "--out-traj",    out_traj,
                  "--out-summary", out_sum,
                  "--no-display")
        subprocess.run(cmd)

    # Tracking is GPU-bound — run sequentially to avoid OOM
    for rec in records:
        _track_one(rec)
    return 0


def stage_augment(args):
    """
    Augment training-split tracked CSVs with geometric transforms.
    Writes augmented CSVs alongside originals and appends them to splits.csv
    so the features stage picks them up automatically.

    Only augments rows with split == 'train' — val/test are kept clean.
    """
    from config import CSV_OUT_DIR
    import csv as csv_mod

    splits_csv = Path("data/splits.csv")
    if not splits_csv.exists():
        print("[ERROR] splits.csv not found — run --stage split first")
        return 1

    aug_dir = CSV_OUT_DIR / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    augmentations = getattr(args, "augmentations", None) or ["flip", "rotate", "scale"]

    with open(splits_csv) as f:
        records = list(csv_mod.DictReader(f))

    fieldnames = list(records[0].keys()) if records else []
    new_records = []

    from src.data.augmentation import augment_csv

    for rec in records:
        if rec.get("split") != "train":
            continue
        stem     = Path(rec["video_path"]).stem
        src_csv  = str(CSV_OUT_DIR / f"{stem}.csv")
        if not Path(src_csv).exists():
            print(f"  [SKIP] No tracked CSV for {stem}")
            continue

        created = augment_csv(src_csv, str(aug_dir), augmentations)
        for aug_path in created:
            aug_stem = Path(aug_path).stem
            # Check if this augmented entry already exists in splits.csv
            if any(r.get("video_name") == aug_stem for r in records + new_records):
                print(f"  [SKIP] Already in splits.csv: {aug_stem}")
                continue
            new_rec = dict(rec)
            new_rec["video_name"] = aug_stem
            new_rec["video_path"] = str(aug_path)
            new_records.append(new_rec)

    if new_records:
        with open(splits_csv, "a", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerows(new_records)
        print(f"\nAdded {len(new_records)} augmented entries to splits.csv")
    else:
        print("\nNo new augmented entries (already done or no train CSVs found).")

    return 0


def stage_features(args):
    """
    Parallel feature extraction using joblib.
    Reads splits.csv, extracts:
      - Window features  → threat_windows.csv     (XGBoost)
      - Sequence features→ sequence_windows.npz   (BiLSTM)
      - Graph sequences  → graph_sequences.npz    (STGAT)
    """
    from config import (
        CSV_OUT_DIR, THREAT_WINDOWS_CSV,
        SEQUENCE_WINDOWS_NPZ, GRAPH_SEQ_NPZ,
    )
    import csv, os
    import joblib

    splits_csv = Path("data/splits.csv")
    if not splits_csv.exists():
        print("[ERROR] splits.csv not found — run --stage split first")
        return 1

    with open(splits_csv) as f:
        records = list(csv.DictReader(f))

    # Clear output files so we start fresh
    for p in [THREAT_WINDOWS_CSV, SEQUENCE_WINDOWS_NPZ, GRAPH_SEQ_NPZ]:
        if p.exists():
            p.unlink()

    def _extract_one(rec):
        vpath = Path(rec["video_path"])
        stem  = vpath.stem
        # Augmented records store the full CSV path directly in video_path
        csv_path = str(vpath) if vpath.suffix.lower() == ".csv" \
                   else str(CSV_OUT_DIR / f"{stem}.csv")
        if not Path(csv_path).exists():
            print(f"  [SKIP] No tracked CSV for {stem} — run --stage track first")
            return None, None, None

        from src.data.feature_extractor import (
            extract_windows, extract_sequences, extract_graph_sequences,
        )

        threat_type  = rec.get("threat_type", "")
        binary_label = int(rec.get("binary_label", -1))
        kw = dict(csv_path=csv_path, video_name=stem,
                  threat_type=threat_type, binary_label=binary_label)

        rows             = extract_windows(**kw)
        X, y, names      = extract_sequences(**kw)
        Xg, Ag, vg, yg, ng = extract_graph_sequences(**kw)

        return rows, (X, y, names), (Xg, Ag, vg, yg, ng)

    n_jobs = int(getattr(args, "jobs", -1)) if hasattr(args, "jobs") else -1
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        joblib.delayed(_extract_one)(rec) for rec in records
    )

    import numpy as np
    from src.data.feature_extractor import (
        write_windows, append_sequences, append_graph_sequences,
    )

    first_window = True
    all_X,  all_y,  all_names  = [], [], []
    all_Xg, all_Ag, all_vg     = [], [], []
    all_yg, all_ng             = [], []

    for rows, seq_data, graph_data in results:
        if rows is None:
            continue
        if rows:
            write_windows(rows, str(THREAT_WINDOWS_CSV), append=not first_window)
            first_window = False
        if seq_data is not None and len(seq_data[0]) > 0:
            all_X.append(seq_data[0]); all_y.append(seq_data[1])
            all_names.extend(seq_data[2])
        if graph_data is not None and len(graph_data[0]) > 0:
            all_Xg.append(graph_data[0]); all_Ag.append(graph_data[1])
            all_vg.append(graph_data[2]); all_yg.append(graph_data[3])
            all_ng.extend(graph_data[4])

    if all_X:
        append_sequences(np.concatenate(all_X), np.concatenate(all_y),
                         all_names, str(SEQUENCE_WINDOWS_NPZ))
    if all_Xg:
        append_graph_sequences(
            np.concatenate(all_Xg), np.concatenate(all_Ag),
            np.concatenate(all_vg), np.concatenate(all_yg),
            all_ng, str(GRAPH_SEQ_NPZ),
        )

    print("\nFeature extraction complete.")
    return 0


def stage_validate(args):
    return _run(_py("src/data/data_validator.py"), "Data Validation")


def stage_eda_pre(args):
    return _run(_py("src/evaluation/pre_train_eda.py"), "Pre-Training EDA")


def stage_train(args):
    model = getattr(args, "model", "all") or "all"
    return _run(_py("src/models/train_models.py", "--model", model), f"Train [{model}]")


def stage_evaluate(args):
    model = getattr(args, "model", "all") or "all"
    return _run(_py("src/evaluation/evaluate.py", "--model", model), f"Evaluate [{model}]")


def stage_eda_post(args):
    model = getattr(args, "model", "xgb") or "xgb"
    return _run(_py("src/evaluation/post_train_eda.py", "--model", model), f"Post-Train EDA [{model}]")


def stage_explain(args):
    model = getattr(args, "model", "all") or "all"
    return _run(_py("src/xai/explainability.py", "--model", model), f"xAI [{model}]")


def stage_infer(args):
    cmd = _py("src/inference/infer_threat.py")
    if getattr(args, "raw_video", None):
        cmd += ["--raw-video", args.raw_video]
    elif getattr(args, "input", None):
        cmd += ["--input", args.input]
    else:
        print("[ERROR] For --stage infer, provide --raw-video or --input")
        return 1
    if getattr(args, "output_video", None):
        cmd += ["--output-video", args.output_video]
    return _run(cmd, "Inference")


def stage_all(args):
    """Run the full training pipeline end-to-end."""
    stages = [
        ("prepare",   stage_prepare),
        ("split",     stage_split),
        ("track",     stage_track),
        # ("augment",   stage_augment),  # skipped — enough images collected
        ("features",  stage_features),
        ("validate",  stage_validate),
        ("eda-pre",   stage_eda_pre),
        ("train",     stage_train),
        ("evaluate",  stage_evaluate),
        ("eda-post",  stage_eda_post),
        ("explain",   stage_explain),
    ]
    failures = []
    for name, fn in stages:
        rc = fn(args)
        if rc != 0:
            failures.append(name)
            print(f"\n[WARN] Stage '{name}' returned non-zero ({rc}). Continuing...")

    print(f"\n{'='*60}")
    if failures:
        print(f"Pipeline complete with failures in: {', '.join(failures)}")
    else:
        print("Pipeline complete — all stages OK.")
    print(f"{'='*60}")
    return 0 if not failures else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

STAGE_MAP = {
    "all":          stage_all,
    "prepare":      stage_prepare,
    "split":        stage_split,
    "track":        stage_track,
    "augment":      stage_augment,
    "features":     stage_features,
    "batch-extract": stage_features,
    "validate":     stage_validate,
    "eda-pre":      stage_eda_pre,
    "train":        stage_train,
    "evaluate":     stage_evaluate,
    "eda-post":     stage_eda_post,
    "explain":      stage_explain,
    "infer":        stage_infer,
}


def main():
    parser = argparse.ArgumentParser(
        description="Threat Detection Pipeline — master caller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  prepare        Extract frames from raw_videos/ into frames/ (skips already done)
  split          Discover frame dirs / videos and assign train/val/test splits
  track          Run YOLO tracking on all sources in splits.csv
  augment        Augment training-split tracked CSVs (flip/rotate/scale)
  features       Extract window + sequence features (parallel via joblib)
  validate       Check data quality before training
  eda-pre        Exploratory analysis before training
  train          Train models (--model rf | xgb | bilstm | all)
  evaluate       Evaluate on held-out test split
  eda-post       Post-training error analysis
  explain        SHAP / BiLSTM attention xAI
  infer          Run inference on a new video
  all            Run full pipeline (prepare→split→track→...→explain)  [default]
""",
    )
    parser.add_argument("--stage",    default="all", choices=list(STAGE_MAP.keys()),
                        help="Pipeline stage to run (default: all = full run from frame extraction)")
    parser.add_argument("--augmentations", nargs="+",
                        default=["flip", "rotate", "scale"],
                        choices=["flip", "rotate", "scale", "all"],
                        help="Augmentation types for --stage augment")
    parser.add_argument("--model",    default="all",
                        choices=["xgb", "bilstm", "stgat", "all"],
                        help="Model for train/evaluate/explain/eda-post")
    parser.add_argument("--raw-video", help="Raw video path for --stage infer")
    parser.add_argument("--input",    help="Tracked CSV for --stage infer")
    parser.add_argument("--output-video", help="Annotated video output for --stage infer")
    parser.add_argument("--fps",      type=float, default=None,
                        help="Target FPS for frame extraction (--stage prepare/all)")
    parser.add_argument("--resize",   nargs=2, type=int, metavar=("W", "H"), default=None,
                        help="Resize frames during extraction (e.g. --resize 1280 720)")
    parser.add_argument("--jobs",     type=int, default=-1, help="Parallel workers (-1=all cores)")
    args = parser.parse_args()

    fn = STAGE_MAP[args.stage]
    rc = fn(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()

"""
mot17_visualizer.py

Renders two output videos for a single MOT17 sequence:
  1. trajectory_<seq>.mp4  — all GT tracks with colored trajectory trails + direction arrows
  2. pairs_<seq>.mp4       — same, but highlights specific leader/follower pairs

Pairs to highlight are read from data/derived/video_pair_windows_scored.csv
(or pair_windows_for_labeling.csv if the scored file doesn't exist yet).
Falls back to drawing all tracks if no pair file is found.

Usage:
    python mot17_visualizer.py                        # uses default sequence below
    python mot17_visualizer.py MOT17-04-FRCNN         # specify sequence name
"""

import os
import sys
import math
import configparser
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MOT17_TRAIN_DIR = "data/MOT17/train"
OUTPUT_DIR = "data/outputs/mot17_viz"

DEFAULT_SEQUENCE = "MOT17-04-FRCNN"

TRAJECTORY_TRAIL = 60       # how many past frames to draw in trail (None = unlimited)
DIRECTION_SMOOTH_N = 12     # frames used for direction smoothing

COLORS = [
    (0, 255, 0),    (255, 85, 0),   (0, 85, 255),   (255, 0, 170),
    (0, 255, 170),  (170, 0, 255),  (255, 255, 0),  (0, 170, 255),
    (255, 0, 0),    (170, 255, 0),  (0, 200, 200),  (200, 0, 200),
]

LEADER_COLOR   = (0, 255, 255)    # yellow
FOLLOWER_COLOR = (255, 128, 0)    # orange


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def track_color(track_id: int):
    return COLORS[track_id % len(COLORS)]


def read_seqinfo(seq_dir):
    ini_path = os.path.join(seq_dir, "seqinfo.ini")
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    s = cfg["Sequence"]
    return {
        "name":       s.get("name", os.path.basename(seq_dir)),
        "fps":        float(s.get("frameRate", 30)),
        "seq_length": int(s.get("seqLength", 0)),
        "width":      int(s.get("imWidth", 1920)),
        "height":     int(s.get("imHeight", 1080)),
        "im_dir":     s.get("imDir", "img1"),
        "im_ext":     s.get("imExt", ".jpg"),
    }


def read_mot_gt(gt_path):
    """Returns dict: frame -> list of (track_id, cx, cy, x1, y1, x2, y2)"""
    df = pd.read_csv(gt_path, header=None,
                     names=["frame","track_id","x","y","w","h","conf","class_id","vis"])
    df = df[df["class_id"] == 1].copy()   # pedestrians only
    df["cx"] = (df["x"] + df["w"] / 2).astype(int)
    df["cy"] = (df["y"] + df["h"] / 2).astype(int)
    df["x1"] = df["x"].astype(int)
    df["y1"] = df["y"].astype(int)
    df["x2"] = (df["x"] + df["w"]).astype(int)
    df["y2"] = (df["y"] + df["h"]).astype(int)

    frame_map = defaultdict(list)
    for _, row in df.iterrows():
        frame_map[int(row["frame"])].append((
            int(row["track_id"]),
            int(row["cx"]), int(row["cy"]),
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
        ))
    return frame_map


def compute_direction(pts, smooth_n=DIRECTION_SMOOTH_N):
    """Centroid-based direction from recent trajectory points. Returns (label, arrow, ux, uy) or None."""
    if len(pts) < 4:
        return None
    recent = pts[-smooth_n:]
    mid = len(recent) // 2
    first_half  = recent[:mid]
    second_half = recent[mid:]
    x1 = sum(p[0] for p in first_half)  / len(first_half)
    y1 = sum(p[1] for p in first_half)  / len(first_half)
    x2 = sum(p[0] for p in second_half) / len(second_half)
    y2 = sum(p[1] for p in second_half) / len(second_half)
    vx, vy = x2 - x1, y2 - y1
    mag = math.sqrt(vx**2 + vy**2)
    if mag < 2.0:
        return None
    angle = math.degrees(math.atan2(vy, vx))
    if angle < 0:
        angle += 360
    labels  = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
    arrows  = ["→",     "↘",          "↓",    "↙",         "←",    "↖",       "↑",  "↗"]
    idx = int((angle + 22.5) % 360 / 45)
    return labels[idx], arrows[idx], vx / mag, vy / mag


def draw_trajectory(frame, pts, color, trail=TRAJECTORY_TRAIL):
    if trail is not None:
        pts = pts[-trail:]
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=2)


def draw_arrow(frame, cx, cy, ux, uy, color, length=45):
    tip_x = int(cx + ux * length)
    tip_y = int(cy + uy * length)
    cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y), color, 2, tipLength=0.35)


def load_pairs(seq_name):
    """
    Load highlighted pairs from scored CSV or labeling CSV.
    Returns list of (leader_id, follower_id) tuples.
    """
    # try scored first
    for csv_path in [
        "data/derived/video_pair_windows_scored.csv",
        "data/derived/pair_windows_for_labeling.csv",
    ]:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        # filter to this sequence if column exists
        if "sequence" in df.columns:
            df = df[df["sequence"] == seq_name]
        if df.empty:
            continue

        # if scored, keep only predicted following pairs
        if "pred_following" in df.columns:
            df = df[df["pred_following"] == 1]

        pairs = list(zip(df["leader_id"].astype(int), df["follower_id"].astype(int)))
        return list(dict.fromkeys(pairs))   # deduplicate, preserve order

    return []


# ─────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────

def render_sequence(seq_name):
    seq_dir = os.path.join(MOT17_TRAIN_DIR, seq_name)
    if not os.path.isdir(seq_dir):
        print(f"Sequence not found: {seq_dir}")
        sys.exit(1)

    info    = read_seqinfo(seq_dir)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    img_dir = os.path.join(seq_dir, info["im_dir"])

    frame_map = read_mot_gt(gt_path)
    frames    = sorted(frame_map.keys())
    pairs     = load_pairs(seq_name)

    pair_set   = set(pairs)
    leader_ids = {p[0] for p in pairs}
    follower_ids = {p[1] for p in pairs}

    print(f"Sequence: {info['name']}  {info['width']}x{info['height']} @ {info['fps']} fps")
    print(f"Frames: {len(frames)},  GT tracks: {len({t for dets in frame_map.values() for t,*_ in dets})}")
    print(f"Pairs to highlight: {len(pairs)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size   = (info["width"], info["height"])
    fps    = info["fps"]

    traj_path  = os.path.join(OUTPUT_DIR, f"trajectory_{seq_name}.mp4")
    pairs_path = os.path.join(OUTPUT_DIR, f"pairs_{seq_name}.mp4")

    traj_writer  = cv2.VideoWriter(traj_path,  fourcc, fps, size)
    pairs_writer = cv2.VideoWriter(pairs_path, fourcc, fps, size)

    # rolling trajectory history: track_id -> list of (cx, cy)
    traj_history = defaultdict(list)

    for frame_idx, frame_no in enumerate(frames):
        img_file = os.path.join(img_dir, f"{frame_no:06d}{info['im_ext']}")
        base_frame = cv2.imread(img_file)
        if base_frame is None:
            print(f"  [warn] missing frame image: {img_file}")
            continue

        dets = frame_map[frame_no]

        # update trajectory history
        for tid, cx, cy, x1, y1, x2, y2 in dets:
            traj_history[tid].append((cx, cy))

        # ── TRAJECTORY VIDEO ──────────────────────────────
        traj_frame = base_frame.copy()

        for tid, pts in traj_history.items():
            color = track_color(tid)
            draw_trajectory(traj_frame, pts, color)

        for tid, cx, cy, x1, y1, x2, y2 in dets:
            color = track_color(tid)
            pts   = traj_history[tid]

            cv2.rectangle(traj_frame, (x1, y1), (x2, y2), color, 1)
            cv2.circle(traj_frame, (cx, cy), 4, color, -1)

            dir_info = compute_direction(pts)
            if dir_info:
                _, arrow, ux, uy = dir_info
                draw_arrow(traj_frame, cx, cy, ux, uy, color)
                label = f"ID {tid}  {arrow}"
            else:
                label = f"ID {tid}"

            cv2.putText(traj_frame, label,
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        traj_writer.write(traj_frame)

        # ── PAIR HIGHLIGHT VIDEO ──────────────────────────
        pair_frame = base_frame.copy()

        # draw all trails faded (gray) first
        for tid, pts in traj_history.items():
            draw_trajectory(pair_frame, pts, (100, 100, 100))

        # overlay highlighted tracks on top
        for tid, cx, cy, x1, y1, x2, y2 in dets:
            if tid in leader_ids:
                color = LEADER_COLOR
                role  = "LEADER"
            elif tid in follower_ids:
                color = FOLLOWER_COLOR
                role  = "FOLLOWER"
            else:
                color = (60, 60, 60)
                role  = None

            pts = traj_history[tid]

            if role:
                draw_trajectory(pair_frame, pts, color)
                cv2.rectangle(pair_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(pair_frame, (cx, cy), 5, color, -1)
                dir_info = compute_direction(pts)
                if dir_info:
                    _, arrow, ux, uy = dir_info
                    draw_arrow(pair_frame, cx, cy, ux, uy, color)
                    label = f"[{role}] ID {tid}  {arrow}"
                else:
                    label = f"[{role}] ID {tid}"
                cv2.putText(pair_frame, label,
                            (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.rectangle(pair_frame, (x1, y1), (x2, y2), color, 1)

        # legend
        if pairs:
            legend_y = 28
            cv2.putText(pair_frame, f"Following pairs: {len(pairs)}",
                        (12, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(pair_frame, "LEADER",
                        (12, legend_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LEADER_COLOR, 2)
            cv2.putText(pair_frame, "FOLLOWER",
                        (12, legend_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FOLLOWER_COLOR, 2)

        pairs_writer.write(pair_frame)

        if frame_idx % 100 == 0:
            print(f"  frame {frame_no}/{frames[-1]}")

    traj_writer.release()
    pairs_writer.release()

    print(f"\nSaved:")
    print(f"  {traj_path}")
    print(f"  {pairs_path}")


def main():
    seq_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEQUENCE
    render_sequence(seq_name)


if __name__ == "__main__":
    main()

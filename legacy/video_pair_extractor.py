"""
video_pair_extractor.py

Converts tracked_output_01.csv (from track_video.py) into the same
pair-window feature format as mot17_pair_extractor.py, then optionally
applies the trained RF model to score each window.

Output: data/derived/video_pair_windows.csv
        data/derived/video_pair_windows_scored.csv  (if model present)
"""

import os
import math
import pandas as pd
import joblib
from collections import defaultdict

INPUT_CSV = "data/outputs/csv/tracked_output_01.csv"
OUTPUT_CSV = "data/derived/video_pair_windows.csv"
OUTPUT_SCORED_CSV = "data/derived/video_pair_windows_scored.csv"
MODEL_PATH = "data/derived/following_rf.pkl"

WINDOW_SIZE = 30
WINDOW_STRIDE = 15
MIN_POINTS = 15
MAX_PAIR_DISTANCE = 250.0

FEATURES = [
    "num_points",
    "direction_similarity",
    "avg_distance",
    "distance_variance",
    "speed_similarity",
    "behind_ratio",
]


def euclidean(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def cosine_similarity(v1, v2):
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)


def get_motion_vector(points):
    if len(points) < 2:
        return (0.0, 0.0)
    _, x1, y1 = points[0]
    _, x2, y2 = points[-1]
    return (x2 - x1, y2 - y1)


def compute_pair_features(points_a, points_b):
    n = min(len(points_a), len(points_b))
    if n < MIN_POINTS:
        return None

    pa = points_a[-n:]
    pb = points_b[-n:]

    distances = []
    a_speeds = []
    b_speeds = []
    behind_count = 0

    vec_a = get_motion_vector(pa)
    vec_b = get_motion_vector(pb)
    dir_sim = cosine_similarity(vec_a, vec_b)

    dominant_horizontal = abs(vec_a[0]) >= abs(vec_a[1])

    for i in range(n):
        _, ax, ay = pa[i]
        _, bx, by = pb[i]

        distances.append(euclidean(ax, ay, bx, by))

        if i > 0:
            _, ax_prev, ay_prev = pa[i - 1]
            _, bx_prev, by_prev = pb[i - 1]
            a_speeds.append(euclidean(ax, ay, ax_prev, ay_prev))
            b_speeds.append(euclidean(bx, by, bx_prev, by_prev))

        if dominant_horizontal:
            if vec_a[0] > 0 and bx < ax:
                behind_count += 1
            elif vec_a[0] < 0 and bx > ax:
                behind_count += 1
        else:
            if vec_a[1] > 0 and by < ay:
                behind_count += 1
            elif vec_a[1] < 0 and by > ay:
                behind_count += 1

    avg_dist = sum(distances) / len(distances)
    mean_d = avg_dist
    dist_var = sum((d - mean_d) ** 2 for d in distances) / len(distances)

    speed_sim = 0.0
    if a_speeds and b_speeds:
        mean_a = sum(a_speeds) / len(a_speeds)
        mean_b = sum(b_speeds) / len(b_speeds)
        denom = max(mean_a, mean_b, 1e-6)
        speed_sim = 1.0 - abs(mean_a - mean_b) / denom

    behind_ratio = behind_count / n

    if avg_dist > MAX_PAIR_DISTANCE:
        return None

    return {
        "num_points": n,
        "direction_similarity": dir_sim,
        "avg_distance": avg_dist,
        "distance_variance": dist_var,
        "speed_similarity": speed_sim,
        "behind_ratio": behind_ratio,
    }


def load_tracks(csv_path):
    df = pd.read_csv(csv_path)
    track_map = defaultdict(list)
    for _, row in df.iterrows():
        track_map[int(row["track_id"])].append(
            (int(row["frame_idx"]), float(row["cx"]), float(row["cy"]))
        )
    # sort each track by frame
    for tid in track_map:
        track_map[tid].sort(key=lambda x: x[0])
    return track_map


def main():
    os.makedirs("data/derived", exist_ok=True)

    print(f"Loading tracks from {INPUT_CSV}...")
    track_map = load_tracks(INPUT_CSV)
    track_ids = sorted(track_map.keys())

    all_frames = sorted(set(f for pts in track_map.values() for f, _, _ in pts))
    print(f"Tracks: {len(track_ids)}, Frames: {len(all_frames)}")

    rows = []
    start_idx = 0
    while start_idx < len(all_frames):
        end_idx = start_idx + WINDOW_SIZE
        window_frames = all_frames[start_idx:end_idx]
        if len(window_frames) < WINDOW_SIZE:
            break

        f_start, f_end = window_frames[0], window_frames[-1]

        # filter each track to this window
        window_tracks = {}
        for tid in track_ids:
            pts = [(f, x, y) for f, x, y in track_map[tid] if f_start <= f <= f_end]
            if len(pts) >= MIN_POINTS:
                window_tracks[tid] = pts

        wids = sorted(window_tracks.keys())
        for leader_id in wids:
            for follower_id in wids:
                if leader_id == follower_id:
                    continue
                feat = compute_pair_features(
                    window_tracks[leader_id], window_tracks[follower_id]
                )
                if feat is None:
                    continue
                rows.append({
                    "leader_id": leader_id,
                    "follower_id": follower_id,
                    "start_frame": f_start,
                    "end_frame": f_end,
                    **feat,
                })

        start_idx += WINDOW_STRIDE

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} pair windows to {OUTPUT_CSV}")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        X = out_df[FEATURES]
        out_df["pred_following"] = model.predict(X)
        if hasattr(model, "predict_proba"):
            out_df["pred_score"] = model.predict_proba(X)[:, 1]
        out_df.to_csv(OUTPUT_SCORED_CSV, index=False)
        print(f"Scored and saved to {OUTPUT_SCORED_CSV}")
        n_following = out_df["pred_following"].sum()
        print(f"Predicted following pairs: {n_following}/{len(out_df)}")
    else:
        print(f"No model found at {MODEL_PATH} — run train_following_baseline.py first.")


if __name__ == "__main__":
    main()

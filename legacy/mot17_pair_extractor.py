import os
import math
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
MOT17_TRAIN_DIR = "data/MOT17/train"
OUTPUT_CSV = "data/derived/pair_windows_for_labeling.csv"

WINDOW_SIZE = 30      # frames per sample window
WINDOW_STRIDE = 15    # overlap
MIN_POINTS = 15       # minimum points per track in window
MAX_PAIR_DISTANCE = 250.0  # reject pairs too far apart


def euclidean(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def cosine_similarity(v1, v2):
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)


def read_mot_gt(gt_path):
    """
    Reads MOT ground truth:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    df = pd.read_csv(gt_path, header=None)
    df.columns = [
        "frame", "track_id", "x", "y", "w", "h",
        "conf", "class_id", "visibility"
    ]

    # Keep person class only if available
    # In MOTChallenge, class_id=1 is pedestrian in gt usage context
    df = df[df["class_id"] == 1].copy()

    df["cx"] = df["x"] + df["w"] / 2.0
    df["cy"] = df["y"] + df["h"] / 2.0
    return df


def build_window_track_map(df_window):
    track_map = defaultdict(list)
    for _, row in df_window.iterrows():
        track_map[int(row["track_id"])].append(
            (int(row["frame"]), float(row["cx"]), float(row["cy"]))
        )
    return track_map


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

        # Is B behind A?
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
    dist_var = pd.Series(distances).var() if len(distances) > 1 else 0.0
    speed_sim = 0.0
    if len(a_speeds) > 0 and len(b_speeds) > 0:
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
        "behind_ratio": behind_ratio
    }


def process_sequence(seq_dir):
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    seq_name = os.path.basename(seq_dir)

    if not os.path.exists(gt_path):
        return []

    df = read_mot_gt(gt_path)
    if df.empty:
        return []

    frames = sorted(df["frame"].unique())
    rows = []

    start_idx = 0
    while start_idx < len(frames):
        end_idx = start_idx + WINDOW_SIZE
        window_frames = frames[start_idx:end_idx]
        if len(window_frames) < WINDOW_SIZE:
            break

        f_start = window_frames[0]
        f_end = window_frames[-1]

        df_window = df[(df["frame"] >= f_start) & (df["frame"] <= f_end)]
        track_map = build_window_track_map(df_window)
        track_ids = sorted(track_map.keys())

        for leader_id in track_ids:
            for follower_id in track_ids:
                if leader_id == follower_id:
                    continue

                feat = compute_pair_features(
                    track_map[leader_id],
                    track_map[follower_id]
                )
                if feat is None:
                    continue

                rows.append({
                    "sequence": seq_name,
                    "leader_id": leader_id,
                    "follower_id": follower_id,
                    "start_frame": f_start,
                    "end_frame": f_end,
                    **feat,
                    "following_label": ""  # manual label later
                })

        start_idx += WINDOW_STRIDE

    return rows


def main():
    os.makedirs("data/derived", exist_ok=True)

    all_rows = []
    for seq_name in sorted(os.listdir(MOT17_TRAIN_DIR)):
        seq_dir = os.path.join(MOT17_TRAIN_DIR, seq_name)
        if os.path.isdir(seq_dir):
            seq_rows = process_sequence(seq_dir)
            all_rows.extend(seq_rows)
            print(f"Processed {seq_name}: {len(seq_rows)} pair windows")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total pair windows: {len(out_df)}")


if __name__ == "__main__":
    main()
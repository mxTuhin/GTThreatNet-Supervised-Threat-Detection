"""
threat_feature_extractor.py
============================
Extracts group-level threat features from a tracked CSV (output of track_video.py).

For each time window and each potential "target" person, it computes features
that describe the collective behavior of all other nearby persons (the threat group).

Output: threat_windows.csv with one row per (video, window, target_id)
        with a blank `threat_label` column ready for manual annotation.

Usage:
    python src/threat_feature_extractor.py --input outputs/csv/myvideo.csv \
                                           --video-name myvideo \
                                           --output data/derived/threat_windows.csv
"""

import os
import csv
import math
import argparse
from collections import defaultdict

# ─────────────────────────────────────────────
# Config defaults (override via CLI args)
# ─────────────────────────────────────────────
WINDOW_SIZE = 30          # frames per sample window
WINDOW_STRIDE = 15        # stride between windows (overlap = WINDOW_SIZE - WINDOW_STRIDE)
MIN_TRACK_POINTS = 10     # min detections a track must have inside the window
PROXIMITY_RADIUS = 300.0  # pixels — max distance to be counted as "nearby"
MIN_NEARBY = 1            # at least this many nearby persons to produce a row


def load_tracked_csv(csv_path: str) -> dict:
    """
    Reads track_video.py output CSV.
    Returns: dict[frame_idx -> list of (track_id, cx, cy)]
    """
    frame_map = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            track_id  = int(row["track_id"])
            cx        = float(row["cx"])
            cy        = float(row["cy"])
            frame_map[frame_idx].append((track_id, cx, cy))
    return frame_map


def build_window_track_map(frame_map: dict, window_frames: list) -> dict:
    """
    Returns: dict[track_id -> list of (frame_idx, cx, cy)]
    for all detections within the window.
    """
    tracks = defaultdict(list)
    for f in window_frames:
        for track_id, cx, cy in frame_map.get(f, []):
            tracks[track_id].append((f, cx, cy))
    # sort each track by frame
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return tracks


def euclidean(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def closing_rate(points_other, points_target) -> float:
    """
    Returns average closing rate (px/frame) of `other` toward `target`.
    Positive value = closing in. Negative = moving away.
    Uses overlapping frames (matched by index, not frame_id).
    """
    n = min(len(points_other), len(points_target))
    if n < 2:
        return 0.0

    po = points_other[-n:]
    pt = points_target[-n:]

    rates = []
    for i in range(1, n):
        d_prev = euclidean(po[i-1][1], po[i-1][2], pt[i-1][1], pt[i-1][2])
        d_curr = euclidean(po[i  ][1], po[i  ][2], pt[i  ][1], pt[i  ][2])
        rates.append(d_prev - d_curr)  # positive = getting closer

    return sum(rates) / len(rates) if rates else 0.0


def velocity_toward_target(points_other, points_target) -> float:
    """
    Component of `other`'s velocity that is directed toward the current
    position of `target`.  Returns max per-frame value in the window.
    """
    n = min(len(points_other), len(points_target))
    if n < 2:
        return 0.0

    po = points_other[-n:]
    pt = points_target[-n:]

    max_v = 0.0
    for i in range(1, n):
        # velocity vector of other
        vx = po[i][1] - po[i-1][1]
        vy = po[i][2] - po[i-1][2]

        # direction from other toward target (at current frame)
        dx = pt[i][1] - po[i][1]
        dy = pt[i][2] - po[i][2]
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue

        # dot product of velocity with unit direction toward target
        v_toward = (vx * dx + vy * dy) / dist
        max_v = max(max_v, v_toward)

    return max_v


def angle_from_target(tx, ty, ox, oy) -> float:
    """Angle in degrees from target T to other person O."""
    return math.degrees(math.atan2(oy - ty, ox - tx))


def angle_spread(angles: list) -> float:
    """
    Circular std dev of angles (degrees).
    High spread = persons present in many directions around target (surrounding signal).
    """
    if len(angles) < 2:
        return 0.0
    rads = [math.radians(a) for a in angles]
    sin_mean = sum(math.sin(r) for r in rads) / len(rads)
    cos_mean = sum(math.cos(r) for r in rads) / len(rads)
    R = math.sqrt(sin_mean**2 + cos_mean**2)  # mean resultant length
    return math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))))


def converging_sector_count(closing_persons_angles: list, n_sectors: int = 4) -> int:
    """
    Count how many distinct angular sectors (quadrants by default) have at least
    one person who is actively CLOSING toward the target.

    High value (3–4) = persons converging from multiple directions = strong surrounding signal.
    Low value (1) = everyone from same direction = fast_approach or following.

    closing_persons_angles: list of angles (degrees) only for persons whose closing_rate > 0.
    """
    if not closing_persons_angles:
        return 0
    sector_size = 360.0 / n_sectors
    occupied = set()
    for a in closing_persons_angles:
        sector = int((a % 360) / sector_size)
        occupied.add(sector)
    return len(occupied)


def distance_consistency(points_other, points_target) -> float:
    """
    Inverse of distance variance between other and target over the window.
    High value = very consistent spacing → key signal for following behavior.
    Returns value in [0, 1] (1 = perfectly consistent distance).
    """
    n = min(len(points_other), len(points_target))
    if n < 3:
        return 0.0
    po = points_other[-n:]
    pt = points_target[-n:]
    dists = [euclidean(po[i][1], po[i][2], pt[i][1], pt[i][2]) for i in range(n)]
    mean_d = sum(dists) / len(dists)
    if mean_d < 1e-6:
        return 0.0
    variance = sum((d - mean_d) ** 2 for d in dists) / len(dists)
    # normalize by mean distance so scale-invariant
    cv = math.sqrt(variance) / mean_d   # coefficient of variation
    return max(0.0, 1.0 - cv)           # high = consistent


def mean_speed(points) -> float:
    if len(points) < 2:
        return 0.0
    speeds = [
        euclidean(points[i][1], points[i][2], points[i-1][1], points[i-1][2])
        for i in range(1, len(points))
    ]
    return sum(speeds) / len(speeds)


def movement_direction(points):
    """Returns (vx, vy) unit vector of overall movement, or None if stationary."""
    if len(points) < 2:
        return None
    vx = points[-1][1] - points[0][1]
    vy = points[-1][2] - points[0][2]
    mag = math.sqrt(vx**2 + vy**2)
    if mag < 1e-6:
        return None
    return vx / mag, vy / mag


def is_behind(target_pts, other_last_cx, other_last_cy) -> bool:
    """True if other person is in the rear half-plane of target's movement."""
    mv = movement_direction(target_pts)
    if mv is None:
        return False
    vx, vy = mv
    tx, ty = target_pts[-1][1], target_pts[-1][2]
    # vector from target to other
    dx = other_last_cx - tx
    dy = other_last_cy - ty
    # dot product with movement direction — negative = behind
    return (dx * vx + dy * vy) < 0


def compute_threat_features(target_pts: list, nearby_tracks: dict) -> dict:
    """
    Compute group-level threat features for a single target person.

    Parameters
    ----------
    target_pts   : list of (frame_idx, cx, cy) for the target
    nearby_tracks: dict of track_id -> [(frame_idx, cx, cy)] for nearby persons

    Returns
    -------
    dict of feature name -> value, or None if too few data
    """
    if len(target_pts) < 4:
        return None

    n_nearby = len(nearby_tracks)
    if n_nearby < MIN_NEARBY:
        return None

    closing_rates  = []
    closing_speeds = []  # only positive (actually closing) rates
    velocities_toward = []
    angles = []
    closing_angles = []  # angles only for persons actively closing
    group_speeds = []
    behind_count = 0
    dist_consistencies = []

    for tid, other_pts in nearby_tracks.items():
        if len(other_pts) < 2:
            continue

        rate = closing_rate(other_pts, target_pts)
        closing_rates.append(rate)

        ox, oy = other_pts[-1][1], other_pts[-1][2]
        tx, ty = target_pts[-1][1], target_pts[-1][2]
        ang = angle_from_target(tx, ty, ox, oy)
        angles.append(ang)

        if rate > 0:
            closing_speeds.append(rate)
            closing_angles.append(ang)

        vtt = velocity_toward_target(other_pts, target_pts)
        velocities_toward.append(vtt)

        group_speeds.append(mean_speed(other_pts))

        if is_behind(target_pts, ox, oy):
            behind_count += 1

        dist_consistencies.append(distance_consistency(other_pts, target_pts))

    num_closing = sum(1 for r in closing_rates if r > 0)

    # min distance across all nearby persons (at last overlapping frame)
    min_dist = float("inf")
    for other_pts in nearby_tracks.values():
        if other_pts:
            ox, oy = other_pts[-1][1], other_pts[-1][2]
            tx, ty = target_pts[-1][1], target_pts[-1][2]
            d = euclidean(tx, ty, ox, oy)
            min_dist = min(min_dist, d)
    if min_dist == float("inf"):
        min_dist = PROXIMITY_RADIUS

    return {
        # ── general proximity ──
        "threat_person_count":        n_nearby,
        "min_distance":               min_dist,
        # ── closing / approach ──
        "num_closing_persons":        num_closing,
        "max_closing_speed":          max(closing_speeds) if closing_speeds else 0.0,
        "avg_closing_rate":           sum(closing_rates) / len(closing_rates) if closing_rates else 0.0,
        "max_velocity_toward_target": max(velocities_toward) if velocities_toward else 0.0,
        # ── surrounding-specific ──
        "angle_spread":               angle_spread(angles),
        "converging_sector_count":    converging_sector_count(closing_angles),
        # ── following-specific ──
        "behind_person_count":        behind_count,
        "avg_distance_consistency":   (sum(dist_consistencies) / len(dist_consistencies)
                                       if dist_consistencies else 0.0),
        # ── group motion ──
        "avg_speed_of_group":         sum(group_speeds) / len(group_speeds) if group_speeds else 0.0,
        "target_speed":               mean_speed(target_pts),
    }


def extract_windows(csv_path: str, video_name: str,
                    window_size: int, stride: int,
                    proximity: float = PROXIMITY_RADIUS) -> list:
    """
    Main extraction loop.
    Returns list of dicts, one per (window, target_id).
    """
    frame_map = load_tracked_csv(csv_path)
    all_frames = sorted(frame_map.keys())

    if not all_frames:
        return []

    rows = []
    i = 0
    while i < len(all_frames):
        window_frames = all_frames[i : i + window_size]
        if len(window_frames) < window_size:
            break

        f_start = window_frames[0]
        f_end   = window_frames[-1]

        track_map = build_window_track_map(frame_map, window_frames)

        # filter tracks with enough points in this window
        valid_tracks = {
            tid: pts
            for tid, pts in track_map.items()
            if len(pts) >= MIN_TRACK_POINTS
        }

        for target_id, target_pts in valid_tracks.items():
            # find the target's last known position to measure proximity
            tx, ty = target_pts[-1][1], target_pts[-1][2]

            # nearby = other valid tracks whose last position is within proximity radius
            nearby = {}
            for other_id, other_pts in valid_tracks.items():
                if other_id == target_id:
                    continue
                ox, oy = other_pts[-1][1], other_pts[-1][2]
                if euclidean(tx, ty, ox, oy) <= proximity:
                    nearby[other_id] = other_pts

            if len(nearby) < MIN_NEARBY:
                continue

            feats = compute_threat_features(target_pts, nearby)
            if feats is None:
                continue

            rows.append({
                "video_name":  video_name,
                "start_frame": f_start,
                "end_frame":   f_end,
                "target_id":   target_id,
                **feats,
                "threat_label": "",   # fill in manually or via autolabel
                "threat_type":  "",   # normal / following / surrounding / fast_approach
            })

        i += stride

    return rows


def write_output(rows: list, output_csv: str, append: bool = False):
    """Write (or append) feature rows to output CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    fieldnames = [
        "video_name", "start_frame", "end_frame", "target_id",
        # general proximity
        "threat_person_count", "min_distance",
        # closing / approach
        "num_closing_persons", "max_closing_speed", "avg_closing_rate",
        "max_velocity_toward_target",
        # surrounding-specific
        "angle_spread", "converging_sector_count",
        # following-specific
        "behind_person_count", "avg_distance_consistency",
        # group motion
        "avg_speed_of_group", "target_speed",
        # labels
        "threat_label", "threat_type",
    ]

    mode = "a" if append else "w"
    write_header = not (append and os.path.exists(output_csv))

    with open(output_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"{'Appended' if append else 'Wrote'} {len(rows)} rows → {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract threat features from tracked CSV")
    parser.add_argument("--input",       required=True,  help="Path to tracked CSV (track_video.py output)")
    parser.add_argument("--video-name",  required=True,  help="Identifier for this video (e.g. 'surround_001')")
    parser.add_argument("--output",      default="data/derived/threat_windows.csv",
                        help="Output CSV path")
    parser.add_argument("--append",      action="store_true",
                        help="Append to existing output CSV instead of overwriting")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride",      type=int, default=WINDOW_STRIDE)
    parser.add_argument("--proximity",   type=float, default=PROXIMITY_RADIUS,
                        help="Max pixel distance to count as 'nearby'")
    args = parser.parse_args()

    rows = extract_windows(
        csv_path    = args.input,
        video_name  = args.video_name,
        window_size = args.window_size,
        stride      = args.stride,
        proximity   = args.proximity,
    )

    print(f"Extracted {len(rows)} threat windows from '{args.video_name}'")
    write_output(rows, args.output, append=args.append)


if __name__ == "__main__":
    main()

import os
import csv
from collections import defaultdict
from utilities.trajectory_utils import (
    vector_from_points,
    cosine_similarity,
    average_distance,
    behind_ratio,
    distance_variance,
    speed_similarity
)
import cv2
import numpy as np

CSV_PATH = "../data/outputs/csv/tracked_output_01.csv"
SUMMARY_CSV_PATH = "../data/outputs/csv/track_summary_01.csv"
VIDEO_PATH = "../data/raw_videos/data-vid.mp4"
OUTPUT_IMG_DIR = "../data/outputs/images"
DISPLAY_IMAGES = False  # set True to attempt to pop up windows (may fail in headless)

MIN_TRAJ_LEN = 20
MIN_DIRECTION_SIM = 0.75
MIN_BEHIND_RATIO = 0.60
MIN_AVG_DISTANCE = 30
MAX_AVG_DISTANCE = 180


def load_tracks(csv_path):
    """
    Returns track_id -> list of (frame_idx, cx, cy)
    """
    tracks = defaultdict(list)
    # dets_map: (track_id, frame_idx) -> (x1,y1,x2,y2,cx,cy,confidence)
    dets_map = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            track_id = int(row["track_id"])
            x1 = int(row.get("x1", 0))
            y1 = int(row.get("y1", 0))
            x2 = int(row.get("x2", 0))
            y2 = int(row.get("y2", 0))
            cx = int(row.get("cx", (x1 + x2) // 2))
            cy = int(row.get("cy", (y1 + y2) // 2))
            conf = float(row.get("confidence", 0.0)) if row.get("confidence", "") != "" else 0.0
            tracks[track_id].append((frame_idx, cx, cy))
            dets_map[(track_id, frame_idx)] = (x1, y1, x2, y2, cx, cy, conf)

    return tracks, dets_map


def load_track_summary(summary_csv_path):
    """
    Returns a dict: track_id -> {start_frame, end_frame, frame_count}
    """
    summary = {}
    try:
        with open(summary_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row["track_id"])
                start = int(row["start_frame"]) if row["start_frame"] != "" else None
                end = int(row["end_frame"]) if row["end_frame"] != "" else None
                count = int(row["frame_count"]) if row["frame_count"] != "" else 0
                summary[tid] = {"start_frame": start, "end_frame": end, "frame_count": count}
    except FileNotFoundError:
        # summary CSV may not exist yet; return empty dict
        pass
    return summary


def score_following(leader_points, follower_points):
    if len(leader_points) < MIN_TRAJ_LEN or len(follower_points) < MIN_TRAJ_LEN:
        return None

    leader_vec = vector_from_points(leader_points)
    follower_vec = vector_from_points(follower_points)

    if leader_vec is None or follower_vec is None:
        return None

    dir_sim = cosine_similarity(leader_vec, follower_vec)
    avg_dist = average_distance(leader_points, follower_points)
    b_ratio = behind_ratio(leader_points, follower_points)
    dist_var = distance_variance(leader_points, follower_points)
    spd_sim = speed_similarity(leader_points, follower_points)

    if avg_dist is None:
        return None

    score = 0.0
    score += max(0.0, dir_sim) * 0.4
    score += b_ratio * 0.3
    score += spd_sim * 0.15

    # Distance preference: closer but not too close
    if MIN_AVG_DISTANCE <= avg_dist <= MAX_AVG_DISTANCE:
        score += 0.15

    return {
        "num_points": min(len(leader_points), len(follower_points)),
        "direction_similarity": dir_sim,
        "avg_distance": avg_dist,
        "distance_variance": dist_var,
        "speed_similarity": spd_sim,
        "behind_ratio": b_ratio,
        "score": score
    }


# ---------------------
# Image extraction utils
# ---------------------

def ensure_out_dir():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def get_frame_from_cap(cap, frame_idx):
    # Seek and read a single frame. Returns BGR image or None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def save_image(path, image):
    cv2.imwrite(path, image)


def make_side_by_side(img_a, img_b, height=360):
    # Resize both to same height, keep aspect ratio
    def resize_to_h(img, h):
        h0, w0 = img.shape[:2]
        if h0 == h:
            return img
        w = int(w0 * (h / h0))
        return cv2.resize(img, (w, h))

    ra = resize_to_h(img_a, height)
    rb = resize_to_h(img_b, height)
    return np.hstack([ra, rb])


def save_representative_frames(cap, track_id, start, end, prefix, dets_map=None):
    """
    Save representative frames for a track: start, mid, end (if distinct).
    Returns list of saved file paths.
    """
    saved = []
    if start is None or end is None:
        return saved

    frames_to_save = [start]
    if end != start:
        mid = (start + end) // 2
        if mid != start and mid != end:
            frames_to_save.append(mid)
        frames_to_save.append(end)

    for fi in frames_to_save:
        img = get_frame_from_cap(cap, fi)
        if img is None:
            continue
        # annotate with bbox + id if available
        ann = img.copy()
        if dets_map is not None and (track_id, fi) in dets_map:
            x1, y1, x2, y2, cx, cy, conf = dets_map[(track_id, fi)]
            # draw bbox
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw center
            cv2.circle(ann, (cx, cy), 4, (0, 0, 255), -1)
            # id label
            cv2.putText(ann, f"ID {track_id}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fname = os.path.join(OUTPUT_IMG_DIR, f"{prefix}_track{track_id}_frame{fi}.jpg")
        save_image(fname, ann)
        saved.append(fname)
    return saved


def save_pair_images(cap, leader_id, follower_id, leader_start, leader_end, follower_start, follower_end, dets_map=None):
    # skip invalid ids
    if leader_id is None or follower_id is None:
        return []
    if leader_id < 0 or follower_id < 0:
        return []

    saved_paths = []
    leader_paths = save_representative_frames(cap, leader_id, leader_start, leader_end, "leader", dets_map=dets_map)
    follower_paths = save_representative_frames(cap, follower_id, follower_start, follower_end, "follower", dets_map=dets_map)

    # create side-by-side composites for the first available pair of representative frames
    if leader_paths and follower_paths:
        # pick the first saved of each
        la = cv2.imread(leader_paths[0])
        fb = cv2.imread(follower_paths[0])
        if la is not None and fb is not None:
            composite = make_side_by_side(la, fb, height=360)
            comp_name = os.path.join(OUTPUT_IMG_DIR, f"pair_le{leader_id}_fu{follower_id}.jpg")
            save_image(comp_name, composite)
            saved_paths.append(comp_name)

            # optionally display
            if DISPLAY_IMAGES:
                try:
                    cv2.imshow(f"Leader {leader_id} | Follower {follower_id}", composite)
                    cv2.waitKey(500)  # show briefly
                    cv2.destroyWindow(f"Leader {leader_id} | Follower {follower_id}")
                except Exception:
                    # in headless environments this may fail; ignore
                    pass

    return saved_paths


def main():
    tracks, dets_map = load_tracks(CSV_PATH)
    summaries = load_track_summary(SUMMARY_CSV_PATH)
    track_ids = list(tracks.keys())

    results = []

    for leader_id in track_ids:
        for follower_id in track_ids:
            if leader_id == follower_id:
                continue

            info = score_following(tracks[leader_id], tracks[follower_id])
            if info is None:
                continue

            if (
                info["direction_similarity"] >= MIN_DIRECTION_SIM
                and info["behind_ratio"] >= MIN_BEHIND_RATIO
                and MIN_AVG_DISTANCE <= info["avg_distance"] <= MAX_AVG_DISTANCE
            ):
                leader_summary = summaries.get(leader_id, {})
                follower_summary = summaries.get(follower_id, {})

                results.append({
                    "leader_id": leader_id,
                    "follower_id": follower_id,
                    "leader_start": leader_summary.get("start_frame"),
                    "leader_end": leader_summary.get("end_frame"),
                    "leader_count": leader_summary.get("frame_count"),
                    "follower_start": follower_summary.get("start_frame"),
                    "follower_end": follower_summary.get("end_frame"),
                    "follower_count": follower_summary.get("frame_count"),
                    **info
                })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\nPotential following pairs:\n")

    # prepare video capture and output dir for images
    ensure_out_dir()
    try:
        cap = open_video(VIDEO_PATH)
    except RuntimeError:
        cap = None

    for r in results[:20]:
        print(
            f"Leader={r['leader_id']}[{r.get('leader_start')}-{r.get('leader_end')}|n={r.get('leader_count')}] "
            f"Follower={r['follower_id']}[{r.get('follower_start')}-{r.get('follower_end')}|n={r.get('follower_count')}] "
            f"score={r['score']:.3f} "
            f"dir_sim={r['direction_similarity']:.3f} "
            f"avg_dist={r['avg_distance']:.1f} "
            f"behind_ratio={r['behind_ratio']:.2f}"
        )

        # determine start/end frames if summary missing
        lstart = r.get('leader_start')
        lend = r.get('leader_end')
        if (lstart is None or lend is None) and cap is not None:
            # fallback to tracks list
            tpts = tracks.get(r['leader_id'], [])
            if tpts:
                lstart = tpts[0][0]
                lend = tpts[-1][0]

        fstart = r.get('follower_start')
        fend = r.get('follower_end')
        if (fstart is None or fend is None) and cap is not None:
            tpts = tracks.get(r['follower_id'], [])
            if tpts:
                fstart = tpts[0][0]
                fend = tpts[-1][0]

        if cap is not None:
            saved = save_pair_images(cap, r['leader_id'], r['follower_id'], lstart, lend, fstart, fend, dets_map=dets_map)
            if saved:
                print(f"Saved comparison images: {saved}")
            else:
                print("No images saved for this pair (missing frames or capture failed).")
        else:
            print("Video file not opened; cannot extract frames.")

    if cap is not None:
        cap.release()
    # close any open windows if display was used
    if DISPLAY_IMAGES:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
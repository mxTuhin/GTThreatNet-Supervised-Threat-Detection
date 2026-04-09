import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ---------------------------
# Config
# ---------------------------
VIDEO_PATH = "../data/raw_videos/data-vid.mp4"
OUTPUT_VIDEO_PATH = "../data/outputs/video/tracked_output_01.mp4"
OUTPUT_TRAJECTORY_VIDEO_PATH = "../data/outputs/video/trajectory_output_01.mp4"
OUTPUT_CSV_PATH = "../data/outputs/csv/tracked_output_01.csv"
OUTPUT_SUMMARY_CSV_PATH = "../data/outputs/csv/track_summary_01.csv"

# Trajectory trail length (number of past center points to draw); None = unlimited
TRAJECTORY_MAXLEN = None

# Detection model — yolov8m gives noticeably fewer missed detections vs yolov8n,
# which directly reduces the number of frames a track is "lost" and thus ID switches.
# Switch back to "yolov8n.pt" if speed is more important than accuracy.
MODEL_NAME = "yolov8m.pt" #rtdetr-l.pt
CONF_THRESHOLD = 0.35
PERSON_CLASS_ID = 0         # COCO: person

# Tracker config.
# custom_botsort.yaml  — BoT-SORT + ReID + 5-second track buffer (recommended)
# custom_bytetrack.yaml — ByteTrack + 5-second buffer, no ReID (lighter fallback)
TRACKER_CONFIG = "custom_botsort.yaml"

# create nested output dirs
os.makedirs("data/outputs/csv", exist_ok=True)
os.makedirs("data/outputs/video", exist_ok=True)

def get_box_center(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


# Distinct palette — cycles if more than len(COLORS) IDs appear
COLORS = [
    (0, 255, 0),    # green
    (255, 85, 0),   # orange
    (0, 85, 255),   # blue
    (255, 0, 170),  # pink
    (0, 255, 170),  # cyan-green
    (170, 0, 255),  # purple
    (255, 255, 0),  # yellow
    (0, 170, 255),  # sky blue
    (255, 0, 0),    # red
    (170, 255, 0),  # lime
]

def track_color(track_id: int):
    return COLORS[track_id % len(COLORS)]


def draw_trajectories(frame, trajectory_history: dict):
    """
    Draw a polyline trail for every track_id in trajectory_history.
    trajectory_history: dict[int, list[tuple[int,int]]]  (track_id -> [(cx,cy), ...])
    """
    for tid, pts in trajectory_history.items():
        if len(pts) < 2:
            continue
        color = track_color(tid)
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_arr], isClosed=False, color=color, thickness=2)

def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # safer lookup for VideoWriter_fourcc (some stubs/type checkers may flag it)
    fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fourcc_func):
        fourcc = fourcc_func(*"mp4v")
    else:
        # fallback: 0 will let OpenCV choose a default codec on some platforms
        fourcc = 0

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    traj_writer = cv2.VideoWriter(OUTPUT_TRAJECTORY_VIDEO_PATH, fourcc, fps, (width, height))

    # track-level stats: store start, end frames and count per track
    track_stats = defaultdict(lambda: {"count": 0, "min_frame": None, "max_frame": None})

    # trajectory history: track_id -> list of (cx, cy) in order
    trajectory_history: dict = defaultdict(list)

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame_idx", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"
        ])

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            annotated = frame.copy()

            if results and len(results) > 0:
                boxes = results[0].boxes

                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
                    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([])
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                    for i in range(len(xyxy)):
                        cls_id = clss[i] if len(clss) > i else -1
                        if cls_id != PERSON_CLASS_ID:
                            continue

                        track_id = int(ids[i]) if ids is not None and len(ids) > i else -1
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        confidence = float(confs[i]) if len(confs) > i else 0.0
                        cx, cy = get_box_center(x1, y1, x2, y2)

                        # write detection row
                        csv_writer.writerow([
                            frame_idx, track_id, x1, y1, x2, y2, cx, cy, confidence
                        ])

                        # update per-track stats
                        stats = track_stats[track_id]
                        stats["count"] += 1
                        if stats["min_frame"] is None or frame_idx < stats["min_frame"]:
                            stats["min_frame"] = frame_idx
                        if stats["max_frame"] is None or frame_idx > stats["max_frame"]:
                            stats["max_frame"] = frame_idx

                        # update trajectory history
                        pts = trajectory_history[track_id]
                        pts.append((cx, cy))
                        if TRAJECTORY_MAXLEN is not None and len(pts) > TRAJECTORY_MAXLEN:
                            del pts[0]

                        color = track_color(track_id)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(annotated, (cx, cy), 4, color, -1)
                        cv2.putText(
                            annotated,
                            f"ID {track_id}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

            # draw trajectory trails on top of boxes/labels
            draw_trajectories(annotated, trajectory_history)

            # trajectory-only frame: blank canvas + trails + labels (no boxes)
            traj_frame = frame.copy()
            draw_trajectories(traj_frame, trajectory_history)
            # mark current positions
            for tid, pts in trajectory_history.items():
                if pts:
                    cx, cy = pts[-1]
                    cv2.circle(traj_frame, (cx, cy), 5, track_color(tid), -1)
                    cv2.putText(
                        traj_frame,
                        f"ID {tid}",
                        (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        track_color(tid),
                        1,
                    )

            writer.write(annotated)
            traj_writer.write(traj_frame)
            cv2.imshow("Tracking", annotated)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

            frame_idx += 1

    cap.release()
    writer.release()
    traj_writer.release()
    cv2.destroyAllWindows()

    # write per-track summary CSV so downstream code can quickly see frame ranges and counts
    with open(OUTPUT_SUMMARY_CSV_PATH, "w", newline="") as sf:
        s_writer = csv.writer(sf)
        s_writer.writerow(["track_id", "start_frame", "end_frame", "frame_count"])
        for tid, s in sorted(track_stats.items(), key=lambda x: int(x[0])):
            s_writer.writerow([tid, s["min_frame"], s["max_frame"], s["count"]])

    print(f"Saved tracked video to: {OUTPUT_VIDEO_PATH}")
    print(f"Saved trajectory video to: {OUTPUT_TRAJECTORY_VIDEO_PATH}")
    print(f"Saved tracks CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved track summary CSV to: {OUTPUT_SUMMARY_CSV_PATH}")

if __name__ == "__main__":
    main()
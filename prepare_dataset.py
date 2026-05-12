"""
prepare_dataset.py — VideoToFrames pipeline (run this ONCE before pipeline.py)
================================================================================
Extracts image frames from raw videos and saves them as organized directories.

  data/raw_videos/<type>/<video>.mp4
      → data/frames/<type>/<video_name>/000001.jpg
                                        000002.jpg
                                        ...
                                        fps.txt       ← actual FPS stored here

The main ML pipeline (pipeline.py) works entirely from data/frames/ after this.

Directory structure mirrors raw_videos/ exactly so all folder-label mappings
(normal / following / surrounding / fast_approach) are preserved automatically.

Usage:
    python prepare_dataset.py                          # all videos, native FPS
    python prepare_dataset.py --fps 15                 # downsample to 15 fps
    python prepare_dataset.py --resize 1280 720        # resize frames
    python prepare_dataset.py --quality 85             # JPEG quality (default 92)
    python prepare_dataset.py --input-dir path/to/vids # custom video source
    python prepare_dataset.py --out-dir  path/to/out   # custom output root
    python prepare_dataset.py --overwrite              # re-extract already done clips
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_VIDEOS_DIR, FRAMES_DIR, VIDEO_EXTENSIONS


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_video(
    video_path: Path,
    out_dir:    Path,
    target_fps: float | None,
    resize:     tuple[int, int] | None,
    quality:    int,
    overwrite:  bool,
) -> int:
    """
    Extract frames from one video into out_dir.
    Returns number of frames written (0 if skipped).
    """
    # Skip if already extracted (unless --overwrite)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        print(f"  [SKIP] Already extracted: {out_dir.name}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = target_fps if target_fps else src_fps

    # Frame step: keep every Nth frame to achieve target_fps
    step = max(1, round(src_fps / out_fps))
    actual_fps = src_fps / step

    written  = 0
    src_idx  = 0
    out_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if src_idx % step == 0:
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            fname = out_dir / f"{out_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            out_idx  += 1
            written  += 1

        src_idx += 1

    cap.release()

    # Write fps sidecar so the tracker knows the real frame rate
    (out_dir / "fps.txt").write_text(f"{actual_fps:.4f}\n")

    return written


# ── Batch discovery and processing ───────────────────────────────────────────

def run(
    input_dir:  Path,
    out_root:   Path,
    target_fps: float | None,
    resize:     tuple[int, int] | None,
    quality:    int,
    overwrite:  bool,
):
    videos = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No video files found under {input_dir}")
        print("Expected subfolders: normal/  threat/following/  threat/surrounding/  threat/fast_approach/")
        return

    print(f"Found {len(videos)} video(s) under {input_dir}")
    if target_fps:
        print(f"  Target FPS : {target_fps}")
    if resize:
        print(f"  Resize to  : {resize[0]}×{resize[1]}")
    print()

    total_frames = 0
    t0 = time.time()

    for video_path in videos:
        # Mirror the subdirectory structure from input_dir into out_root
        rel   = video_path.relative_to(input_dir)
        out_dir = out_root / rel.with_suffix("")   # strip .mp4 etc → becomes dir name

        print(f"  → {rel}  ({video_path.stat().st_size // 1024} KB)")
        n = extract_video(video_path, out_dir, target_fps, resize, quality, overwrite)
        if n:
            print(f"     wrote {n} frames to {out_dir}")
        total_frames += n

    elapsed = time.time() - t0
    print(f"\nDone. {total_frames} frames written in {elapsed:.1f}s")
    print(f"Frame dataset location: {out_root}")
    print("\nNext step:")
    print("  python pipeline.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract image frames from raw videos → data/frames/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_dataset.py                     # all videos, native FPS
  python prepare_dataset.py --fps 15            # downsample to 15 fps
  python prepare_dataset.py --resize 1280 720   # resize every frame
  python prepare_dataset.py --overwrite         # redo already-extracted clips
""",
    )
    parser.add_argument("--input-dir", default=str(RAW_VIDEOS_DIR),
                        help=f"Root folder of raw videos (default: {RAW_VIDEOS_DIR})")
    parser.add_argument("--out-dir",   default=str(FRAMES_DIR),
                        help=f"Output root for frame directories (default: {FRAMES_DIR})")
    parser.add_argument("--fps",       type=float, default=None,
                        help="Target FPS (default: keep source FPS)")
    parser.add_argument("--resize",    nargs=2, type=int, metavar=("W", "H"),
                        default=None,
                        help="Resize frames to W H (e.g. --resize 1280 720)")
    parser.add_argument("--quality",   type=int, default=92,
                        help="JPEG quality 1-100 (default: 92)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract clips that already have frames on disk")
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize else None

    run(
        input_dir  = Path(args.input_dir),
        out_root   = Path(args.out_dir),
        target_fps = args.fps,
        resize     = resize,
        quality    = args.quality,
        overwrite  = args.overwrite,
    )


if __name__ == "__main__":
    main()

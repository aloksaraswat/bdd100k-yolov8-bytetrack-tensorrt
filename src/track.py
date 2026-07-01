"""Run ByteTrack on top of the detector and save an annotated clip + GIF.

ByteTrack is tracking-by-detection: it associates boxes frame to frame and,
crucially, keeps the *low-confidence* detections as candidates during
association. That second association pass is what holds an ID through a brief
occlusion instead of spawning a new one when the box score dips. We use the
ByteTrack config that ships with Ultralytics so the tracker is reproducible.

    python src/track.py --weights weights/best.pt --source clip.mp4
"""
import argparse
from pathlib import Path


def make_gif(mp4_path, gif_path, max_frames=120, stride=2, width=640):
    """Best-effort GIF for the README. Skipped silently if imageio is absent."""
    try:
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image
    except Exception as e:
        print(f"(skipping GIF: {e})")
        return

    try:
        frames = []
        for i, frame in enumerate(iio.imiter(mp4_path)):
            if i % stride:
                continue
            img = Image.fromarray(frame)
            if img.width > width:
                img = img.resize((width, int(img.height * width / img.width)))
            frames.append(np.asarray(img))
            if len(frames) >= max_frames:
                break
        if frames:
            iio.imwrite(gif_path, frames, duration=80, loop=0)
            print(f"GIF: {gif_path}")
    except Exception as e:
        print(f"(GIF step skipped: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help=".pt or .engine")
    ap.add_argument("--source", required=True, help="video file or stream")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="track")
    ap.add_argument("--gif", default="assets/tracking_demo.gif")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        persist=True,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    # Drain the generator so Ultralytics writes the annotated video, and count
    # the unique track IDs we held across the clip.
    seen_ids, frames = set(), 0
    out_dir = None
    for r in results:
        frames += 1
        out_dir = Path(r.save_dir)
        if r.boxes is not None and r.boxes.id is not None:
            seen_ids.update(int(i) for i in r.boxes.id.tolist())

    print(f"Processed {frames} frames, {len(seen_ids)} unique track IDs.")

    if out_dir is not None:
        mp4s = list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.avi"))
        if mp4s:
            print(f"Annotated video: {mp4s[0]}")
            Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
            make_gif(str(mp4s[0]), args.gif)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one webcam RGB frame and pair it with an existing RGB-D template depth bundle."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--template-dir", default="captures/realsense_20260609_190015_filtered")
    parser.add_argument("--output-root", default="captures")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", action="store_true", help="Show a preview window and press s to save.")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture_frame(cap, warmup: int, preview: bool):
    frame = None
    for _ in range(max(0, warmup)):
        ok, frame = cap.read()
        if not ok:
            frame = None
        time.sleep(0.03)

    if preview:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Webcam frame read failed")
            cv2.imshow("webcam capture - press s to save, q to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord(" ")):
                cv2.destroyAllWindows()
                return frame
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                raise SystemExit("Canceled")

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Webcam frame read failed")
    return frame


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    template_dir = (project_root / args.template_dir).resolve()
    if not template_dir.exists():
        raise FileNotFoundError(f"Missing template dir: {template_dir}")

    depth_path = template_dir / "depth.png"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing template depth image: {depth_path}")
    depth_width, depth_height = Image.open(depth_path).size

    output_name = args.output_name or f"webcam_rgbd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (project_root / args.output_root / output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    cap = open_camera(args.camera_index, args.width, args.height)
    try:
        frame_bgr = capture_frame(cap, args.warmup, args.preview)
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if (frame_rgb.shape[1], frame_rgb.shape[0]) != (depth_width, depth_height):
        frame_rgb = cv2.resize(frame_rgb, (depth_width, depth_height), interpolation=cv2.INTER_AREA)
    Image.fromarray(frame_rgb).save(output_dir / "color.png")

    for name in ["depth.png", "workspace_mask.png", "meta.mat", "camera_info.json"]:
        source = template_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    print(output_dir)


if __name__ == "__main__":
    main()

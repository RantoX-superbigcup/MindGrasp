from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def parse_roi(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    values = [int(float(v.strip())) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have positive width and height")
    return x1, y1, x2, y2


def ensure_unique_dir(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    base = path
    for idx in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{idx:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory under {base.parent}")


def make_workspace_mask(depth: np.ndarray, roi: Optional[Tuple[int, int, int, int]], mode: str) -> np.ndarray:
    if mode == "all":
        mask = np.ones(depth.shape, dtype=np.uint8) * 255
    else:
        mask = (depth > 0).astype(np.uint8) * 255
    if roi is not None:
        x1, y1, x2, y2 = roi
        roi_mask = np.zeros_like(mask, dtype=np.uint8)
        roi_mask[y1:y2 + 1, x1:x2 + 1] = 255
        mask = np.where(roi_mask > 0, mask, 0).astype(np.uint8)
    return mask



def filter_depth_range(depth_u16: np.ndarray, depth_scale: float, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    valid = (depth_u16 > 0) & (depth_m >= float(min_depth_m)) & (depth_m <= float(max_depth_m))
    return np.where(valid, depth_u16, 0).astype(np.uint16)



def set_sensor_option(sensor, option, value, name: str) -> Optional[float]:
    if not sensor.supports(option):
        print(f"[RealSense] option not supported: {name}")
        return None
    option_range = sensor.get_option_range(option)
    clamped = max(float(option_range.min), min(float(option_range.max), float(value)))
    sensor.set_option(option, clamped)
    actual = float(sensor.get_option(option))
    print(f"[RealSense] {name}={actual}")
    return actual



def set_filter_option(block, option, value, name: str) -> None:
    try:
        option_range = block.get_option_range(option)
        clamped = max(float(option_range.min), min(float(option_range.max), float(value)))
        block.set_option(option, clamped)
        actual = float(block.get_option(option))
        print(f"[RealSense] filter {name}={actual}")
    except Exception as exc:
        print(f"[RealSense] skip filter option {name}: {exc}")


def configure_realsense_device(profile, args, rs) -> dict:
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    applied = {}

    if args.depth_preset != "none" and depth_sensor.supports(rs.option.visual_preset):
        preset_map = {
            "default": int(rs.rs400_visual_preset.default),
            "hand": int(rs.rs400_visual_preset.hand),
            "high_accuracy": int(rs.rs400_visual_preset.high_accuracy),
            "high_density": int(rs.rs400_visual_preset.high_density),
            "medium_density": int(rs.rs400_visual_preset.medium_density),
        }
        applied["depth_preset"] = set_sensor_option(depth_sensor, rs.option.visual_preset, preset_map[args.depth_preset], "visual_preset")

    if args.emitter != "keep":
        emitter_map = {"off": 0, "on": 1, "auto": 2}
        applied["emitter_enabled"] = set_sensor_option(depth_sensor, rs.option.emitter_enabled, emitter_map[args.emitter], "emitter_enabled")

    if args.laser_power is not None:
        applied["laser_power"] = set_sensor_option(depth_sensor, rs.option.laser_power, args.laser_power, "laser_power")

    if args.depth_exposure_us is not None:
        set_sensor_option(depth_sensor, rs.option.enable_auto_exposure, 0, "depth_auto_exposure")
        applied["depth_exposure_us"] = set_sensor_option(depth_sensor, rs.option.exposure, args.depth_exposure_us, "depth_exposure_us")
    elif args.depth_auto_exposure != "keep":
        applied["depth_auto_exposure"] = set_sensor_option(
            depth_sensor,
            rs.option.enable_auto_exposure,
            1 if args.depth_auto_exposure == "on" else 0,
            "depth_auto_exposure",
        )

    return applied


def create_depth_filters(args, rs):
    filters = []
    if not args.enable_filters:
        return filters

    threshold = rs.threshold_filter(float(args.min_depth_m), float(args.max_depth_m))
    filters.append(("threshold", threshold))

    spatial = rs.spatial_filter()
    set_filter_option(spatial, rs.option.filter_magnitude, args.spatial_magnitude, "spatial_magnitude")
    set_filter_option(spatial, rs.option.filter_smooth_alpha, args.spatial_alpha, "spatial_alpha")
    set_filter_option(spatial, rs.option.filter_smooth_delta, args.spatial_delta, "spatial_delta")
    set_filter_option(spatial, rs.option.holes_fill, args.spatial_holes_fill, "spatial_holes_fill")
    filters.append(("spatial", spatial))

    temporal = rs.temporal_filter()
    set_filter_option(temporal, rs.option.filter_smooth_alpha, args.temporal_alpha, "temporal_alpha")
    set_filter_option(temporal, rs.option.filter_smooth_delta, args.temporal_delta, "temporal_delta")
    filters.append(("temporal", temporal))

    if args.hole_filling:
        filters.append(("hole_filling", rs.hole_filling_filter(int(args.hole_filling_mode))))

    print("[RealSense] enabled filters: " + ", ".join(name for name, _ in filters))
    return filters


def get_aligned_frames(pipeline, align, filters):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    for _name, depth_filter in filters:
        depth_frame = depth_filter.process(depth_frame)
    return color_frame, depth_frame



def fuse_depth_frames(depth_frames: List[np.ndarray], mode: str) -> np.ndarray:
    if not depth_frames:
        raise RuntimeError("No depth frames to fuse")
    if len(depth_frames) == 1 or mode == "last":
        return depth_frames[-1].astype(np.uint16, copy=False)

    stack = np.stack(depth_frames, axis=0).astype(np.uint16, copy=False)
    valid = stack > 0
    valid_count = valid.sum(axis=0)

    if mode == "mean":
        summed = stack.astype(np.float32).sum(axis=0)
        mean = np.divide(summed, valid_count, out=np.zeros_like(summed), where=valid_count > 0)
        return np.where(valid_count > 0, np.rint(mean), 0).astype(np.uint16)

    if mode != "median":
        raise ValueError(f"Unsupported depth fusion mode: {mode}")

    # Zeros are invalid RealSense pixels. Sorting puts them first, so jump over
    # invalid values and take the median among valid depth samples per pixel.
    sorted_stack = np.sort(stack, axis=0)
    sample_count = stack.shape[0]
    median_index = sample_count - valid_count + (valid_count // 2)
    median_index = np.clip(median_index, 0, sample_count - 1).astype(np.int64)
    fused = np.take_along_axis(sorted_stack, median_index[None, ...], axis=0)[0]
    return np.where(valid_count > 0, fused, 0).astype(np.uint16)


def capture_depth_burst(pipeline, align, filters, frame_count: int, fusion_mode: str):
    frame_count = max(1, int(frame_count))
    depth_frames: List[np.ndarray] = []
    last_color = None
    color_intrinsics = None

    for _ in range(frame_count):
        color_frame, depth_frame = get_aligned_frames(pipeline, align, filters)
        if not color_frame or not depth_frame:
            continue
        last_color = np.asanyarray(color_frame.get_data()).copy()
        depth_frames.append(np.asanyarray(depth_frame.get_data()).copy())
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    if last_color is None or color_intrinsics is None or not depth_frames:
        raise RuntimeError("Failed to capture a valid RGB-D burst")
    fused_depth = fuse_depth_frames(depth_frames, fusion_mode)
    return last_color, fused_depth, color_intrinsics, len(depth_frames)


def save_rgbd_bundle(
    output_dir: Path,
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    color_intrinsics,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    workspace_mode: str,
    workspace_roi: Optional[Tuple[int, int, int, int]],
    metadata: dict,
) -> None:
    import cv2
    import scipy.io as scio

    output_dir.mkdir(parents=True, exist_ok=True)
    color_path = output_dir / "color.png"
    depth_path = output_dir / "depth.png"
    mask_path = output_dir / "workspace_mask.png"
    meta_path = output_dir / "meta.mat"
    info_path = output_dir / "camera_info.json"

    depth_u16 = filter_depth_range(depth_u16, depth_scale, min_depth_m, max_depth_m)
    workspace_mask = make_workspace_mask(depth_u16, workspace_roi, workspace_mode)
    intrinsic_matrix = np.array(
        [
            [float(color_intrinsics.fx), 0.0, float(color_intrinsics.ppx)],
            [0.0, float(color_intrinsics.fy), float(color_intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    factor_depth = np.array([[1.0 / float(depth_scale)]], dtype=np.float32)

    if not cv2.imwrite(str(color_path), color_bgr):
        raise RuntimeError(f"Failed to write {color_path}")
    if not cv2.imwrite(str(depth_path), depth_u16):
        raise RuntimeError(f"Failed to write {depth_path}")
    if not cv2.imwrite(str(mask_path), workspace_mask):
        raise RuntimeError(f"Failed to write {mask_path}")

    scio.savemat(
        str(meta_path),
        {
            "intrinsic_matrix": intrinsic_matrix,
            "factor_depth": factor_depth,
        },
    )

    info = {
        **metadata,
        "files": {
            "color": "color.png",
            "depth": "depth.png",
            "workspace_mask": "workspace_mask.png",
            "meta": "meta.mat",
        },
        "image_width": int(color_intrinsics.width),
        "image_height": int(color_intrinsics.height),
        "intrinsic_matrix": intrinsic_matrix.tolist(),
        "depth_scale_m_per_unit": float(depth_scale),
        "factor_depth": float(factor_depth[0, 0]),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "workspace_mode": workspace_mode,
        "workspace_roi": list(workspace_roi) if workspace_roi else None,
        "note": "Depth is aligned to color. GraspNet reads depth_m = depth_png / factor_depth.",
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def capture(args: argparse.Namespace) -> Optional[Path]:
    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install RealSense Python SDK first: pip install pyrealsense2 opencv-python scipy"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    applied_realsense_options = configure_realsense_device(profile, args, rs)
    depth_filters = create_depth_filters(args, rs)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "unknown"
    device_serial = device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else "unknown"

    print(f"[RealSense] device={device_name}, serial={device_serial}")
    print(f"[RealSense] depth_scale={depth_scale} m/unit, factor_depth={1.0 / depth_scale:.3f}")
    print(f"[RealSense] warming up {args.warmup} frames...")

    last_color = None
    last_depth = None
    color_intrinsics = None
    workspace_roi = parse_roi(args.workspace_roi)

    try:
        for _ in range(args.warmup):
            color_frame, depth_frame = get_aligned_frames(pipeline, align, depth_filters)
            if color_frame and depth_frame:
                last_color = np.asanyarray(color_frame.get_data())
                last_depth = np.asanyarray(depth_frame.get_data())
                color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        if args.preview:
            print("[preview] Press 's' to save one frame, 'q' to quit.")
            while True:
                color_frame, depth_frame = get_aligned_frames(pipeline, align, depth_filters)
                if not color_frame or not depth_frame:
                    continue
                last_color = np.asanyarray(color_frame.get_data())
                last_depth = np.asanyarray(depth_frame.get_data())
                color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

                depth_vis = cv2.convertScaleAbs(last_depth, alpha=args.depth_vis_alpha)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                shown = np.hstack((last_color, depth_vis))
                cv2.putText(shown, "left=color right=depth | s=save q=quit", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("RealSense RGB-D capture", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return None
                if key == ord("s"):
                    break
        else:
            color_frame, depth_frame = get_aligned_frames(pipeline, align, depth_filters)
            if not color_frame or not depth_frame:
                raise RuntimeError("Failed to capture aligned color/depth frames")
            last_color = np.asanyarray(color_frame.get_data())
            last_depth = np.asanyarray(depth_frame.get_data())
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        if args.depth_frames > 1 or args.depth_fusion != "last":
            print(f"[RealSense] capturing {args.depth_frames} depth frames for {args.depth_fusion} fusion...")
            last_color, last_depth, color_intrinsics, fused_frame_count = capture_depth_burst(
                pipeline,
                align,
                depth_filters,
                args.depth_frames,
                args.depth_fusion,
            )
        else:
            fused_frame_count = 1

        if last_color is None or last_depth is None or color_intrinsics is None:
            raise RuntimeError("No valid RGB-D frame captured")

        root = Path(args.output_dir)
        if args.timestamp:
            root = root / datetime.now().strftime("realsense_%Y%m%d_%H%M%S")
        output_dir = ensure_unique_dir(root, overwrite=args.overwrite)

        metadata = {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "device_name": device_name,
            "device_serial": device_serial,
            "stream_width": args.width,
            "stream_height": args.height,
            "stream_fps": args.fps,
            "realsense_options": applied_realsense_options,
            "depth_filters": [name for name, _ in depth_filters],
            "depth_fusion": args.depth_fusion,
            "depth_frames_requested": int(args.depth_frames),
            "depth_frames_used": int(fused_frame_count),
        }
        save_rgbd_bundle(
            output_dir=output_dir,
            color_bgr=last_color,
            depth_u16=last_depth,
            color_intrinsics=color_intrinsics,
            depth_scale=depth_scale,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            workspace_mode=args.workspace_mode,
            workspace_roi=workspace_roi,
            metadata=metadata,
        )
        print(f"[saved] {output_dir}")
        print("[next] run_target_grasp_demo.py --frame-dir " + str(output_dir))
        return output_dir
    finally:
        pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RealSense RGB-D frame in GraspNet-compatible format.")
    parser.add_argument("--output-dir", default="captures", help="Output directory. With --timestamp, a subfolder is created here.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", help="Optional RealSense serial number when multiple cameras are connected.")
    parser.add_argument("--warmup", type=int, default=60, help="Frames to skip before saving, for auto-exposure/filter stability.")
    parser.add_argument("--depth-frames", type=int, default=9, help="Depth frames to fuse after pressing save. Use 1 with --depth-fusion last for old single-frame behavior.")
    parser.add_argument("--depth-fusion", choices=["median", "mean", "last"], default="median", help="How to fuse the captured depth burst. median is best for static tabletop noise.")
    parser.add_argument("--depth-preset", choices=["none", "default", "hand", "high_accuracy", "high_density", "medium_density"], default="high_accuracy", help="D400 depth visual preset. high_accuracy is usually cleaner; high_density keeps more points.")
    parser.add_argument("--emitter", choices=["keep", "off", "on", "auto"], default="on", help="Depth projector/emitter setting.")
    parser.add_argument("--laser-power", type=float, default=360.0, help="D415 laser power, usually 0-360. Lower it if close-range surfaces are overexposed.")
    parser.add_argument("--depth-auto-exposure", choices=["keep", "on", "off"], default="on")
    parser.add_argument("--depth-exposure-us", type=float, help="Manual depth exposure in microseconds. If set, disables depth auto exposure.")
    parser.add_argument("--enable-filters", action=argparse.BooleanOptionalAction, default=True, help="Apply RealSense threshold/spatial/temporal/hole filling filters before saving.")
    parser.add_argument("--spatial-magnitude", type=float, default=2.0)
    parser.add_argument("--spatial-alpha", type=float, default=0.5)
    parser.add_argument("--spatial-delta", type=float, default=20.0)
    parser.add_argument("--spatial-holes-fill", type=float, default=2.0)
    parser.add_argument("--temporal-alpha", type=float, default=0.4)
    parser.add_argument("--temporal-delta", type=float, default=20.0)
    parser.add_argument("--hole-filling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hole-filling-mode", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--preview", action="store_true", help="Show color/depth preview; press s to save, q to quit.")
    parser.add_argument("--timestamp", action="store_true", help="Create a timestamped subfolder under --output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--workspace-mode", choices=["valid", "all"], default="valid", help="valid uses depth>0 as workspace; all uses the whole image.")
    parser.add_argument("--min-depth-m", type=float, default=0.30, help="Drop depth points closer than this distance in meters.")
    parser.add_argument("--max-depth-m", type=float, default=1.20, help="Drop depth points farther than this distance in meters. This removes RealSense 65535 invalid rays.")
    parser.add_argument("--workspace-roi", help="Optional workspace ROI x1,y1,x2,y2 applied to workspace_mask.png.")
    parser.add_argument("--depth-vis-alpha", type=float, default=0.03, help="Depth visualization scale for preview only.")
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()

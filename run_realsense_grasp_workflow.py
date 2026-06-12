from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_target_grasp_demo as target_demo
from capture_realsense_rgbd import capture as capture_realsense
legacy_arm = importlib.import_module("arm_control.grasp_to_arm")
from arm_control.arm_serial import ArmLink
from qwen_config import resolve_qwen_api_key, resolve_qwen_base_url, resolve_qwen_model


def image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_vector(text: str, length: int, name: str) -> np.ndarray:
    values = [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]
    if len(values) != length:
        raise ValueError(f"{name} must have {length} numeric values")
    return np.asarray(values, dtype=float)


def parse_matrix3(text: str, name: str = "matrix") -> np.ndarray:
    rows = []
    for row in str(text).split(";"):
        row = row.strip()
        if not row:
            continue
        rows.append([float(item.strip()) for item in row.split(",") if item.strip()])
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be 3 rows of 3 comma-separated values, separated by semicolons")
    return matrix


def configure_legacy_arm(args: argparse.Namespace) -> None:
    legacy_arm.R_bc = parse_matrix3(args.camera_to_base_rotation, "camera_to_base_rotation")
    legacy_arm.t_bc = parse_vector(args.camera_to_base_translation_m, 3, "camera_to_base_translation_m")
    legacy_arm.L1 = float(args.arm_l1_mm)
    legacy_arm.L2 = float(args.arm_l2_mm)


def call_qwen_readable_options(
    rgb_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    max_options: int,
    excluded_targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    import requests
    from PIL import Image

    width, height = Image.open(rgb_path).size
    key_text = "/".join(chr(ord("A") + i) for i in range(max_options))
    excluded_targets = [str(item).strip() for item in (excluded_targets or []) if str(item).strip()]
    exclude_text = ""
    if excluded_targets:
        exclude_text = "\n本轮必须排除这些上一轮已经展示过、用户表示不要的目标或同义物体：" + "、".join(excluded_targets[:24]) + "。不要再次输出它们。\n"
    prompt = f"""
你是具身抓取系统的目标选择界面生成器。请观察图片，列出桌面上清晰、独立、可抓取的物体，最多 {max_options} 个。
这些选项会给病人/操作者选择，所以必须可读、清楚、容易区分。{exclude_text}

请只输出严格 JSON，不要输出 Markdown，不要解释。图像宽高 width={width}, height={height}。
每个 bbox 用原图像素坐标 [x1, y1, x2, y2]，仅用于界面预览；真正定位会由 GroundingDINO/SAM 完成。

JSON 格式：
{{
  "question": "请选择你想抓取的目标",
  "options": [
    {{
      "key": "A",
      "label": "香蕉",
      "target_id": "banana",
      "description": "画面中部偏左的黄色弯曲香蕉",
      "grounding_prompts": ["yellow banana", "banana"],
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }}
  ]
}}

要求：
1. key 按 {key_text} 依次使用。
2. label 用简短中文名，target_id 用英文小写下划线，方便 GroundingDINO 定位。
3. description 用中文，12-30 字，写清楚颜色、位置、形状或相邻参照物，例如“右侧红色饼干盒”“左下橙黑色电钻”。
4. grounding_prompts 用 2-5 个英文短语，专门给 GroundingDINO 定位；不要只写自造 target_id。例：["green spiky toy", "toy case", "case"]、["black mug", "mug", "cup"]。
5. 不要把同一个物体重复列出；不要列桌面、机械臂、背景、阴影。
6. 如果 max_options=3，只输出最清楚的 3 个真实物体；不要输出“没有我想要的”，程序会自动添加 D 选项。
""".strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(rgb_path)}},
                ],
            }
        ],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    result = target_demo.extract_json(content)
    result["raw_content"] = content
    return normalize_readable_options(result, width, height, max_options)


def normalize_readable_options(result: Dict[str, Any], width: int, height: int, max_options: int) -> Dict[str, Any]:
    options = result.get("options", [])[:max_options]
    normalized: List[Dict[str, Any]] = []
    for idx, option in enumerate(options):
        bbox = target_demo.normalize_bbox(option.get("bbox"), width, height)
        if bbox is None:
            continue
        key = chr(ord("A") + len(normalized))
        label = str(option.get("label") or option.get("target_id") or f"目标{idx + 1}")
        target_id = str(option.get("target_id") or label).lower().replace(" ", "_")
        normalized_option = {
            "key": key,
            "label": label,
            "target_id": target_id,
            "description": str(option.get("description") or label),
            "bbox": bbox,
            "confidence": float(option.get("confidence", 0.0) or 0.0),
        }
        raw_prompts = option.get("grounding_prompts") or option.get("grounding_prompt")
        normalized_option["grounding_prompts"] = target_demo.build_grounding_prompts(normalized_option, extra_prompts=raw_prompts)
        normalized.append(normalized_option)
    return {
        "image_width": width,
        "image_height": height,
        "question": str(result.get("question") or "请选择你想抓取的目标"),
        "options": normalized,
        "raw_content": result.get("raw_content", ""),
    }


def print_options(options_result: Dict[str, Any]) -> None:
    print("\n" + options_result.get("question", "请选择目标"))
    print("-" * 72)
    for option in options_result.get("options", []):
        key = option["key"]
        label = option.get("label", "")
        target_id = option.get("target_id", "")
        description = option.get("description", "")
        print(f"{key}. {label} [{target_id}] - {description}")
    print("-" * 72)


def choose_interactively(options_result: Dict[str, Any]) -> str:
    keys = {str(option["key"]).upper() for option in options_result.get("options", [])}
    while True:
        choice = input("请输入目标选项字母，例如 A；输入 q 退出：").strip().upper()
        if choice == "Q":
            raise SystemExit("用户取消选择")
        if choice in keys:
            return choice
        print(f"无效选项：{choice}。可选：{', '.join(sorted(keys))}")



def normalize_place_options(result: Dict[str, Any], width: int, height: int, max_places: int) -> Dict[str, Any]:
    raw_options = result.get("places") or result.get("options") or []
    normalized: List[Dict[str, Any]] = []
    for idx, option in enumerate(raw_options[:max_places]):
        bbox = target_demo.normalize_bbox(option.get("bbox"), width, height)
        if bbox is None:
            continue
        key = str(option.get("key") or f"P{idx + 1}").upper().strip()
        if not key.startswith("P"):
            key = f"P{idx + 1}"
        normalized.append(
            {
                "key": key,
                "label": str(option.get("label") or f"place_{idx + 1}"),
                "target_id": str(option.get("target_id") or option.get("label") or f"place_{idx + 1}").lower().replace(" ", "_"),
                "description": str(option.get("description") or option.get("label") or f"place_{idx + 1}"),
                "bbox": bbox,
                "confidence": float(option.get("confidence", 0.0) or 0.0),
            }
        )
    return {
        "image_width": width,
        "image_height": height,
        "question": str(result.get("question") or "Choose a placement region"),
        "options": normalized,
        "raw_content": result.get("raw_content", ""),
    }


def create_place_context_image(frame_dir: Path, options_result: Dict[str, Any], selected: Dict[str, Any], output_path: Path) -> Path:
    from PIL import Image, ImageDraw

    color_path = frame_dir / "color.png"
    depth_path = frame_dir / "depth.png"
    color = Image.open(color_path).convert("RGB")
    width, height = color.size
    color_draw = ImageDraw.Draw(color)
    selected_bbox = selected.get("bbox")
    if selected_bbox:
        x1, y1, x2, y2 = [int(v) for v in selected_bbox]
        color_draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 40), width=4)
        color_draw.text((x1, max(0, y1 - 18)), "selected target", fill=(255, 60, 40))

    if depth_path.exists():
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        valid = depth > 0
        if np.any(valid):
            lo = float(np.percentile(depth[valid], 2))
            hi = float(np.percentile(depth[valid], 98))
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            gray = (norm * 255).astype(np.uint8)
            depth_rgb = np.stack([gray, 255 - gray, np.zeros_like(gray)], axis=-1)
            depth_rgb[~valid] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            depth_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth_img = Image.fromarray(depth_rgb, mode="RGB")
    else:
        depth_img = Image.new("RGB", (width, height), (0, 0, 0))

    context = Image.new("RGB", (width * 2, height), (20, 20, 20))
    context.paste(color, (0, 0))
    context.paste(depth_img, (width, 0))
    draw = ImageDraw.Draw(context)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(255, 255, 255), width=2)
    draw.rectangle([width, 0, width * 2 - 1, height - 1], outline=(255, 255, 255), width=2)
    draw.text((12, 12), "LEFT: RGB, bbox coords use this image only", fill=(255, 255, 0))
    draw.text((width + 12, 12), "RIGHT: depth preview, nearer/farther shown by color", fill=(255, 255, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context.save(output_path)
    return output_path


def call_qwen_place_options(
    context_path: Path,
    selected: Dict[str, Any],
    width: int,
    height: int,
    base_url: str,
    model: str,
    api_key: str,
    max_places: int,
) -> Dict[str, Any]:
    import requests

    prompt = f"""
You are selecting safe placement regions for a tabletop robotic arm.
The image is a side-by-side RGB-D context. LEFT is the RGB image, RIGHT is depth preview.
Return exactly {max_places} candidate empty tabletop placement regions.
All bbox coordinates must refer to the LEFT RGB image only, with width={width}, height={height}.
Avoid the selected target, occupied objects, robot body, table edges, shadows, and unreachable-looking corners.
Selected target: key={selected.get('key')}, label={selected.get('label')}, description={selected.get('description')}, bbox={selected.get('bbox')}.

Return strict JSON only:
{{
  "question": "Choose placement region",
  "places": [
    {{"key": "P1", "label": "left empty area", "description": "empty tabletop area left of objects", "bbox": [x1,y1,x2,y2], "confidence": 0.0}}
  ]
}}
""".strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(context_path)}},
                ],
            }
        ],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    result = target_demo.extract_json(content)
    result["raw_content"] = content
    return normalize_place_options(result, width, height, max_places)


def load_rgbd_geometry(frame_dir: Path) -> Dict[str, Any]:
    from PIL import Image

    info_path = frame_dir / "camera_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing camera_info.json: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    depth_path = frame_dir / "depth.png"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth image: {depth_path}")
    depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
    intrinsic = np.asarray(info["intrinsic_matrix"], dtype=float)
    depth_scale = float(info.get("depth_scale_m_per_unit") or (1.0 / float(info["factor_depth"])))
    return {"depth": depth, "intrinsic": intrinsic, "depth_scale": depth_scale, "info": info}


def point_from_depth_bbox(frame_dir: Path, bbox: List[int]) -> Dict[str, Any]:
    geom = load_rgbd_geometry(frame_dir)
    depth = geom["depth"]
    intrinsic = geom["intrinsic"]
    depth_scale = geom["depth_scale"]
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = target_demo.normalize_bbox(bbox, width, height) or bbox
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))

    # Use the central area first to avoid object/edge pixels in a large placement bbox.
    shrink_x = max(2, int((x2 - x1) * 0.25))
    shrink_y = max(2, int((y2 - y1) * 0.25))
    rx1 = min(max(0, x1 + shrink_x), width - 1)
    rx2 = min(max(rx1 + 1, x2 - shrink_x), width)
    ry1 = min(max(0, y1 + shrink_y), height - 1)
    ry2 = min(max(ry1 + 1, y2 - shrink_y), height)
    region = depth[ry1:ry2, rx1:rx2]
    valid = region[region > 0]
    if valid.size == 0:
        region = depth[max(0, y1):min(height, y2 + 1), max(0, x1):min(width, x2 + 1)]
        valid = region[region > 0]
    if valid.size == 0:
        raise RuntimeError(f"No valid depth inside placement bbox: {bbox}")

    z_m = float(np.median(valid) * depth_scale)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    ppx, ppy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x_m = (float(cx) - ppx) * z_m / fx
    y_m = (float(cy) - ppy) * z_m / fy
    return {
        "point_camera_m": [x_m, y_m, z_m],
        "uv": [cx, cy],
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "depth_m": z_m,
        "valid_depth_pixels": int(valid.size),
    }


def resolve_place_selection(
    frame_dir: Path,
    options_result: Dict[str, Any],
    target_choice: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    if args.place_mode == "off":
        return None

    selected_target = target_demo.choose_option(options_result, target_choice)
    width = int(options_result.get("image_width") or 0)
    height = int(options_result.get("image_height") or 0)
    if width <= 0 or height <= 0:
        from PIL import Image
        width, height = Image.open(frame_dir / "color.png").size

    if args.place_mode == "manual":
        if args.place_point_camera_m:
            point = parse_vector(args.place_point_camera_m, 3, "place_point_camera_m").tolist()
            place_result = {
                "selected_place": {
                    "key": args.place_choice or "P0",
                    "label": "manual_camera_point",
                    "description": "manual camera-frame point",
                    "bbox": None,
                },
                "point": {"point_camera_m": point, "uv": None, "bbox": None, "depth_m": point[2], "valid_depth_pixels": 0},
                "options": None,
                "mode": "manual_point",
            }
        elif args.place_bbox:
            bbox = target_demo.normalize_bbox([float(v.strip()) for v in args.place_bbox.split(",")], width, height)
            if bbox is None:
                raise ValueError("--place-bbox must be x1,y1,x2,y2")
            selected_place = {"key": args.place_choice or "P0", "label": "manual_bbox", "description": "manual placement bbox", "bbox": bbox}
            place_result = {
                "selected_place": selected_place,
                "point": point_from_depth_bbox(frame_dir, bbox),
                "options": {"question": "manual placement", "options": [selected_place], "image_width": width, "image_height": height},
                "mode": "manual_bbox",
            }
        else:
            raise ValueError("--place-mode manual needs --place-bbox or --place-point-camera-m")
    else:
        api_key = resolve_qwen_api_key(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"--place-mode qwen needs API key. Set {args.api_key_env} or configs/local_secrets.json")
        context_path = create_place_context_image(frame_dir, options_result, selected_target, run_dir / "place_context_rgbd.png")
        place_options = call_qwen_place_options(
            context_path=context_path,
            selected=selected_target,
            width=width,
            height=height,
            base_url=resolve_qwen_base_url(args.qwen_base_url),
            model=resolve_qwen_model(args.qwen_model),
            api_key=api_key,
            max_places=args.max_place_options,
        )
        target_demo.write_json(run_dir / "place_options.json", {"place_options": place_options})
        target_demo.draw_options_overlay(frame_dir / "color.png", place_options, run_dir / "place_options_overlay.png")
        print_options(place_options)
        place_choice = args.place_choice.upper().strip() if args.place_choice else choose_interactively(place_options)
        selected_place = target_demo.choose_option(place_options, place_choice)
        place_result = {
            "selected_place": selected_place,
            "point": point_from_depth_bbox(frame_dir, selected_place["bbox"]),
            "options": place_options,
            "mode": "qwen",
            "context_image": str(context_path),
        }

    target_demo.write_json(run_dir / "place_selection.json", place_result)
    return place_result


def build_capture_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(PROJECT_ROOT / args.capture_output_dir),
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        warmup=args.warmup,
        depth_frames=args.depth_frames,
        depth_fusion=args.depth_fusion,
        preview=not args.no_capture_preview,
        timestamp=True,
        overwrite=False,
        workspace_mode=args.workspace_mode,
        workspace_roi=args.workspace_roi,
        depth_vis_alpha=args.depth_vis_alpha,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        depth_preset=args.depth_preset,
        emitter=args.emitter,
        laser_power=args.laser_power,
        depth_auto_exposure=args.depth_auto_exposure,
        depth_exposure_us=args.depth_exposure_us,
        enable_filters=not args.no_enable_filters,
        spatial_magnitude=args.spatial_magnitude,
        spatial_alpha=args.spatial_alpha,
        spatial_delta=args.spatial_delta,
        spatial_holes_fill=args.spatial_holes_fill,
        temporal_alpha=args.temporal_alpha,
        temporal_delta=args.temporal_delta,
        hole_filling=not args.no_hole_filling,
        hole_filling_mode=args.hole_filling_mode,
    )


def build_target_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mask=None,
        bbox_ui=False,
        refine_qwen_bbox=False,
        api_key_env=args.api_key_env,
        qwen_base_url=args.qwen_base_url,
        qwen_model=args.qwen_model,
        localizer=args.localizer,
        allow_qwen_bbox_grasp=False,
        target_prompt=args.target_prompt,
        groundingdino_config=str((PROJECT_ROOT / args.groundingdino_config).resolve()),
        groundingdino_checkpoint=str((PROJECT_ROOT / args.groundingdino_checkpoint).resolve()),
        grounding_box_threshold=args.grounding_box_threshold,
        grounding_text_threshold=args.grounding_text_threshold,
        device=args.device,
        segmenter=args.segmenter,
        sam_checkpoint=str((PROJECT_ROOT / args.sam_checkpoint).resolve()),
        sam_model_type=args.sam_model_type,
        grabcut_iter=5,
        no_save_overlays=False,
        output_dir=str(output_dir),
    )


def build_pose_output(summary: Dict[str, Any], camera_frame_id: str) -> Dict[str, Any]:
    grasps = summary.get("top_grasps", [])
    selected = summary.get("selected_option", {})
    if not grasps:
        return {
            "schema_version": "intentgrasp.pose.v1",
            "status": "no_grasp",
            "target": selected,
            "reason": "No target grasp remained after filtering.",
        }
    best = grasps[0]
    return {
        "schema_version": "intentgrasp.pose.v1",
        "status": "pose_ready_camera_frame",
        "target": {
            "key": selected.get("key"),
            "label": selected.get("label"),
            "target_id": selected.get("target_id"),
            "description": selected.get("description"),
        },
        "grasp_pose": {
            "frame_id": camera_frame_id,
            "position_m": best["translation"],
            "quaternion_xyzw": best["quaternion"],
            "rotation_matrix": best["rotation_matrix"],
        },
        "gripper": {
            "width_m": best.get("width"),
            "depth_m": best.get("depth"),
        },
        "score": best.get("score"),
        "projected_uv": best.get("projected_uv"),
        "integration_note": "This is a camera-frame grasp pose. Before driving a real arm, apply hand-eye calibration T_base_camera and robot IK/planning.",
    }



def build_arm_command_output(pose_output: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.arm_mode == "off":
        return {
            "schema_version": "intentgrasp.arm_command.v1",
            "status": "disabled",
            "reason": "arm_mode=off",
        }
    if pose_output.get("status") != "pose_ready_camera_frame":
        return {
            "schema_version": "intentgrasp.arm_command.v1",
            "status": "no_pose",
            "reason": pose_output.get("reason", pose_output.get("status", "unknown")),
        }
    grasp_pose = pose_output.get("grasp_pose", {})
    configure_legacy_arm(args)
    legacy_output = legacy_arm.grasp_to_arm(
        grasp_pose["position_m"],
        grasp_pose["quaternion_xyzw"],
        standoff_mm=float(args.arm_standoff_mm),
        approach_axis=int(args.arm_approach_axis),
    )
    reachable = bool(legacy_output.get("reachable"))
    return {
        "schema_version": "intentgrasp.arm_command.v1",
        "status": "ready_to_send" if reachable else "not_reachable",
        "packet": str(legacy_output.get("packet") or ""),
        "reachable": reachable,
        "reason": str(legacy_output.get("reason") or ("ok" if reachable else "not_reachable")),
        "command": {
            "r_mm": float(legacy_output.get("r_mm", 0.0)),
            "h_mm": float(legacy_output.get("h_mm", 0.0)),
            "yaw_deg": float(legacy_output.get("yaw_deg", 0.0)),
            "elbow": int(legacy_output.get("elbow", 1)),
            "preferred_elbow": int(legacy_output.get("preferred_elbow", legacy_output.get("elbow", 1))),
        },
        "kinematics": {
            "l1_mm": float(legacy_arm.L1),
            "l2_mm": float(legacy_arm.L2),
            "joint_model": str(legacy_output.get("joint_model") or "arm_control_v3"),
            "joint_angles": legacy_output.get("joint_angles"),
            "ik_candidates": legacy_output.get("ik_candidates"),
            "yaw_min_deg": float(args.arm_yaw_min_deg),
            "yaw_max_deg": float(args.arm_yaw_max_deg),
        },
        "transform": {
            "camera_to_base_R": np.asarray(legacy_arm.R_bc, dtype=float).tolist(),
            "camera_to_base_t_m": np.asarray(legacy_arm.t_bc, dtype=float).tolist(),
            "camera_mount": args.camera_mount,
            "capture_pose": {
                "angle1_deg": float(args.capture_angle1_deg),
                "angle2_deg": float(args.capture_angle2_deg),
                "camera_height_m": float(args.capture_camera_height_m),
                "camera_forward_offset_m": float(args.capture_camera_forward_offset_m),
                "camera_pitch_down_deg": float(args.capture_camera_pitch_down_deg),
            },
            "standoff_mm": float(args.arm_standoff_mm),
            "approach_axis": int(args.arm_approach_axis),
        },
        "converter": "arm_control.grasp_to_arm.grasp_to_arm",
        "serial_adapter": "arm_control.arm_serial.ArmLink",
        "firmware_protocol": "<C r;h;yaw;elbow>",
        "safety_note": "This command uses arm_control/grasp_to_arm.py and arm_control/arm_serial.py. Validate calibration before arm-mode=serial.",
        "mode": args.arm_mode,
        "source_pose_status": pose_output.get("status"),
    }


def execute_arm_command_if_needed(arm_output: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.arm_mode != "serial":
        return {
            "mode": args.arm_mode,
            "sent": False,
            "reason": "serial execution disabled",
        }
    if not arm_output.get("reachable"):
        return {
            "mode": "serial",
            "sent": False,
            "success": False,
            "reason": f"command not reachable: {arm_output.get('reason')}",
        }
    packet = str(arm_output.get("packet") or "")
    if not packet:
        return {
            "mode": "serial",
            "sent": False,
            "success": False,
            "reason": "empty packet",
        }

    arm = ArmLink(
        port=args.arm_port,
        baud=args.arm_baud,
        disable_reset=args.arm_disable_reset,
        ready_timeout=args.arm_ready_timeout,
    )
    try:
        ok, log = arm.send_and_wait(packet, timeout=args.arm_timeout)
    finally:
        arm.close()
    return {
        "mode": "serial",
        "sent": True,
        "success": bool(ok),
        "packet": packet,
        "port": args.arm_port,
        "baud": int(args.arm_baud),
        "log": log,
        "reason": "done" if ok else "error_or_timeout",
    }



def build_pick_place_plan(
    pose_output: Dict[str, Any],
    arm_output: Dict[str, Any],
    place_result: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    if place_result is None:
        return None
    return {
        "schema_version": "intentgrasp.pick_place_plan.v1",
        "status": "unsupported",
        "reason": "pick/place planning previously depended on transform.py base-point conversion; current firmware path is limited to grasp_to_arm.py single grasp packets.",
        "selected_place": place_result.get("selected_place"),
        "place_point": place_result.get("point"),
        "steps": [],
        "safety_note": "Use arm_control/grasp_to_arm.py and arm_control/arm_serial.py for single grasp execution first.",
    }


def execute_packet_plan_if_needed(plan: Optional[Dict[str, Any]], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if plan is None:
        return None
    if args.arm_mode != "serial":
        return {
            "mode": args.arm_mode,
            "sent": False,
            "reason": "serial execution disabled",
        }
    if plan.get("status") != "ready_to_send":
        return {
            "mode": "serial",
            "sent": False,
            "success": False,
            "reason": plan.get("reason", "plan not ready"),
        }

    logs: List[Dict[str, Any]] = []
    arm = ArmLink(
        port=args.arm_port,
        baud=args.arm_baud,
        disable_reset=args.arm_disable_reset,
        ready_timeout=args.arm_ready_timeout,
    )
    try:
        for step in plan.get("steps", []):
            packet = step.get("packet")
            if not packet:
                continue
            done_key = "Gripper set" if step.get("kind") == "gripper" else "Traj done"
            ok, log = arm.send_and_wait(packet, done_key=done_key, timeout=args.arm_timeout)
            logs.append({"step": step.get("name"), "packet": packet, "success": bool(ok), "log": log})
            if not ok:
                return {"mode": "serial", "sent": True, "success": False, "reason": f"step failed: {step.get('name')}", "log": logs}
    finally:
        arm.close()
    return {"mode": "serial", "sent": True, "success": True, "reason": "done", "log": logs}


def run_grasp_pipeline(
    frame_dir: Path,
    options_result: Dict[str, Any],
    choice: str,
    args: argparse.Namespace,
    run_dir: Path,
    place_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_demo.prepare_huggingface_cache(PROJECT_ROOT)
    target_demo.prepare_windows_dll_paths()

    selected = target_demo.choose_option(options_result, choice)
    target_args = build_target_args(args, run_dir)
    target_region = target_demo.resolve_target_region(frame_dir / "color.png", selected, target_args, run_dir)
    selected_overlay_result = {
        **options_result,
        "options": [{**selected, "bbox": target_region["bbox"]}],
    }
    target_demo.draw_options_overlay(
        frame_dir / "color.png",
        selected_overlay_result,
        run_dir / "target_overlay.png",
        selected_key=selected["key"],
        target_mask=target_region["mask"],
    )

    graspnet_root = (PROJECT_ROOT / args.graspnet_root).resolve()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    target_demo.add_graspnet_paths(graspnet_root)

    net, _device = target_demo.get_net(checkpoint_path, args.num_view)
    end_points, full_cloud, target_cloud, intrinsic = target_demo.load_frame(frame_dir, target_region["mask"], args.num_point)
    gg = target_demo.get_grasps(net, end_points)
    if args.collision_thresh > 0:
        gg = target_demo.run_collision_detection(gg, full_cloud, args.collision_thresh, args.voxel_size)
    gg = target_demo.safe_nms(gg)
    gg.sort_by_score()
    filtered, filtered_uv, filter_mode = target_demo.filter_grasps_by_region(
        gg,
        target_region["bbox"],
        target_region["mask"],
        intrinsic,
        pad=args.bbox_pad,
        fallback_nearest=False,
    )
    filtered.sort_by_score()

    summary = {
        "selected_option": selected,
        "target_region": {
            "bbox": target_region["bbox"],
            "source": target_region["source"],
            "mask_path": target_region["mask_path"],
            "overlay_path": target_region["overlay_path"],
            "metadata": target_region["metadata"],
        },
        "filter_mode": filter_mode,
        "total_grasps_after_collision_nms": len(gg),
        "target_grasps": len(filtered),
        "top_grasps": target_demo.summarize_grasps(filtered, filtered_uv, args.top_k),
    }
    pose_output = build_pose_output(summary, args.camera_frame_id)
    arm_output = build_arm_command_output(pose_output, args)
    pick_place_plan = build_pick_place_plan(pose_output, arm_output, place_result, args)
    result = {
        "frame_dir": str(frame_dir),
        "options": options_result,
        "grasp_summary": summary,
        "robot_pose_output": pose_output,
        "arm_command": arm_output,
        "place_selection": place_result,
        "pick_place_plan": pick_place_plan,
    }
    target_demo.write_json(run_dir / "workflow_result.json", result)
    target_demo.write_json(run_dir / "grasp_pose.json", pose_output)
    target_demo.write_json(run_dir / "arm_command.json", arm_output)
    if pick_place_plan is not None:
        target_demo.write_json(run_dir / "pick_place_plan.json", pick_place_plan)
    target_demo.write_json(run_dir / "target_grasps.json", summary)

    if not args.no_vis:
        if args.vis_mode == "compare":
            target_demo.visualize_compare(full_cloud, target_cloud, filtered, args.top_k)
        else:
            target_demo.visualize(target_cloud, filtered, args.top_k)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command RealSense -> Qwen options -> GroundingDINO/SAM -> GraspNet pose workflow.")
    parser.add_argument("--frame-dir", help="Use an existing RGB-D folder instead of capturing a new RealSense frame.")
    parser.add_argument("--capture-output-dir", default="captures")
    parser.add_argument("--workflow-output-dir", default="outputs/realsense_workflow")
    parser.add_argument("--choice", help="Target option key. If omitted, ask interactively after Qwen options are shown.")
    parser.add_argument("--options-json", help="Reuse saved options JSON instead of calling Qwen.")
    parser.add_argument("--max-options", type=int, default=3)
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-model")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--depth-frames", type=int, default=9)
    parser.add_argument("--depth-fusion", choices=["median", "mean", "last"], default="median")
    parser.add_argument("--no-capture-preview", action="store_true")
    parser.add_argument("--workspace-mode", choices=["valid", "all"], default="valid")
    parser.add_argument("--workspace-roi")
    parser.add_argument("--depth-vis-alpha", type=float, default=0.03)
    parser.add_argument("--min-depth-m", type=float, default=0.30)
    parser.add_argument("--max-depth-m", type=float, default=1.20)
    parser.add_argument("--depth-preset", choices=["none", "default", "hand", "high_accuracy", "high_density", "medium_density"], default="high_accuracy")
    parser.add_argument("--emitter", choices=["keep", "off", "on", "auto"], default="on")
    parser.add_argument("--laser-power", type=float, default=360.0)
    parser.add_argument("--depth-auto-exposure", choices=["keep", "on", "off"], default="on")
    parser.add_argument("--depth-exposure-us", type=float)
    parser.add_argument("--no-enable-filters", action="store_true")
    parser.add_argument("--spatial-magnitude", type=float, default=2.0)
    parser.add_argument("--spatial-alpha", type=float, default=0.5)
    parser.add_argument("--spatial-delta", type=float, default=20.0)
    parser.add_argument("--spatial-holes-fill", type=float, default=2.0)
    parser.add_argument("--temporal-alpha", type=float, default=0.4)
    parser.add_argument("--temporal-delta", type=float, default=20.0)
    parser.add_argument("--no-hole-filling", action="store_true")
    parser.add_argument("--hole-filling-mode", type=int, default=1, choices=[0, 1, 2])

    parser.add_argument("--localizer", choices=["groundingdino"], default="groundingdino")
    parser.add_argument("--segmenter", choices=["sam"], default="sam")
    parser.add_argument("--target-prompt")
    parser.add_argument("--groundingdino-config", default="weights/groundingdino/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino-checkpoint", default="weights/groundingdino/groundingdino_swint_ogc.pth")
    parser.add_argument("--grounding-box-threshold", type=float, default=0.35)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.25)
    parser.add_argument("--sam-checkpoint", default="weights/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--graspnet-root", default="third_party/graspnet-baseline")
    parser.add_argument("--checkpoint", default="weights/graspnet/checkpoint-rs.tar")
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bbox-pad", type=int, default=20)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--camera-frame-id", default="camera_color_optical_frame")

    parser.add_argument("--arm-mode", choices=["off", "command", "serial"], default="command", help="off=no arm output, command=write arm_command.json only, serial=send to Arduino after pose generation.")
    parser.add_argument("--arm-port", default="COM3")
    parser.add_argument("--arm-baud", type=int, default=115200)
    parser.add_argument("--arm-disable-reset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--arm-ready-timeout", type=float, default=3.0)
    parser.add_argument("--arm-timeout", type=float, default=8.0)
    parser.add_argument("--arm-standoff-mm", type=float, default=40.0)
    parser.add_argument("--arm-approach-axis", type=int, default=0)
    parser.add_argument("--arm-l1-mm", type=float, default=130.0)
    parser.add_argument("--arm-l2-mm", type=float, default=200.0)
    parser.add_argument("--arm-yaw-min-deg", type=float, default=-90.0)
    parser.add_argument("--arm-yaw-max-deg", type=float, default=90.0)
    parser.add_argument("--camera-to-base-rotation", default="0,-0.70710678,0.70710678;-1,0,0;0,-0.70710678,-0.70710678", help="3x3 R_bc as rows separated by ';'. Default uses point-A camera pose: x=0.04m, z=0.16m, optical axis forward/down 45 deg.")
    parser.add_argument("--camera-to-base-translation-m", default="0.04,0,0.16", help="Camera origin in robot base frame, meters: x,y,z. Default is 40 mm in front of point A and 160 mm above the base pivot.")
    parser.add_argument("--camera-mount", default="front_of_point_a")
    parser.add_argument("--capture-angle1-deg", type=float, default=90.0)
    parser.add_argument("--capture-angle2-deg", type=float, default=0.0)
    parser.add_argument("--capture-camera-height-m", type=float, default=0.16)
    parser.add_argument("--capture-camera-forward-offset-m", type=float, default=0.04)
    parser.add_argument("--capture-camera-pitch-down-deg", type=float, default=45.0)

    parser.add_argument("--place-mode", choices=["off", "qwen", "manual"], default="off", help="off=no placement planning, qwen=ask Qwen for 3 placement regions, manual=use --place-bbox or --place-point-camera-m.")
    parser.add_argument("--place-choice", help="Placement option key, e.g. P1. If omitted in qwen mode, ask interactively.")
    parser.add_argument("--max-place-options", type=int, default=3)
    parser.add_argument("--place-bbox", help="Manual placement bbox in RGB pixels: x1,y1,x2,y2.")
    parser.add_argument("--place-point-camera-m", help="Manual placement point in camera frame meters: x,y,z.")
    parser.add_argument("--place-hover-mm", type=float, default=60.0)
    parser.add_argument("--place-release-height-mm", type=float, default=20.0)
    parser.add_argument("--post-grasp-lift-mm", type=float, default=60.0)
    parser.add_argument("--place-elbow", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--gripper-open-deg", type=float, default=60.0)
    parser.add_argument("--gripper-close-deg", type=float, default=120.0)

    parser.add_argument("--vis-mode", choices=["target", "compare"], default="compare")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.workflow_output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_dir:
        frame_dir = (PROJECT_ROOT / args.frame_dir).resolve()
    else:
        frame_dir = capture_realsense(build_capture_args(args))
        if frame_dir is None:
            raise SystemExit("未保存采集帧，流程结束。")
        frame_dir = Path(frame_dir).resolve()

    rgb_path = frame_dir / "color.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {rgb_path}")

    if args.options_json:
        raw = json.loads((PROJECT_ROOT / args.options_json).read_text(encoding="utf-8"))
        options_result = raw.get("qwen_options", raw)
    else:
        api_key = resolve_qwen_api_key(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set {args.api_key_env} or configs/local_secrets.json")
        print("[Qwen] 正在生成可读目标选项...")
        options_result = call_qwen_readable_options(
            rgb_path,
            resolve_qwen_base_url(args.qwen_base_url),
            resolve_qwen_model(args.qwen_model),
            api_key,
            args.max_options,
        )

    target_demo.write_json(run_dir / "qwen_options.json", {"qwen_options": options_result})
    target_demo.draw_options_overlay(rgb_path, options_result, run_dir / "qwen_options_overlay.png")
    print_options(options_result)
    choice = args.choice.upper().strip() if args.choice else choose_interactively(options_result)
    place_result = resolve_place_selection(frame_dir, options_result, choice, args, run_dir)

    print(f"[workflow] selected target: {choice}")
    result = run_grasp_pipeline(frame_dir, options_result, choice, args, run_dir, place_result=place_result)
    if result.get("pick_place_plan") is not None:
        arm_execution = execute_packet_plan_if_needed(result.get("pick_place_plan"), args)
    else:
        arm_execution = execute_arm_command_if_needed(result.get("arm_command", {}), args)
    result["arm_execution"] = arm_execution
    target_demo.write_json(run_dir / "workflow_result.json", result)
    target_demo.write_json(run_dir / "arm_execution.json", arm_execution or {})

    pose_path = run_dir / "grasp_pose.json"
    arm_path = run_dir / "arm_command.json"
    pick_place_path = run_dir / "pick_place_plan.json"
    print()
    print("[done] workflow finished")
    print(f"[saved] {run_dir}")
    print(f"[pose]  {pose_path}")
    print(f"[arm]   {arm_path}")
    if result.get("pick_place_plan") is not None:
        print(f"[plan]  {pick_place_path}")
    print(json.dumps(result["robot_pose_output"], ensure_ascii=False, indent=2))
    print(json.dumps(result.get("arm_command", {}), ensure_ascii=False, indent=2))
    if result.get("pick_place_plan") is not None:
        print(json.dumps(result.get("pick_place_plan", {}), ensure_ascii=False, indent=2))
    if args.arm_mode == "serial":
        print(json.dumps(arm_execution, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()

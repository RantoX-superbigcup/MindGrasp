from __future__ import annotations

import argparse
import base64
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


def image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def call_qwen_readable_options(
    rgb_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    max_options: int,
) -> Dict[str, Any]:
    import requests
    from PIL import Image

    width, height = Image.open(rgb_path).size
    key_text = "/".join(chr(ord("A") + i) for i in range(max_options))
    prompt = f"""
你是具身抓取系统的目标选择界面生成器。请观察图片，列出桌面上清晰、独立、可抓取的物体，最多 {max_options} 个。
这些选项会给病人/操作者选择，所以必须可读、清楚、容易区分。

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
6. 如果画面中有 7 个可抓取物体，就输出 7 个，不要只输出 4 个。
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
        key = str(option.get("key") or chr(ord("A") + idx)).upper()[:1]
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


def run_grasp_pipeline(
    frame_dir: Path,
    options_result: Dict[str, Any],
    choice: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Any]:
    target_demo.prepare_huggingface_cache(PROJECT_ROOT)
    target_demo.prepare_windows_dll_paths()

    selected = target_demo.choose_option(options_result, choice)
    target_args = build_target_args(args, run_dir)
    target_region = target_demo.resolve_target_region(frame_dir / "color.png", selected, target_args, run_dir)
    target_demo.draw_options_overlay(
        frame_dir / "color.png",
        options_result,
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
    result = {
        "frame_dir": str(frame_dir),
        "options": options_result,
        "grasp_summary": summary,
        "robot_pose_output": pose_output,
    }
    target_demo.write_json(run_dir / "workflow_result.json", result)
    target_demo.write_json(run_dir / "grasp_pose.json", pose_output)
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
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--qwen-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--qwen-model", default="qwen3-vl-flash")

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
        api_key = os.getenv(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key. Set it first: $env:{args.api_key_env}=\"your_key\"")
        print("[Qwen] 正在生成可读目标选项...")
        options_result = call_qwen_readable_options(rgb_path, args.qwen_base_url, args.qwen_model, api_key, args.max_options)

    target_demo.write_json(run_dir / "qwen_options.json", {"qwen_options": options_result})
    target_demo.draw_options_overlay(rgb_path, options_result, run_dir / "qwen_options_overlay.png")
    print_options(options_result)
    choice = args.choice.upper().strip() if args.choice else choose_interactively(options_result)

    print(f"[workflow] 选择目标：{choice}")
    result = run_grasp_pipeline(frame_dir, options_result, choice, args, run_dir)
    pose_path = run_dir / "grasp_pose.json"
    print("\n[done] 完整流程结束")
    print(f"[saved] {run_dir}")
    print(f"[pose]  {pose_path}")
    print(json.dumps(result["robot_pose_output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

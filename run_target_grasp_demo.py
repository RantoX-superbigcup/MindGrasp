from __future__ import annotations

"""
Standalone target-grasp demo.

Workflow:
1. Qwen-VL reads the RGB image and generates A/B/C/D object options.
2. The selected object is localized by Qwen bbox or optional GroundingDINO.
3. Optional SAM turns the selected bbox into a target mask.
4. GraspNet predicts scene-level grasp candidates from RGB-D.
5. This script filters candidates to the selected target region and only prints/visualizes target grasps.

This script intentionally does not start BCI, planner, serial control, or the physical robot.
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qwen_config import resolve_qwen_api_key, resolve_qwen_base_url, resolve_qwen_model

_GROUNDINGDINO_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def add_local_vision_paths(root: Optional[Path] = None) -> None:
    project_root = root or Path(__file__).resolve().parent
    for path in [
        project_root / "third_party" / "GroundingDINO",
        project_root / "third_party" / "segment-anything",
    ]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def prepare_huggingface_cache(root: Path) -> None:
    cache_dir = root / "weights" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def prepare_windows_dll_paths() -> None:
    if os.name != "nt":
        return
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cuda_bin = Path(cuda_home) / "bin"
        if cuda_bin.exists():
            os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(cuda_bin))
            except (AttributeError, OSError):
                pass
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(torch_lib))
            except (AttributeError, OSError):
                pass
    except Exception:
        pass


def image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_qwen_for_options(
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
你是辅助抓取系统的视觉理解模块。请观察图片，尽量列出桌面上所有清晰、可抓取、适合作为用户选择目标的独立物体，最多 {max_options} 个。不要只给 4 个；如果桌面上有 7 个可抓取物体，就输出 7 个。

请只输出严格 JSON，不要输出 Markdown，不要解释。bbox 必须使用原图像素坐标，左上角为原点，格式为 [x1, y1, x2, y2]。
图像宽高为 width={width}, height={height}。

JSON 格式：
{{
  "image_width": {width},
  "image_height": {height},
  "question": "请选择你想抓取的物体",
  "options": [
    {{"key": "A", "label": "物体中文名", "target_id": "short_english_id", "description": "中文位置描述", "grounding_prompts": ["english object phrase", "simple object name"], "bbox": [x1, y1, x2, y2], "confidence": 0.0}}
  ]
}}

要求：
1. key 按 {key_text} 依次使用。
2. label 用中文，target_id 用英文小写下划线。
3. 每个 bbox 必须尽量紧贴“单个独立物体”的可见区域，不要把相邻物体、桌面大面积背景、机械臂或整张桌面框进去。
4. 对香蕉、勺子、线缆这类细长/弯曲物体，bbox 只包住可见主体，不要为了矩形方便把旁边瓶子、碗、盒子一起框进去。
5. 只选择真实可见且适合抓取的物体。
6. grounding_prompts 必须是 2-5 个英文短语，给 GroundingDINO/SAM 定位用，例如 ["yellow banana", "banana"]、["green spiky toy", "toy case", "case"]。
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
    result = extract_json(content)
    result["raw_content"] = content
    return normalize_options(result, width, height, max_options)



def call_qwen_refine_bbox(
    rgb_path: Path,
    selected: Dict[str, Any],
    base_url: str,
    model: str,
    api_key: str,
) -> Tuple[List[int], Dict[str, Any]]:
    import requests
    from PIL import Image

    width, height = Image.open(rgb_path).size
    label = selected.get("label") or selected.get("target_id") or "selected object"
    target_id = selected.get("target_id") or label
    current_bbox = selected.get("bbox")
    prompt = f"""
你是机械臂抓取系统的目标精定位模块。用户已经选择目标：{label}，target_id={target_id}。
请重新观察整张图片，只为这个目标输出一个更准确、更紧的 bbox。

当前粗 bbox 是 {current_bbox}，但它可能包含了相邻物体。你必须主动排除其它物体、桌面背景、机械臂和遮挡物。
如果目标是香蕉，只框黄色香蕉可见主体，不要框住旁边白色瓶子、蓝色碗、红色盒子或电钻。

请只输出严格 JSON，不要输出 Markdown，不要解释。图像宽高 width={width}, height={height}。
JSON 格式：
{{
  "target_id": "{target_id}",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0,
  "note": "short reason"
}}
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
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    result = extract_json(content)
    bbox = normalize_bbox(result.get("bbox"), width, height)
    if bbox is None:
        raise RuntimeError(f"Qwen refine returned invalid bbox: {content}")
    result["bbox"] = bbox
    result["raw_content"] = content
    return bbox, result


def normalize_options(result: Dict[str, Any], width: int, height: int, max_options: int) -> Dict[str, Any]:
    options = result.get("options", [])[:max_options]
    normalized = []
    for idx, option in enumerate(options):
        key = str(option.get("key") or chr(ord("A") + idx)).upper()[:1]
        bbox = normalize_bbox(option.get("bbox"), width, height)
        if bbox is None:
            continue
        label = str(option.get("label") or option.get("target_id") or f"object_{idx + 1}")
        target_id = str(option.get("target_id") or f"object_{idx + 1}").lower().replace(" ", "_")
        description = str(option.get("description") or label)
        normalized_option = {
            "key": key,
            "label": label,
            "target_id": target_id,
            "description": description,
            "bbox": bbox,
            "confidence": float(option.get("confidence", 0.0) or 0.0),
        }
        raw_prompts = option.get("grounding_prompts") or option.get("grounding_prompt")
        normalized_option["grounding_prompts"] = build_grounding_prompts(normalized_option, extra_prompts=raw_prompts)
        normalized.append(normalized_option)
    result["image_width"] = width
    result["image_height"] = height
    result["question"] = result.get("question") or "??????????"
    result["options"] = normalized
    return result

def normalize_bbox(value: Any, width: int, height: int) -> Optional[List[int]]:
    if value is None:
        return None
    if len(value) != 4:
        return None
    arr = [float(v) for v in value]
    if max(arr) <= 1.5:
        arr = [arr[0] * width, arr[1] * height, arr[2] * width, arr[3] * height]
    elif max(arr[0], arr[2]) <= 1000 and max(arr[1], arr[3]) <= 1000:
        # Some VLM responses use a 0-1000 coordinate grid even when the prompt
        # asks for pixel coordinates. Rescale only axes that exceed the image.
        if max(arr[0], arr[2]) > width:
            arr[0] = arr[0] * width / 1000.0
            arr[2] = arr[2] * width / 1000.0
        if max(arr[1], arr[3]) > height:
            arr[1] = arr[1] * height / 1000.0
            arr[3] = arr[3] * height / 1000.0
    x1, y1, x2, y2 = arr
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]



def normalize_grounding_prompts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,;??\n]+", value)
    elif isinstance(value, dict):
        raw_items = value.values()
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]

    prompts: List[str] = []
    for item in raw_items:
        phrase = str(item or "").replace("_", " ").strip().strip(".?")
        phrase = re.sub(r"\s+", " ", phrase)
        if not phrase or not re.search(r"[A-Za-z]", phrase):
            continue
        phrase = phrase.lower()
        if phrase not in prompts:
            prompts.append(phrase)
    return prompts


def _append_grounding_prompts(prompts: List[str], candidates: Any) -> None:
    for phrase in normalize_grounding_prompts(candidates):
        if phrase not in prompts:
            prompts.append(phrase)


def build_grounding_prompts(
    selected: Dict[str, Any],
    args: Optional[argparse.Namespace] = None,
    extra_prompts: Any = None,
) -> List[str]:
    prompts: List[str] = []
    manual_prompt = getattr(args, "target_prompt", None) if args is not None else None
    _append_grounding_prompts(prompts, manual_prompt)
    _append_grounding_prompts(prompts, extra_prompts)
    _append_grounding_prompts(prompts, selected.get("grounding_prompts"))

    target_id = str(selected.get("target_id") or "").replace("_", " ")
    label = str(selected.get("label") or "")
    description = str(selected.get("description") or "")
    haystack = f"{target_id} {label} {description}".lower()

    alias_rules = [
        (("durian", "榴莲", "带刺", "刺"), ["green spiky toy", "green cartoon case", "green durian", "durian toy", "toy case", "spiky case", "case"]),
        (("mug", "马克杯"), ["mug", "cup", "black mug"]),
        (("cup", "杯子", "碗", "bowl"), ["cup", "bowl"]),
        (("bottle", "瓶"), ["bottle", "plastic bottle", "blue bottle", "white bottle"]),
        (("phone", "手机"), ["phone", "smartphone", "black phone"]),
        (("box", "盒", "case", "收纳盒"), ["box", "case", "container", "package"]),
        (("banana", "香蕉"), ["banana", "yellow banana"]),
        (("drill", "电钻"), ["drill", "electric drill"]),
    ]
    for keywords, aliases in alias_rules:
        if any(keyword in haystack for keyword in keywords):
            _append_grounding_prompts(prompts, aliases)

    _append_grounding_prompts(prompts, target_id)
    tokens = re.findall(r"[a-zA-Z]+", target_id)
    for token in tokens:
        if len(token) >= 3:
            _append_grounding_prompts(prompts, token)
    if len(tokens) >= 2:
        _append_grounding_prompts(prompts, [" ".join(tokens[:2]), " ".join(tokens[-2:])])

    if not prompts:
        prompts.append("object")
    return prompts[:10]


def bbox_iou(a: Sequence[int], b: Optional[Sequence[int]]) -> float:
    if b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center_affinity(a: Sequence[int], b: Optional[Sequence[int]], width: int, height: int) -> float:
    if b is None:
        return 0.0
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    diag = max(1.0, float(np.hypot(width, height)))
    distance = float(np.hypot(ax - bx, ay - by))
    return max(0.0, 1.0 - distance / diag)


def parse_bbox(text: str, width: int, height: int) -> List[int]:
    bbox = normalize_bbox([float(v.strip()) for v in text.split(",")], width, height)
    if bbox is None:
        raise ValueError("Invalid bbox, expected x1,y1,x2,y2")
    return bbox


def select_bbox_with_opencv(rgb_path: Path) -> List[int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for --bbox-ui. Install it or use --bbox x1,y1,x2,y2.") from exc
    from PIL import Image

    width, height = Image.open(rgb_path).size
    image_bgr = cv2.imread(str(rgb_path))
    if image_bgr is None:
        raise RuntimeError(f"OpenCV failed to read image: {rgb_path}")
    print("[bbox-ui] Drag a tight box around ONLY the selected target, then press Enter/Space. Press C to cancel.")
    roi = cv2.selectROI("Select target bbox", image_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select target bbox")
    x, y, w, h = [int(v) for v in roi]
    if w <= 1 or h <= 1:
        raise RuntimeError("No bbox selected in --bbox-ui")
    bbox = normalize_bbox([x, y, x + w, y + h], width, height)
    if bbox is None:
        raise RuntimeError("Invalid bbox selected in --bbox-ui")
    print(f"[bbox-ui] selected bbox: {bbox}")
    return bbox


def choose_option(options_result: Dict[str, Any], choice: str) -> Dict[str, Any]:
    choice = choice.upper().strip()
    for option in options_result["options"]:
        if option["key"].upper() == choice:
            return option
    keys = ", ".join(option["key"] for option in options_result["options"])
    raise ValueError(f"Choice {choice!r} not found. Available choices: {keys}")


def add_graspnet_paths(graspnet_root: Path) -> None:
    api_root = graspnet_root.parent / "graspnetAPI"
    if api_root.exists() and str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))
    for subdir in ["models", "dataset", "utils", "pointnet2", "knn"]:
        path = str(graspnet_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)


def get_net(checkpoint_path: Path, num_view: int):
    import torch
    from graspnet import GraspNet

    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net, device


def build_bbox_mask(shape: Tuple[int, int], bbox: Sequence[int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = normalize_bbox(bbox, width, height) or bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def load_bool_image(path: Path) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def save_mask_image(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Target mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_options_overlay(
    rgb_path: Path,
    options_result: Dict[str, Any],
    output_path: Path,
    selected_key: Optional[str] = None,
    target_mask: Optional[np.ndarray] = None,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(rgb_path).convert("RGBA")
    if target_mask is not None:
        mask = Image.fromarray((target_mask.astype(np.uint8) * 110), mode="L")
        overlay = Image.new("RGBA", image.size, (20, 220, 120, 0))
        overlay.putalpha(mask)
        image = Image.alpha_composite(image, overlay)

    draw = ImageDraw.Draw(image)
    selected_key = selected_key.upper() if selected_key else None
    for option in options_result.get("options", []):
        bbox = option.get("bbox")
        if not bbox:
            continue
        key = str(option.get("key", "?")).upper()
        color = (255, 60, 40, 255) if key == selected_key else (255, 210, 40, 255)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{key}: {option.get('target_id') or option.get('label') or 'object'}"
        text_box = draw.textbbox((x1, max(0, y1 - 22)), label)
        draw.rectangle(text_box, fill=(0, 0, 0, 190))
        draw.text((x1, max(0, y1 - 22)), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def call_groundingdino_for_bbox(
    rgb_path: Path,
    prompt: str,
    config_path: Path,
    checkpoint_path: Path,
    box_threshold: float,
    text_threshold: float,
    device: str,
    preferred_bbox: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    add_local_vision_paths()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing GroundingDINO config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing GroundingDINO checkpoint: {checkpoint_path}")
    try:
        from groundingdino.util.inference import load_image, load_model, predict
    except ImportError as exc:
        raise RuntimeError(
            "GroundingDINO is not importable. Install/configure the local GitHub source "
            "under third_party/GroundingDINO and pass --groundingdino-config/--groundingdino-checkpoint."
        ) from exc
    from PIL import Image

    width, height = Image.open(rgb_path).size
    cache_key = (str(config_path.resolve()), str(checkpoint_path.resolve()), str(device))
    model = _GROUNDINGDINO_MODEL_CACHE.get(cache_key)
    if model is None:
        model = load_model(str(config_path), str(checkpoint_path), device=device)
        _GROUNDINGDINO_MODEL_CACHE[cache_key] = model
    _image_source, image = load_image(str(rgb_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO found no bbox for prompt: {prompt}")

    boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    logits_np = np.asarray(logits_np, dtype=np.float32).reshape(-1)
    preferred = normalize_bbox(preferred_bbox, width, height) if preferred_bbox is not None else None

    candidate_bboxes: List[List[int]] = []
    rank_scores: List[float] = []
    candidate_meta: List[Dict[str, float]] = []
    for idx, box in enumerate(boxes_np):
        cx, cy, bw, bh = [float(v) for v in box.tolist()]
        candidate_bbox = normalize_bbox(
            [(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height],
            width,
            height,
        )
        if candidate_bbox is None:
            continue
        iou = bbox_iou(candidate_bbox, preferred)
        center = bbox_center_affinity(candidate_bbox, preferred, width, height)
        raw_score = float(logits_np[idx]) if idx < len(logits_np) else 0.0
        rank_score = raw_score + 0.45 * iou + 0.15 * center
        candidate_bboxes.append(candidate_bbox)
        rank_scores.append(rank_score)
        candidate_meta.append({"score": raw_score, "rank_score": rank_score, "preferred_iou": iou, "center_affinity": center})

    if not candidate_bboxes:
        raise RuntimeError("GroundingDINO returned invalid bboxes")
    best = int(np.argmax(np.asarray(rank_scores, dtype=np.float32)))
    bbox = candidate_bboxes[best]
    phrase = str(phrases[best]) if phrases and best < len(phrases) else prompt
    return bbox, {"prompt": prompt, "phrase": phrase, **candidate_meta[best], "candidates": len(candidate_bboxes)}

def call_sam_for_mask(
    rgb_path: Path,
    bbox: Sequence[int],
    checkpoint_path: Path,
    model_type: str,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    add_local_vision_paths()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint_path}")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment-anything is not importable. Install/configure the local GitHub source "
            "under third_party/segment-anything and pass --sam-checkpoint."
        ) from exc
    from PIL import Image

    image = np.asarray(Image.open(rgb_path).convert("RGB"))
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, _logits = predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=True)
    best = int(np.argmax(scores))
    return masks[best].astype(bool), {"score": float(scores[best]), "model_type": model_type}


def call_grabcut_for_mask(rgb_path: Path, bbox: Sequence[int], iterations: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for --segmenter grabcut. Install it or use --segmenter none.") from exc
    from PIL import Image

    image_rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = normalize_bbox(bbox, width, height) or list(bbox)
    rect_w = max(2, int(x2 - x1))
    rect_h = max(2, int(y2 - y1))
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(image_bgr, mask, (int(x1), int(y1), rect_w, rect_h), bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    target = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

    # Keep only the largest connected foreground component inside the selected box.
    target &= build_bbox_mask((height, width), [x1, y1, x2, y2])
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(target.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return build_bbox_mask((height, width), [x1, y1, x2, y2]), {"fallback": "empty_grabcut_mask"}
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    target = labels == largest
    return target.astype(bool), {"iterations": iterations, "component_area": int(stats[largest, cv2.CC_STAT_AREA])}


def resolve_target_region(rgb_path: Path, selected: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    from PIL import Image

    width, height = Image.open(rgb_path).size
    bbox = normalize_bbox(selected.get("bbox"), width, height)
    mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = {}
    localizer = "qwen_bbox"

    if args.mask:
        mask = load_bool_image(Path(args.mask))
        if mask.shape != (height, width):
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {(height, width)}")
        bbox = mask_to_bbox(mask)
        localizer = "manual_mask"
    elif args.bbox_ui:
        bbox = select_bbox_with_opencv(rgb_path)
        selected["bbox"] = bbox
        metadata["manual_bbox_ui"] = {"bbox": bbox}
        localizer = "manual_bbox_ui"
    elif args.refine_qwen_bbox:
        api_key = resolve_qwen_api_key(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"--refine-qwen-bbox needs API key. Set {args.api_key_env} or configs/local_secrets.json")
        bbox, metadata["qwen_refine_bbox"] = call_qwen_refine_bbox(
            rgb_path=rgb_path,
            selected=selected,
            base_url=resolve_qwen_base_url(args.qwen_base_url),
            model=resolve_qwen_model(args.qwen_model),
            api_key=api_key,
        )
        selected["bbox"] = bbox
        localizer = "qwen_refined_bbox"
    elif args.localizer == "groundingdino":
        prompts = build_grounding_prompts(selected, args=args)
        errors: List[str] = []
        grounding_meta: Optional[Dict[str, Any]] = None
        coarse_bbox = bbox
        for prompt in prompts:
            try:
                bbox, grounding_meta = call_groundingdino_for_bbox(
                    rgb_path=rgb_path,
                    prompt=str(prompt),
                    config_path=Path(args.groundingdino_config),
                    checkpoint_path=Path(args.groundingdino_checkpoint),
                    box_threshold=args.grounding_box_threshold,
                    text_threshold=args.grounding_text_threshold,
                    device=args.device,
                    preferred_bbox=coarse_bbox,
                )
                print(f"[grounding] prompt={prompt!r} bbox={bbox} score={grounding_meta.get('score', 0.0):.3f}")
                break
            except RuntimeError as exc:
                errors.append(str(exc))
        if grounding_meta is None:
            tried = ", ".join(prompts)
            tail = errors[-1] if errors else "no candidates"
            raise RuntimeError(f"GroundingDINO found no bbox for prompts: {tried}. Last error: {tail}")
        metadata["groundingdino"] = {**grounding_meta, "tried_prompts": prompts, "coarse_bbox": coarse_bbox}
        selected["bbox"] = bbox
        selected["grounding_prompts"] = prompts
        localizer = "groundingdino"
    elif bbox is None:
        raise ValueError("Selected option has no bbox. For autonomous use, run with --localizer groundingdino --segmenter sam.")

    qwen_only_localizer = localizer in {"qwen_bbox", "qwen_refined_bbox"}
    if qwen_only_localizer and not args.allow_qwen_bbox_grasp:
        raise RuntimeError(
            "Qwen bbox is not reliable enough for autonomous grasping. "
            "Use the automatic path: --localizer groundingdino --segmenter sam. "
            "If you only want a debugging baseline, add --allow-qwen-bbox-grasp explicitly."
        )
    if qwen_only_localizer:
        print("[warning] Debug mode: rough Qwen bbox is entering GraspNet. This is not suitable for patient/autonomous use.")

    if mask is None and args.segmenter == "sam":
        mask, metadata["sam"] = call_sam_for_mask(
            rgb_path=rgb_path,
            bbox=bbox,
            checkpoint_path=Path(args.sam_checkpoint),
            model_type=args.sam_model_type,
            device=args.device,
        )
        bbox = mask_to_bbox(mask)
        localizer = f"{localizer}+sam"
    elif mask is None and args.segmenter == "grabcut":
        mask, metadata["grabcut"] = call_grabcut_for_mask(rgb_path, bbox, args.grabcut_iter)
        bbox = mask_to_bbox(mask)
        localizer = f"{localizer}+grabcut"
    elif mask is None:
        print("[warning] --segmenter none only uses a rectangular bbox. Thin/curved objects like banana may include nearby objects. Prefer --segmenter grabcut or --segmenter sam.")
        mask = build_bbox_mask((height, width), bbox)
        localizer = f"{localizer}_bbox_mask"

    mask_path = output_dir / "target_mask.png"
    overlay_path = output_dir / "target_overlay.png"
    if not args.no_save_overlays:
        save_mask_image(mask, mask_path)

    return {
        "bbox": bbox,
        "mask": mask,
        "mask_path": str(mask_path) if not args.no_save_overlays else None,
        "overlay_path": str(overlay_path) if not args.no_save_overlays else None,
        "source": localizer,
        "metadata": metadata,
    }


def load_frame(frame_dir: Path, target_mask: np.ndarray, num_point: int):
    import open3d as o3d
    import scipy.io as scio
    import torch
    from PIL import Image
    from data_utils import CameraInfo, create_point_cloud_from_depth_image

    rgb_path = frame_dir / "color.png"
    depth_path = frame_dir / "depth.png"
    mask_path = frame_dir / "workspace_mask.png"
    meta_path = frame_dir / "meta.mat"
    color = np.asarray(Image.open(rgb_path), dtype=np.float32) / 255.0
    depth = np.asarray(Image.open(depth_path))
    workspace_mask = load_bool_image(mask_path)
    meta = scio.loadmat(meta_path)
    intrinsic = meta["intrinsic_matrix"]
    factor_depth = meta["factor_depth"]
    height, width = depth.shape[:2]
    if target_mask.shape != (height, width):
        raise ValueError(f"Target mask shape {target_mask.shape} does not match depth shape {(height, width)}")

    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    valid_mask = workspace_mask & (depth > 0)
    full_points = cloud[valid_mask]
    full_colors = color[valid_mask]
    if len(full_points) == 0:
        raise RuntimeError("No valid depth points in workspace")

    target_valid_mask = valid_mask & target_mask
    target_points = cloud[target_valid_mask]
    target_colors = color[target_valid_mask]
    if len(target_points) == 0:
        raise RuntimeError("No valid depth points in selected target region. Check bbox/mask alignment.")

    if len(full_points) >= num_point:
        idxs = np.random.choice(len(full_points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(full_points))
        idxs2 = np.random.choice(len(full_points), num_point - len(full_points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    sampled_points = full_points[idxs]
    sampled_colors = full_colors[idxs]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    end_points = {
        "point_clouds": torch.from_numpy(sampled_points[np.newaxis].astype(np.float32)).to(device),
        "cloud_colors": sampled_colors,
    }

    full_cloud = make_cloud(o3d, full_points, full_colors)
    target_cloud = make_cloud(o3d, target_points, target_colors)
    return end_points, full_cloud, target_cloud, intrinsic


def make_cloud(o3d, points: np.ndarray, colors: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return cloud


def get_grasps(net, end_points):
    import torch
    from graspnet import pred_decode
    from graspnetAPI import GraspGroup

    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    return GraspGroup(grasp_preds[0].detach().cpu().numpy())


def run_collision_detection(gg, cloud, collision_thresh: float, voxel_size: float):
    from collision_detector import ModelFreeCollisionDetector

    detector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
    collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    return gg[~collision_mask]


def project_points(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0][0])
    fy = float(intrinsic[1][1])
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    uv[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return uv


def filter_grasps_by_region(
    gg,
    bbox: Sequence[int],
    target_mask: np.ndarray,
    intrinsic: np.ndarray,
    pad: int,
    fallback_nearest: bool,
):
    translations = gg.translations
    uv = project_points(translations, intrinsic)
    height, width = target_mask.shape
    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])

    rounded_x = np.rint(uv[:, 0]).astype(np.int64)
    rounded_y = np.rint(uv[:, 1]).astype(np.int64)
    in_bounds = valid & (rounded_x >= 0) & (rounded_x < width) & (rounded_y >= 0) & (rounded_y < height)
    inside_mask = np.zeros(len(uv), dtype=bool)
    inside_mask[in_bounds] = target_mask[rounded_y[in_bounds], rounded_x[in_bounds]]

    x1, y1, x2, y2 = bbox
    inside_bbox = (
        valid
        & (uv[:, 0] >= x1 - pad)
        & (uv[:, 0] <= x2 + pad)
        & (uv[:, 1] >= y1 - pad)
        & (uv[:, 1] <= y2 + pad)
    )

    filtered = gg[inside_mask]
    filtered_uv = uv[inside_mask]
    mode = "mask"
    if len(filtered) == 0:
        filtered = gg[inside_bbox]
        filtered_uv = uv[inside_bbox]
        mode = "bbox_pad"
    if len(filtered) == 0 and fallback_nearest and len(gg) > 0:
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        dist = np.sum((uv - center) ** 2, axis=1)
        dist[~valid] = np.inf
        order = np.argsort(dist)[: min(50, len(gg))]
        filtered = gg[order]
        filtered_uv = uv[order]
        mode = "nearest_to_target_center"
    return filtered, filtered_uv, mode


def safe_nms(gg):
    try:
        return gg.nms()
    except ImportError as exc:
        print(f"[warning] grasp_nms is not installed, skip NMS: {exc}")
        return gg


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(quat)
    if norm > 1e-8:
        quat = quat / norm
    return quat.tolist()


def summarize_grasps(gg, uv: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    result = []
    for i, grasp in enumerate(gg[:top_k]):
        rotation = np.asarray(grasp.rotation_matrix, dtype=float)
        item = {
            "rank": i + 1,
            "score": float(grasp.score),
            "width": float(grasp.width),
            "height": float(grasp.height),
            "depth": float(grasp.depth),
            "translation": np.asarray(grasp.translation, dtype=float).tolist(),
            "quaternion": rotation_matrix_to_quaternion(rotation),
            "rotation_matrix": rotation.tolist(),
        }
        if i < len(uv):
            item["projected_uv"] = [float(uv[i][0]), float(uv[i][1])]
        result.append(item)
    return result


def visualize(target_cloud, gg, top_k: int) -> None:
    import open3d as o3d

    gg = gg[:top_k]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([target_cloud, *grippers], window_name="target grasps")


def visualize_compare(full_cloud, target_cloud, gg, top_k: int) -> None:
    import copy
    import open3d as o3d

    bbox = full_cloud.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    offset = np.array([max(float(extent[0]) * 1.8, 0.65), 0.0, 0.0], dtype=float)

    left_full = copy.deepcopy(full_cloud)
    left_highlight = copy.deepcopy(target_cloud)
    left_highlight.paint_uniform_color([1.0, 0.05, 0.02])

    right_target = copy.deepcopy(target_cloud)
    right_target.translate(offset)

    grippers = []
    for geom in gg[:top_k].to_open3d_geometry_list():
        moved = copy.deepcopy(geom)
        moved.translate(offset)
        grippers.append(moved)

    left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
    right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=offset.tolist())
    print("[compare-vis] Left: original full point cloud, red points = current selected mask in original scene.")
    print("[compare-vis] Right: selected target point cloud pulled out, with filtered grasp candidates.")
    print("[compare-vis] If the red/right cloud is not the banana, the localization/segmentation is wrong before GraspNet.")
    o3d.visualization.draw_geometries(
        [left_full, left_highlight, left_frame, right_target, right_frame, *grippers],
        window_name="left=original scene with selected mask, right=selected target + grasps",
    )


def build_manual_options(rgb_path: Path, bbox_text: str) -> Dict[str, Any]:
    from PIL import Image

    width, height = Image.open(rgb_path).size
    bbox = parse_bbox(bbox_text, width, height)
    return {
        "image_width": width,
        "image_height": height,
        "question": "请选择你想抓取的物体",
        "options": [
            {"key": "A", "label": "手工指定目标", "target_id": "manual_target", "bbox": bbox, "confidence": 1.0}
        ],
    }

def load_options_file(options_path: Path, rgb_path: Path, max_options: int) -> Dict[str, Any]:
    from PIL import Image

    data = json.loads(options_path.read_text(encoding="utf-8"))
    result = data.get("qwen_options", data)
    width, height = Image.open(rgb_path).size
    return normalize_options(result, width, height, max_options)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Qwen/GroundingDINO/SAM target selection + target-filtered GraspNet demo")
    parser.add_argument("--frame-dir", default="third_party/graspnet-baseline/doc/example_data")
    parser.add_argument("--graspnet-root", default="third_party/graspnet-baseline")
    parser.add_argument("--checkpoint", default="weights/graspnet/checkpoint-rs.tar")
    parser.add_argument("--output-dir", default="outputs/target_grasp")
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-model")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--choice", help="A/B/C/D. If omitted, only print/save Qwen options and exit.")
    parser.add_argument("--bbox", help="Manual bbox fallback: x1,y1,x2,y2. Skips Qwen option generation.")
    parser.add_argument("--bbox-ui", action="store_true", help="Debug only: open an OpenCV window to manually drag a bbox. Not suitable for patient/autonomous use.")
    parser.add_argument("--options-json", help="Reuse a saved qwen_options.json instead of calling Qwen again.")
    parser.add_argument("--mask", help="Manual target mask path. If set, this mask overrides bbox filtering.")
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--localizer", choices=["qwen_bbox", "groundingdino"], default="groundingdino")
    parser.add_argument("--refine-qwen-bbox", action="store_true", help="Debug only: call Qwen a second time to tighten the selected object bbox.")
    parser.add_argument("--allow-qwen-bbox-grasp", action="store_true", help="Debug only: allow rough Qwen bbox to enter GraspNet. Do not use for autonomous/patient scenarios.")
    parser.add_argument("--target-prompt", help="Text prompt for GroundingDINO. Default uses selected label/target_id.")
    parser.add_argument("--groundingdino-config", default="weights/groundingdino/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino-checkpoint", default="weights/groundingdino/groundingdino_swint_ogc.pth")
    parser.add_argument("--grounding-box-threshold", type=float, default=0.35)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.25)
    parser.add_argument("--segmenter", choices=["none", "grabcut", "sam"], default="sam")
    parser.add_argument("--sam-checkpoint", default="weights/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--grabcut-iter", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bbox-pad", type=int, default=20)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--no-vis", action="store_true", help="Do not open Open3D window; only print JSON result.")
    parser.add_argument("--vis-mode", choices=["target", "compare"], default="target", help="Open3D view mode. compare shows full scene and selected target side by side.")
    parser.add_argument("--no-save-overlays", action="store_true", help="Do not write option/target overlay images and JSON files.")
    parser.add_argument("--no-fallback-nearest", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    add_local_vision_paths(root)
    prepare_huggingface_cache(root)
    frame_dir = (root / args.frame_dir).resolve()
    graspnet_root = (root / args.graspnet_root).resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    output_dir = (root / args.output_dir).resolve()
    args.groundingdino_config = str((root / args.groundingdino_config).resolve())
    args.groundingdino_checkpoint = str((root / args.groundingdino_checkpoint).resolve())
    args.sam_checkpoint = str((root / args.sam_checkpoint).resolve())
    rgb_path = frame_dir / "color.png"

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
    if not checkpoint_path.exists() and args.choice:
        raise FileNotFoundError(f"Missing GraspNet checkpoint: {checkpoint_path}")

    if args.bbox:
        options_result = build_manual_options(rgb_path, args.bbox)
    elif args.options_json:
        options_result = load_options_file((root / args.options_json).resolve(), rgb_path, args.max_options)
    else:
        api_key = resolve_qwen_api_key(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set {args.api_key_env} or configs/local_secrets.json")
        options_result = call_qwen_for_options(
            rgb_path,
            resolve_qwen_base_url(args.qwen_base_url),
            resolve_qwen_model(args.qwen_model),
            api_key,
            args.max_options,
        )

    print(json.dumps({"qwen_options": options_result}, ensure_ascii=False, indent=2))
    if not args.no_save_overlays:
        write_json(output_dir / "qwen_options.json", {"qwen_options": options_result})
        draw_options_overlay(rgb_path, options_result, output_dir / "qwen_options_overlay.png")
        print(f"\nSaved option overlay: {output_dir / 'qwen_options_overlay.png'}")

    if not args.choice:
        print("\nNo --choice provided. Re-run with --choice A/B/C/D to run target-filtered GraspNet.")
        return

    selected = choose_option(options_result, args.choice)
    target_region = resolve_target_region(rgb_path, selected, args, output_dir)
    if not args.no_save_overlays:
        selected_overlay_result = {
            **options_result,
            "options": [{**selected, "bbox": target_region["bbox"]}],
        }
        draw_options_overlay(
            rgb_path,
            selected_overlay_result,
            output_dir / "target_overlay.png",
            selected_key=selected["key"],
            target_mask=target_region["mask"],
        )

    prepare_windows_dll_paths()
    add_graspnet_paths(graspnet_root)

    net, _device = get_net(checkpoint_path, args.num_view)
    end_points, full_cloud, target_cloud, intrinsic = load_frame(frame_dir, target_region["mask"], args.num_point)
    gg = get_grasps(net, end_points)
    if args.collision_thresh > 0:
        gg = run_collision_detection(gg, full_cloud, args.collision_thresh, args.voxel_size)
    gg = safe_nms(gg)
    gg.sort_by_score()
    filtered, filtered_uv, filter_mode = filter_grasps_by_region(
        gg,
        target_region["bbox"],
        target_region["mask"],
        intrinsic,
        pad=args.bbox_pad,
        fallback_nearest=not args.no_fallback_nearest,
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
        "top_grasps": summarize_grasps(filtered, filtered_uv, args.top_k),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.no_save_overlays:
        write_json(output_dir / "target_grasps.json", summary)

    if not args.no_vis:
        if args.vis_mode == "compare":
            visualize_compare(full_cloud, target_cloud, filtered, args.top_k)
        else:
            visualize(target_cloud, filtered, args.top_k)


if __name__ == "__main__":
    main()

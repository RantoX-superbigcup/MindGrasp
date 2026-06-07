from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class FrameworkConfig:
    project_root: Path
    graspnet_root: Path
    graspnet_checkpoint: Optional[Path]
    qwen_base_url: str
    qwen_model: str
    qwen_api_key_env: str
    robot_port: str
    robot_baudrate: int
    robot_protocol: str
    dry_run: bool
    collision_thresh: float
    top_k_grasps: int

    @property
    def qwen_api_key(self) -> str:
        return os.getenv(self.qwen_api_key_env, "")


def _resolve(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(path: str | Path) -> FrameworkConfig:
    config_path = Path(path).resolve()
    data: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = _resolve(config_path.parent, data.get("project_root")) or config_path.parent.parent
    return FrameworkConfig(
        project_root=project_root,
        graspnet_root=_resolve(project_root, data["graspnet_root"]),
        graspnet_checkpoint=_resolve(project_root, data.get("graspnet_checkpoint")),
        qwen_base_url=data.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        qwen_model=data.get("qwen_model", "qwen2.5-vl-72b-instruct"),
        qwen_api_key_env=data.get("qwen_api_key_env", "QWEN_API_KEY"),
        robot_port=data.get("robot_port", "COM3"),
        robot_baudrate=int(data.get("robot_baudrate", 9600)),
        robot_protocol=data.get("robot_protocol", "beta_l4"),
        dry_run=bool(data.get("dry_run", True)),
        collision_thresh=float(data.get("collision_thresh", 0.01)),
        top_k_grasps=int(data.get("top_k_grasps", 10)),
    )
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


from .schemas import FrameBundle, PerceptionResult


class QwenVisionClient:
    def __init__(self, base_url: str, model: str, api_key: str, dry_run: bool = True) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.dry_run = dry_run

    def analyze_scene(self, frame: FrameBundle, target_label: str) -> PerceptionResult:
        if self.dry_run or not self.api_key:
            return PerceptionResult(
                target_label=target_label,
                target_instance=f"{target_label}_1",
                occluded=False,
                need_clarification=False,
                confidence=0.5,
                raw_response={"mode": "dry_run"},
            )

        image_url = self._image_to_data_url(frame.rgb_path)
        prompt = (
            "你是辅助取物系统的视觉理解模块。请根据图像识别目标实例、遮挡关系和是否需要用户澄清。"
            "只输出 JSON，字段包括 target_instance,target_bbox,occluded,occluder,need_clarification,"
            "clarification_question,confidence。目标类别: " + target_label
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        import requests

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = self._extract_json(content)
        return PerceptionResult(
            target_label=target_label,
            target_instance=parsed.get("target_instance"),
            target_bbox=parsed.get("target_bbox"),
            occluded=bool(parsed.get("occluded", False)),
            occluder=parsed.get("occluder"),
            need_clarification=bool(parsed.get("need_clarification", False)),
            clarification_question=parsed.get("clarification_question"),
            confidence=float(parsed.get("confidence", 0.0)),
            raw_response={"content": content, "parsed": parsed},
        )

    @staticmethod
    def _image_to_data_url(path: str) -> str:
        image_path = Path(path)
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match: Optional[re.Match[str]] = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise
            return json.loads(match.group(0))
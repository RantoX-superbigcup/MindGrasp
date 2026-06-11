from __future__ import annotations

from typing import Any, Dict, Optional

from .schemas import TaskContext
from .state_machine import EmbodiedPickPipeline


class IPCBridge:
    def __init__(self, pipeline: EmbodiedPickPipeline) -> None:
        self.pipeline = pipeline

    def handle_platform_json(self, ipc_json: Dict[str, Any], frame_dir: Optional[str] = None) -> Optional[TaskContext]:
        msg = ipc_json.get("msg")
        if msg != "ipc_algorithm_test":
            return None
        result_args = ipc_json.get("result_args", {})
        command = result_args.get("data")
        return self.pipeline.handle_bci_command(command, frame_dir=frame_dir)
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from .bci import BCICommandRouter
from .config import FrameworkConfig
from .graspnet_adapter import GraspNetAdapter
from .planner import GraspPlanner
from .qwen_client import QwenVisionClient
from .robot import DryRunRobotController, SerialArmController
from .schemas import FrameBundle, SystemStatus, TaskContext


class EmbodiedPickPipeline:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.router = BCICommandRouter()
        self.qwen = QwenVisionClient(config.qwen_base_url, config.qwen_model, config.qwen_api_key, config.dry_run)
        self.graspnet = GraspNetAdapter(
            config.graspnet_root,
            config.graspnet_checkpoint,
            dry_run=config.dry_run,
            collision_thresh=config.collision_thresh,
            top_k=config.top_k_grasps,
        )
        self.planner = GraspPlanner(dry_run=config.dry_run)
        self.robot = DryRunRobotController() if config.dry_run else SerialArmController(
            config.robot_port,
            config.robot_baudrate,
            config.robot_protocol,
        )
        self.context = TaskContext(task_id=self._new_task_id())

    def handle_bci_command(self, raw_command: object, frame_dir: Optional[str] = None) -> TaskContext:
        command = self.router.apply(raw_command)
        if command.name == "emergency_stop":
            self.context.execution = self.robot.emergency_stop()
            self.context.status = SystemStatus.EMERGENCY_STOP
            return self.context
        if command.name in {"previous_target", "next_target"}:
            self.context.target_label = self.router.highlighted_target
            self.context.status = SystemStatus.IDLE
            return self.context
        if command.name == "confirm_target":
            self.context.target_label = self.router.confirmed_target
            self.context.status = SystemStatus.TARGET_SELECTED
            return self.context
        if command.name in {"execute", "execute_with_current_info"}:
            target = self.router.confirmed_target or self.router.highlighted_target
            return self.run_task(target, frame_dir)
        if command.name == "cancel":
            self.context = TaskContext(task_id=self._new_task_id())
            return self.context
        return self.context

    def run_task(self, target_label: str, frame_dir: Optional[str]) -> TaskContext:
        self.context = TaskContext(task_id=self._new_task_id(), target_label=target_label)
        self.context.status = SystemStatus.PLANNING
        frame = self._make_frame_bundle(frame_dir)
        self.context.frame = frame
        self.context.perception = self.qwen.analyze_scene(frame, target_label)
        if self.context.perception.need_clarification:
            self.context.status = SystemStatus.CLARIFYING
            return self.context
        self.context.grasp_candidates = self.graspnet.predict(frame, self.context.perception.target_mask_path)
        self.context.plan = self.planner.plan(self.context)
        if not self.context.plan.success:
            self.context.status = SystemStatus.FAILED
            return self.context
        self.context.status = SystemStatus.EXECUTING
        self.context.execution = self.robot.execute(self.context.plan)
        self.context.status = SystemStatus.DONE if self.context.execution.success else SystemStatus.FAILED
        return self.context

    def _make_frame_bundle(self, frame_dir: Optional[str]) -> FrameBundle:
        if frame_dir:
            base = Path(frame_dir)
            return FrameBundle(
                rgb_path=str(base / "color.png"),
                depth_path=str(base / "depth.png"),
                workspace_mask_path=str(base / "workspace_mask.png"),
                meta_path=str(base / "meta.mat"),
            )
        example = self.config.graspnet_root / "doc" / "example_data"
        return FrameBundle(
            rgb_path=str(example / "color.png"),
            depth_path=str(example / "depth.png"),
            workspace_mask_path=str(example / "workspace_mask.png"),
            meta_path=str(example / "meta.mat"),
        )

    @staticmethod
    def _new_task_id() -> str:
        return f"task_{uuid.uuid4().hex[:8]}"
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SystemStatus(str, Enum):
    IDLE = "idle"
    TARGET_SELECTED = "target_selected"
    CLARIFYING = "clarifying"
    PLANNING = "planning"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class Pose6D:
    position: List[float]
    quaternion: List[float]
    frame_id: str = "camera"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameBundle:
    rgb_path: str
    depth_path: Optional[str] = None
    workspace_mask_path: Optional[str] = None
    meta_path: Optional[str] = None
    camera_intrinsics: Optional[List[float]] = None


@dataclass
class PerceptionResult:
    target_label: str
    target_instance: Optional[str] = None
    target_bbox: Optional[List[int]] = None
    target_mask_path: Optional[str] = None
    occluded: bool = False
    occluder: Optional[str] = None
    need_clarification: bool = False
    clarification_question: Optional[str] = None
    confidence: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraspCandidate:
    pose: Pose6D
    score: float = 0.0
    width: Optional[float] = None
    depth: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    success: bool
    selected_grasp: Optional[GraspCandidate] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class ExecutionResult:
    success: bool
    reason: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    task_id: str
    target_label: Optional[str] = None
    frame: Optional[FrameBundle] = None
    perception: Optional[PerceptionResult] = None
    grasp_candidates: List[GraspCandidate] = field(default_factory=list)
    plan: Optional[PlanResult] = None
    execution: Optional[ExecutionResult] = None
    status: SystemStatus = SystemStatus.IDLE

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
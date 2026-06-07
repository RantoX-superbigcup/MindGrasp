from __future__ import annotations

from typing import List

from .schemas import GraspCandidate, PlanResult, TaskContext


class GraspPlanner:
    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def plan(self, context: TaskContext) -> PlanResult:
        if not context.grasp_candidates:
            return PlanResult(success=False, reason="no grasp candidates")
        selected = self._select_candidate(context.grasp_candidates)
        trajectory = [
            {"name": "pregrasp", "pose": selected.pose.as_dict()},
            {"name": "grasp", "pose": selected.pose.as_dict()},
            {"name": "close_gripper", "width": selected.width},
            {"name": "lift", "delta_z": 0.08},
            {"name": "place", "target": "delivery_area"},
        ]
        return PlanResult(success=True, selected_grasp=selected, trajectory=trajectory, reason="dry planner selected top grasp")

    @staticmethod
    def _select_candidate(candidates: List[GraspCandidate]) -> GraspCandidate:
        return sorted(candidates, key=lambda item: item.score, reverse=True)[0]


class MoveIt2Planner:
    """Placeholder adapter for ROS 2 MoveIt integration.

    Keep this class thin: convert selected grasp pose into PoseStamped, add table and
    obstacles to the planning scene, then request pregrasp -> grasp -> lift -> place.
    """

    def plan(self, context: TaskContext) -> PlanResult:
        raise NotImplementedError("MoveIt 2 integration should run inside a ROS 2 node")
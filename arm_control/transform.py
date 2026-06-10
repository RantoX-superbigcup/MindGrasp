from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np


DEFAULT_CAMERA_TO_BASE_R = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)
DEFAULT_CAMERA_TO_BASE_T = np.array([0.0, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class ArmCommandConfig:
    camera_to_base_R: np.ndarray = DEFAULT_CAMERA_TO_BASE_R
    camera_to_base_t_m: np.ndarray = DEFAULT_CAMERA_TO_BASE_T
    l1_mm: float = 100.0
    l2_mm: float = 100.0
    standoff_mm: float = 40.0
    approach_axis: int = 0
    yaw_min_deg: float = -90.0
    yaw_max_deg: float = 90.0


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


def quat_to_rotation_matrix(qxyzw: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = [float(v) for v in qxyzw]
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _ik_reachable(r_mm: float, h_mm: float, l1_mm: float, l2_mm: float) -> Tuple[bool, float]:
    reach2 = r_mm * r_mm + h_mm * h_mm
    c2 = (reach2 - l1_mm * l1_mm - l2_mm * l2_mm) / (2.0 * l1_mm * l2_mm)
    distance = float(np.hypot(r_mm, h_mm))
    reachable = (-1.0 <= c2 <= 1.0) and (distance <= l1_mm + l2_mm) and (distance >= abs(l1_mm - l2_mm))
    return bool(reachable), float(c2)


def compute_arm_command(position_m: Sequence[float], quaternion_xyzw: Sequence[float], config: ArmCommandConfig) -> Dict[str, Any]:
    position_cam = np.asarray(position_m, dtype=float)
    if position_cam.shape != (3,):
        raise ValueError("position_m must be length 3")
    if config.approach_axis not in (0, 1, 2):
        raise ValueError("approach_axis must be 0, 1, or 2")

    R_bc = np.asarray(config.camera_to_base_R, dtype=float).reshape(3, 3)
    t_bc = np.asarray(config.camera_to_base_t_m, dtype=float).reshape(3)
    R_cam = quat_to_rotation_matrix(quaternion_xyzw)
    approach_cam = R_cam[:, int(config.approach_axis)]

    grasp_base_m = R_bc @ position_cam + t_bc
    approach_base = R_bc @ approach_cam
    norm = float(np.linalg.norm(approach_base))
    if norm > 1e-12:
        approach_base = approach_base / norm

    pregrasp_base_m = grasp_base_m - (float(config.standoff_mm) / 1000.0) * approach_base
    X, Y, Z = (pregrasp_base_m * 1000.0).tolist()

    yaw_deg = float(np.degrees(np.arctan2(Y, X)))
    r_mm = float(np.hypot(X, Y))
    h_mm = float(Z)
    elbow = 1 if float(approach_base[2]) <= 0.0 else -1
    ik_reachable, c2 = _ik_reachable(r_mm, h_mm, float(config.l1_mm), float(config.l2_mm))
    yaw_reachable = float(config.yaw_min_deg) <= yaw_deg <= float(config.yaw_max_deg)
    reachable = bool(ik_reachable and yaw_reachable)
    packet = "<C %.2f;%.2f;%.2f;%d>" % (r_mm, h_mm, yaw_deg, elbow)

    reasons = []
    if not ik_reachable:
        reasons.append("outside_2link_workspace")
    if not yaw_reachable:
        reasons.append("outside_yaw_range")

    return {
        "schema_version": "intentgrasp.arm_command.v1",
        "status": "ready_to_send" if reachable else "not_reachable",
        "packet": packet,
        "reachable": reachable,
        "reason": ",".join(reasons) if reasons else "ok",
        "command": {
            "r_mm": r_mm,
            "h_mm": h_mm,
            "yaw_deg": yaw_deg,
            "elbow": elbow,
        },
        "kinematics": {
            "l1_mm": float(config.l1_mm),
            "l2_mm": float(config.l2_mm),
            "c2": c2,
            "yaw_min_deg": float(config.yaw_min_deg),
            "yaw_max_deg": float(config.yaw_max_deg),
        },
        "transform": {
            "camera_to_base_R": R_bc.tolist(),
            "camera_to_base_t_m": t_bc.tolist(),
            "standoff_mm": float(config.standoff_mm),
            "approach_axis": int(config.approach_axis),
            "grasp_base_m": grasp_base_m.tolist(),
            "pregrasp_base_m": pregrasp_base_m.tolist(),
            "approach_base": approach_base.tolist(),
        },
        "firmware_protocol": "<C r;h;yaw;elbow>",
        "safety_note": "This command assumes camera-to-base calibration and matching servo geometry. Use arm-mode=serial only after validating arm_command.json.",
    }



def transform_camera_point_to_base(point_m: Sequence[float], config: ArmCommandConfig) -> np.ndarray:
    point_cam = np.asarray(point_m, dtype=float)
    if point_cam.shape != (3,):
        raise ValueError("point_m must be length 3")
    R_bc = np.asarray(config.camera_to_base_R, dtype=float).reshape(3, 3)
    t_bc = np.asarray(config.camera_to_base_t_m, dtype=float).reshape(3)
    return R_bc @ point_cam + t_bc


def compute_arm_command_from_base_point(
    point_base_m: Sequence[float],
    config: ArmCommandConfig,
    elbow: int = 1,
    label: str = "cartesian_point",
) -> Dict[str, Any]:
    point_base = np.asarray(point_base_m, dtype=float)
    if point_base.shape != (3,):
        raise ValueError("point_base_m must be length 3")
    X, Y, Z = (point_base * 1000.0).tolist()
    yaw_deg = float(np.degrees(np.arctan2(Y, X)))
    r_mm = float(np.hypot(X, Y))
    h_mm = float(Z)
    elbow = 1 if int(elbow) >= 0 else -1
    ik_reachable, c2 = _ik_reachable(r_mm, h_mm, float(config.l1_mm), float(config.l2_mm))
    yaw_reachable = float(config.yaw_min_deg) <= yaw_deg <= float(config.yaw_max_deg)
    reachable = bool(ik_reachable and yaw_reachable)
    packet = "<C %.2f;%.2f;%.2f;%d>" % (r_mm, h_mm, yaw_deg, elbow)

    reasons = []
    if not ik_reachable:
        reasons.append("outside_2link_workspace")
    if not yaw_reachable:
        reasons.append("outside_yaw_range")

    return {
        "schema_version": "intentgrasp.arm_command.v1",
        "status": "ready_to_send" if reachable else "not_reachable",
        "label": label,
        "packet": packet,
        "reachable": reachable,
        "reason": ",".join(reasons) if reasons else "ok",
        "command": {
            "r_mm": r_mm,
            "h_mm": h_mm,
            "yaw_deg": yaw_deg,
            "elbow": elbow,
        },
        "kinematics": {
            "l1_mm": float(config.l1_mm),
            "l2_mm": float(config.l2_mm),
            "c2": c2,
            "yaw_min_deg": float(config.yaw_min_deg),
            "yaw_max_deg": float(config.yaw_max_deg),
        },
        "transform": {
            "camera_to_base_R": np.asarray(config.camera_to_base_R, dtype=float).reshape(3, 3).tolist(),
            "camera_to_base_t_m": np.asarray(config.camera_to_base_t_m, dtype=float).reshape(3).tolist(),
            "target_base_m": point_base.tolist(),
        },
        "firmware_protocol": "<C r;h;yaw;elbow>",
        "safety_note": "This command assumes camera-to-base calibration and matching servo geometry. Use serial mode only after validating the plan.",
    }


def compute_arm_command_from_camera_point(
    point_m: Sequence[float],
    config: ArmCommandConfig,
    elbow: int = 1,
    label: str = "camera_point",
) -> Dict[str, Any]:
    point_base = transform_camera_point_to_base(point_m, config)
    command = compute_arm_command_from_base_point(point_base, config, elbow=elbow, label=label)
    command["transform"]["target_camera_m"] = np.asarray(point_m, dtype=float).tolist()
    return command

"""
Convert a camera-frame 6D grasp pose into the arm firmware packet.

Firmware target: arm_control_v3.
Protocol: <C r;h;yaw;elbow>

The v3 firmware uses physical joint angles:
  angle1 range: 0..90 deg, where 90 deg is vertical up.
  angle2 range: 0..120 deg, where 0 deg means L2 is straight with L1.

The firmware IK returns DH angles first:
  physical_angle1 = dh_a1
  physical_angle2 = A2_STRAIGHT_PHYS - dh_a2

This module mirrors that check before sending, so the PC does not send an
elbow solution that the firmware will reject as "IK out of joint range".
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


# Camera -> base transform for the fixed capture pose.
# Base axes: X=front, Y=left, Z=up. RealSense axes: x=right, y=down, z=depth.
# Capture pose: camera origin is 40 mm in front of point A; point A is 160 mm above the base pivot.
# Optical axis is forward/down 45 deg.
R_bc = np.array(
    [
        [0.0, -0.70710678, 0.70710678],
        [-1.0, 0.0, 0.0],
        [0.0, -0.70710678, -0.70710678],
    ],
    dtype=float,
)
t_bc = np.array([0.04, 0.0, 0.16], dtype=float)  # camera origin in base frame, meters


# Arm geometry and v3 physical joint limits.
L1 = 130.0
L2 = 200.0
A1_PHYS_MIN = 0.0
A1_PHYS_MAX = 90.0
A2_PHYS_MIN = 0.0
A2_PHYS_MAX = 120.0
A2_STRAIGHT_PHYS = 0.0
YAW_MIN_DEG = -90.0
YAW_MAX_DEG = 90.0


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _ik2link_v3(r_mm: float, h_mm: float, elbow: int) -> Dict[str, float | bool | int | str]:
    """Mirror arm_control_v3.ino ik2link() and physical-angle conversion."""
    reach2 = r_mm * r_mm + h_mm * h_mm
    c2 = (reach2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        return {
            "elbow": int(elbow),
            "reachable_by_length": False,
            "joint_range_ok": False,
            "reason": "outside_arm_reach",
            "c2": float(c2),
        }

    s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow < 0:
        s2 = -s2

    dh_a2 = math.degrees(math.atan2(s2, c2))
    dh_a1 = math.degrees(math.atan2(h_mm, r_mm) - math.atan2(L2 * s2, L1 + L2 * c2))
    phys_a1 = dh_a1
    phys_a2 = A2_STRAIGHT_PHYS - dh_a2
    joint_range_ok = (
        A1_PHYS_MIN <= phys_a1 <= A1_PHYS_MAX
        and A2_PHYS_MIN <= phys_a2 <= A2_PHYS_MAX
    )
    return {
        "elbow": int(elbow),
        "reachable_by_length": True,
        "joint_range_ok": bool(joint_range_ok),
        "reason": "ok" if joint_range_ok else "outside_v3_joint_range",
        "c2": float(c2),
        "dh_a1_deg": float(dh_a1),
        "dh_a2_deg": float(dh_a2),
        "angle1_phys_deg": float(phys_a1),
        "angle2_phys_deg": float(phys_a2),
    }


def _choose_v3_elbow(r_mm: float, h_mm: float, preferred_elbow: int) -> tuple[Optional[dict], List[dict]]:
    candidates: List[dict] = []
    for elbow in (preferred_elbow, -preferred_elbow):
        if any(item["elbow"] == elbow for item in candidates):
            continue
        candidates.append(_ik2link_v3(r_mm, h_mm, elbow))

    for candidate in candidates:
        if candidate.get("reachable_by_length") and candidate.get("joint_range_ok"):
            return candidate, candidates
    return None, candidates


def grasp_to_arm(t_cam, quat, standoff_mm=40.0, approach_axis=0):
    """
    Convert one grasp pose to a v3 arm command.

    t_cam: 3-vector in meters, camera frame.
    quat: quaternion (qx, qy, qz, qw).
    """
    t_cam = np.asarray(t_cam, dtype=float)
    R_cam = quat_to_R(*quat)
    approach_cam = R_cam[:, approach_axis]

    p_base = R_bc @ t_cam + t_bc
    approach_base = R_bc @ approach_cam
    p_pre = p_base - (standoff_mm / 1000.0) * approach_base

    X, Y, Z = p_pre * 1000.0
    yaw_deg = math.degrees(math.atan2(Y, X))
    r_mm = math.hypot(X, Y)
    h_mm = float(Z)

    preferred_elbow = 1 if approach_base[2] <= 0 else -1
    chosen, candidates = _choose_v3_elbow(r_mm, h_mm, preferred_elbow)
    yaw_ok = YAW_MIN_DEG <= yaw_deg <= YAW_MAX_DEG
    reachable = bool(chosen) and yaw_ok
    elbow = int(chosen["elbow"]) if chosen else int(preferred_elbow)
    packet = "<C %.2f;%.2f;%.2f;%d>" % (r_mm, h_mm, yaw_deg, elbow)

    if not yaw_ok:
        reason = "outside_yaw_range"
    elif chosen:
        reason = "ok"
    elif any(candidate.get("reachable_by_length") for candidate in candidates):
        reason = "outside_v3_joint_range"
    else:
        reason = "outside_arm_reach"

    return {
        "yaw_deg": float(yaw_deg),
        "r_mm": float(r_mm),
        "h_mm": float(h_mm),
        "elbow": elbow,
        "preferred_elbow": int(preferred_elbow),
        "reachable": reachable,
        "reason": reason,
        "packet": packet,
        "joint_model": "arm_control_v3",
        "joint_angles": chosen,
        "ik_candidates": candidates,
    }


if __name__ == "__main__":
    t_cam = [-0.13592329621315002, -0.018345916643738747, 0.43700000643730164]
    quat = [0.33771167410005865, -0.4692806285216222, 0.21354892390463523, 0.7874791260535476]
    cmd = grasp_to_arm(t_cam, quat, standoff_mm=40.0)
    print("packet:", cmd["packet"])
    print("reachable:", cmd["reachable"], cmd["reason"])
    print("chosen joints:", cmd["joint_angles"])
    print("candidates:", cmd["ik_candidates"])

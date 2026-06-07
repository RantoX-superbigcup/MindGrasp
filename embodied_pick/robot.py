from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, List, Optional, Sequence

from .schemas import ExecutionResult, PlanResult


@dataclass(frozen=True)
class JointAngles:
    motor1: int
    motor2: int
    motor3: int
    motor4: int
    motor5: int

    @classmethod
    def from_sequence(cls, values: Sequence[float | int]) -> "JointAngles":
        if len(values) != 5:
            raise ValueError("JointAngles requires exactly 5 motor angles")
        clamped = [_clamp_servo_angle(value) for value in values]
        return cls(*clamped)

    def as_list(self) -> List[int]:
        return [self.motor1, self.motor2, self.motor3, self.motor4, self.motor5]


def _clamp_servo_angle(value: float | int) -> int:
    return max(0, min(180, int(round(float(value)))))


class DryRunRobotController:
    def execute(self, plan: PlanResult) -> ExecutionResult:
        if not plan.success:
            return ExecutionResult(success=False, reason=plan.reason)
        return ExecutionResult(success=True, reason="dry-run execution", telemetry={"steps": plan.trajectory})

    def emergency_stop(self) -> ExecutionResult:
        return ExecutionResult(success=True, reason="dry-run emergency stop")


class SerialArmController:
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        protocol: str = "joint_angles",
        disabled_joints: Optional[Iterable[int]] = None,
        home_angles: Optional[Sequence[int]] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol
        self.disabled_joints = set(disabled_joints or [])
        self.home_angles = JointAngles.from_sequence(home_angles or [90, 90, 90, 90, 90])
        self._serial = None

    def connect(self) -> None:
        if self._serial is not None:
            return
        import serial

        self._serial = serial.Serial(self.port, self.baudrate, timeout=2)
        time.sleep(2.0)

    def execute(self, plan: PlanResult) -> ExecutionResult:
        if not plan.success or plan.selected_grasp is None:
            return ExecutionResult(success=False, reason=plan.reason)
        self.connect()
        command = self._to_arm_command(plan)
        self._serial.write(command.encode("ascii"))
        self._serial.flush()
        return ExecutionResult(
            success=True,
            reason="serial command sent",
            telemetry={
                "command": command.strip(),
                "disabled_joints": sorted(self.disabled_joints),
            },
        )

    def send_joint_angles(self, angles: JointAngles | Sequence[int]) -> ExecutionResult:
        self.connect()
        joint_angles = angles if isinstance(angles, JointAngles) else JointAngles.from_sequence(angles)
        command = self._format_joint_angles(joint_angles)
        self._serial.write(command.encode("ascii"))
        self._serial.flush()
        return ExecutionResult(
            success=True,
            reason="joint angle command sent",
            telemetry={
                "command": command.strip(),
                "disabled_joints": sorted(self.disabled_joints),
            },
        )

    def emergency_stop(self) -> ExecutionResult:
        if self._serial is not None:
            self._serial.write(b"<STOP>\n")
            self._serial.flush()
        return ExecutionResult(success=True, reason="emergency stop sent")

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _to_arm_command(self, plan: PlanResult) -> str:
        if self.protocol in {"joint_angles", "joint_degrees", "joint_angles_dot"}:
            return self._format_joint_angles(self._plan_to_joint_angles(plan))

        grasp = plan.selected_grasp
        x, y, _ = grasp.pose.position
        beta_deg = math.degrees(math.atan2(y, x)) if abs(x) + abs(y) > 1e-6 else 0.0
        l4_mm = float(grasp.depth or 50.0)
        if self.protocol == "beta_l4":
            return f"<{beta_deg:.2f};{l4_mm:.2f}>\n"
        raise ValueError(f"Unsupported robot protocol: {self.protocol}")

    def _format_joint_angles(self, angles: JointAngles | Sequence[int]) -> str:
        joint_angles = angles if isinstance(angles, JointAngles) else JointAngles.from_sequence(angles)
        values = joint_angles.as_list()
        if self.protocol in {"joint_angles", "joint_degrees"}:
            return "<J;" + ";".join(str(value) for value in values) + ">\n"
        if self.protocol == "joint_angles_dot":
            return "<" + ".".join(str(value) for value in values) + ">\n"
        raise ValueError(f"Unsupported robot protocol: {self.protocol}")

    def _plan_to_joint_angles(self, plan: PlanResult) -> JointAngles:
        """Temporary mapping before real inverse kinematics is calibrated."""
        grasp = plan.selected_grasp
        values = self.home_angles.as_list()
        x, y, _ = grasp.pose.position

        if 5 not in self.disabled_joints and abs(x) + abs(y) > 1e-6:
            values[4] = _clamp_servo_angle(90.0 + math.degrees(math.atan2(y, x)))
        if 2 not in self.disabled_joints and grasp.width is not None:
            values[1] = self._width_to_gripper_angle(grasp.width)

        return JointAngles.from_sequence(values)

    @staticmethod
    def _width_to_gripper_angle(width_m: float) -> int:
        width_m = max(0.0, min(0.08, float(width_m)))
        return _clamp_servo_angle(30.0 + (width_m / 0.08) * 60.0)
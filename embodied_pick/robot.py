from __future__ import annotations

import math
import time
from typing import Optional

from .schemas import ExecutionResult, PlanResult


class DryRunRobotController:
    def execute(self, plan: PlanResult) -> ExecutionResult:
        if not plan.success:
            return ExecutionResult(success=False, reason=plan.reason)
        return ExecutionResult(success=True, reason="dry-run execution", telemetry={"steps": plan.trajectory})

    def emergency_stop(self) -> ExecutionResult:
        return ExecutionResult(success=True, reason="dry-run emergency stop")


class SerialArmController:
    def __init__(self, port: str, baudrate: int = 9600, protocol: str = "beta_l4") -> None:
        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol
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
        return ExecutionResult(success=True, reason="serial command sent", telemetry={"command": command.strip()})

    def emergency_stop(self) -> ExecutionResult:
        if self._serial is not None:
            self._serial.write(b"<STOP;0>\n")
            self._serial.flush()
        return ExecutionResult(success=True, reason="emergency stop sent")

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _to_arm_command(self, plan: PlanResult) -> str:
        grasp = plan.selected_grasp
        x, y, _ = grasp.pose.position
        beta_deg = math.degrees(math.atan2(y, x)) if abs(x) + abs(y) > 1e-6 else 0.0
        l4_mm = float(grasp.depth or 50.0)
        if self.protocol == "beta_l4":
            return f"<{beta_deg:.2f};{l4_mm:.2f}>\n"
        raise ValueError(f"Unsupported robot protocol: {self.protocol}")
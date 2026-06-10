from __future__ import annotations

import time
from typing import Iterable, List, Tuple


class ArmLink:
    def __init__(self, port: str = "COM3", baud: int = 115200, disable_reset: bool = True):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for serial arm execution: pip install pyserial") from exc

        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = int(baud)
        self.ser.timeout = 0
        self.ser.write_timeout = 1
        if disable_reset:
            self.ser.dtr = False
            self.ser.rts = False
        self.ser.open()
        self._rx = ""
        if not disable_reset:
            time.sleep(2.0)
        self.flush_input()

    def send(self, packet: str) -> None:
        if not packet.endswith("\n"):
            packet += "\n"
        self.ser.write(packet.encode("ascii"))
        self.ser.flush()

    def poll_lines(self) -> List[str]:
        data = self.ser.read(self.ser.in_waiting or 1)
        lines: List[str] = []
        if data:
            self._rx += data.decode("ascii", errors="ignore")
            while "\n" in self._rx:
                line, self._rx = self._rx.split("\n", 1)
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def flush_input(self) -> None:
        self.ser.reset_input_buffer()
        self._rx = ""

    def send_and_wait(
        self,
        packet: str,
        done_key: str = "Traj done",
        err_keys: Iterable[str] = ("ERR", "BUSY"),
        timeout: float = 8.0,
    ) -> Tuple[bool, List[str]]:
        self.flush_input()
        self.send(packet)
        t0 = time.time()
        collected: List[str] = []
        err_keys = tuple(err_keys)
        while time.time() - t0 < float(timeout):
            for line in self.poll_lines():
                collected.append(line)
                if done_key in line:
                    return True, collected
                if any(key in line for key in err_keys):
                    return False, collected
            time.sleep(0.002)
        return False, collected

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

"""
Serial transport for the arm firmware.

The firmware protocol used by this project is:
    <C r;h;yaw;elbow>

Opening an Arduino serial port can reset the board. To avoid losing the first
packet during startup, ArmLink waits for the firmware "Ready." banner, clears
stale input, then sends the command. If the firmware reports a malformed packet
once, the command is resent after a short delay.
"""

import time

import serial  # pyserial

try:
    from .grasp_to_arm import grasp_to_arm
except ImportError:
    from grasp_to_arm import grasp_to_arm


class ArmLink:
    def __init__(self, port="COM3", baud=115200, disable_reset=False, ready_timeout=3.0):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0
        self.ser.write_timeout = 1

        if disable_reset:
            self.ser.dtr = False
            self.ser.rts = False

        self.ser.open()
        self._rx = ""
        self._startup_lines = []
        self._ready_checked = False
        self.ready_seen = False

        if not disable_reset:
            self.wait_ready(timeout=ready_timeout)
        else:
            time.sleep(0.2)

        self.flush_input()

    def send(self, packet: str):
        """Send one firmware packet and force it out of the OS buffer."""
        if not packet.endswith("\n"):
            packet += "\n"
        payload = packet.encode("ascii")
        written = self.ser.write(payload)
        if written != len(payload):
            raise RuntimeError(f"serial partial write: {written}/{len(payload)} bytes")
        self.ser.flush()

    def poll_lines(self):
        """Return complete lines that are currently available without blocking."""
        data = self.ser.read(self.ser.in_waiting or 1)
        lines = []
        if data:
            self._rx += data.decode("ascii", errors="ignore")
            while "\n" in self._rx:
                line, self._rx = self._rx.split("\n", 1)
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def flush_input(self):
        self.ser.reset_input_buffer()
        self._rx = ""

    def wait_ready(self, timeout=3.0, ready_key="Ready."):
        """Wait once for the Arduino startup banner after opening the port."""
        if self._ready_checked:
            return self.ready_seen, list(self._startup_lines)

        self._ready_checked = True
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() < deadline:
            for line in self.poll_lines():
                self._startup_lines.append(line)
                if ready_key in line:
                    self.ready_seen = True
                    tail_deadline = time.time() + 0.1
                    while time.time() < tail_deadline:
                        self._startup_lines.extend(self.poll_lines())
                        time.sleep(0.002)
                    return True, list(self._startup_lines)
            time.sleep(0.002)

        return False, list(self._startup_lines)

    def send_and_wait(
        self,
        packet: str,
        done_key="Traj done",
        err_keys=("ERR", "BUSY"),
        timeout=5.0,
        retry_malformed=True,
        retry_busy=True,
        retry_no_response=True,
        retry_delay=0.3,
        busy_retry_delay=1.0,
    ):
        """
        Send a packet and wait for firmware confirmation.

        Returns (ok: bool, lines: list[str]). Startup banner lines are included
        in the returned log to make serial-state diagnosis easier.
        """
        collected = list(self._startup_lines)

        def once():
            self.flush_input()
            time.sleep(0.05)
            self.send(packet)
            deadline = time.time() + max(0.0, float(timeout))
            lines = []
            error_line = ""
            command_started = False
            while time.time() < deadline:
                for line in self.poll_lines():
                    lines.append(line)
                    if "Traj start" in line:
                        command_started = True
                    if done_key in line:
                        return True, lines, error_line
                    if command_started and "malformed packet" in line.lower():
                        continue
                    if command_started and "BUSY" in line:
                        continue
                    if any(key in line for key in err_keys):
                        error_line = line
                        return False, lines, error_line
                time.sleep(0.002)
            return False, lines, error_line

        ok, lines, error_line = once()
        collected.extend(lines)
        if ok:
            return True, collected

        if retry_malformed and "malformed packet" in error_line.lower():
            time.sleep(max(0.0, float(retry_delay)))
            collected.append("PC: retry after ERR: malformed packet")
            ok, lines, _error_line = once()
            collected.extend(lines)
            return bool(ok), collected

        if retry_busy and "BUSY" in error_line:
            time.sleep(max(0.0, float(busy_retry_delay)))
            collected.append("PC: retry after BUSY")
            ok, lines, _error_line = once()
            collected.extend(lines)
            return bool(ok), collected

        if retry_no_response and not error_line and not lines:
            time.sleep(max(0.0, float(retry_delay)))
            collected.append("PC: retry after serial timeout with no response")
            ok, lines, _error_line = once()
            collected.extend(lines)
            return bool(ok), collected

        return False, collected

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    t_cam = [-0.13592329621315002, -0.018345916643738747, 0.43700000643730164]
    quat = [
        0.33771167410005865,
        -0.4692806285216222,
        0.21354892390463523,
        0.7874791260535476,
    ]

    cmd = grasp_to_arm(t_cam, quat, standoff_mm=40.0)
    print("packet:", cmd["packet"], "| reachable:", cmd["reachable"])

    if not cmd["reachable"]:
        print("target is outside the arm workspace; not sent")
        raise SystemExit

    arm = ArmLink(port="COM3", baud=115200, disable_reset=False)
    try:
        ok, log = arm.send_and_wait(cmd["packet"], timeout=8.0)
        for line in log:
            print("  <arm>", line)
        print("success" if ok else "failed/timeout")
    finally:
        arm.close()

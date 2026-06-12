import importlib.util
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ipc_socket.tcp_socket_client import HNNKTcpSocketClient
from app.titile_bar import CustomTitleBar
from app.interaction_state import (
    CAPTURING_STAGE,
    CONFIRM_SELECTION_STAGE,
    EMERGENCY_CONFIRM_STAGE,
    EMERGENCY_STAGE,
    OBJECT_SELECTION_STAGE,
    STOPPED_STAGE,
    build_platform_object_options,
    collect_exclusion_terms,
    confirm_options,
    emergency_confirm_options,
    emergency_options,
)
from qwen_config import ensure_qwen_environment, resolve_qwen_base_url, resolve_qwen_model


PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
DEMO_FRAME_DIR = PROJECT_ROOT / "third_party" / "graspnet-baseline" / "doc" / "example_data"
DEFAULT_OPTIONS_JSON = PROJECT_ROOT / "outputs" / "target_grasp" / "qwen_options.json"
PLATFORM_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "platform_grasp"
REALSENSE_OPTIONS_DIR = PROJECT_ROOT / "outputs" / "realsense_frontend"
BCI_GATE_STATE_PATH = PLATFORM_OUTPUT_DIR / "bci_gate.json"
for local_source in [
    PROJECT_ROOT / "third_party" / "GroundingDINO",
    PROJECT_ROOT / "third_party" / "segment-anything",
]:
    if local_source.exists() and str(local_source) not in sys.path:
        sys.path.insert(0, str(local_source))


def read_options_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"question": "未找到选项文件", "options": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("qwen_options", data)


def env_project_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def command_to_choice(result: Any) -> Optional[str]:
    if isinstance(result, int):
        if 1 <= result <= 8:
            return chr(ord("A") + result - 1)
        return None
    text = str(result).strip().upper()
    if text.isdigit():
        index = int(text)
        if 1 <= index <= 8:
            return chr(ord("A") + index - 1)
    match = re.search(r"[A-H]", text)
    return match.group(0) if match else None


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return PROJECT_ROOT


def prepare_portable_runtime(runtime_python: Path) -> None:
    runtime_root = runtime_python.parent
    unpacker = runtime_root / "Scripts" / "conda-unpack.exe"
    if not unpacker.exists():
        return

    marker = runtime_root / ".mindgrasp_runtime_ready"
    runtime_path = str(runtime_root.resolve())
    try:
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == runtime_path:
            return
        subprocess.run(
            [str(unpacker)],
            cwd=str(runtime_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        marker.write_text(runtime_path, encoding="utf-8")
    except Exception as exc:
        print(f"prepare_portable_runtime failed: {exc}")


def python_executable() -> str:
    configured = os.getenv("MINDGRASP_PYTHON", "").strip()
    if configured:
        return configured

    if getattr(sys, "frozen", False):
        base_dir = app_base_dir()
        portable_candidates = [
            base_dir / "runtime" / "python" / "python.exe",
            base_dir / "runtime" / "python.exe",
            base_dir / "python" / "python.exe",
        ]
        for candidate in portable_candidates:
            if candidate.exists():
                prepare_portable_runtime(candidate)
                return str(candidate)
        return "python"

    return sys.executable

def run_target_grasp_in_process(argv: List[str]) -> str:
    import run_target_grasp_demo as target_demo

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        target_demo.main(argv)
    return stdout.getvalue() + stderr.getvalue()


def create_pose_overlay(summary_path: Path, output_dir: Path, frame_dir: Path) -> Optional[Path]:
    source = output_dir / "target_overlay.png"
    if not source.exists() or not summary_path.exists():
        return None

    try:
        from PIL import Image, ImageDraw

        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        top_grasps = summary.get("top_grasps", [])[:3]
        endpoints_by_rank = _project_grasp_axes(top_grasps, frame_dir / "meta.mat")
        colors = [(255, 40, 40), (50, 180, 255), (255, 180, 40)]

        for idx, grasp in enumerate(top_grasps):
            uv = grasp.get("projected_uv") or []
            if len(uv) != 2:
                continue
            x, y = float(uv[0]), float(uv[1])
            color = colors[idx % len(colors)]
            endpoints = endpoints_by_rank.get(idx)
            if endpoints:
                (x1, y1), (x2, y2) = endpoints
                draw.line([(x1, y1), (x2, y2)], fill=color, width=5)
                draw.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), outline=color, width=2)
                draw.ellipse((x2 - 4, y2 - 4, x2 + 4, y2 + 4), outline=color, width=2)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=(255, 255, 255), width=2)
            score = float(grasp.get("score", 0.0) or 0.0)
            draw.text((x + 10, y - 18), f"#{idx + 1} {score:.3f}", fill=color)

        overlay_path = output_dir / "grasp_pose_overlay.png"
        image.save(overlay_path)
        return overlay_path
    except Exception as exc:
        print(f"create_pose_overlay failed: {exc}")
        return None


def _project_grasp_axes(top_grasps: List[Dict[str, Any]], meta_path: Path) -> Dict[int, Any]:
    if not meta_path.exists():
        return {}
    try:
        import numpy as np
        import scipy.io as scio

        intrinsic = scio.loadmat(meta_path)["intrinsic_matrix"]
        fx = float(intrinsic[0][0])
        fy = float(intrinsic[1][1])
        cx = float(intrinsic[0][2])
        cy = float(intrinsic[1][2])

        def project(point: np.ndarray) -> Optional[tuple]:
            if float(point[2]) <= 1e-6:
                return None
            return (fx * float(point[0]) / float(point[2]) + cx, fy * float(point[1]) / float(point[2]) + cy)

        projected = {}
        for idx, grasp in enumerate(top_grasps):
            translation = np.asarray(grasp.get("translation"), dtype=float)
            rotation = np.asarray(grasp.get("rotation_matrix"), dtype=float)
            if translation.shape != (3,) or rotation.shape != (3, 3):
                continue
            width = float(grasp.get("width", 0.05) or 0.05)
            axis = rotation[:, 1]
            p1 = project(translation - axis * max(width, 0.04) * 0.5)
            p2 = project(translation + axis * max(width, 0.04) * 0.5)
            if p1 and p2:
                projected[idx] = (p1, p2)
        return projected
    except Exception:
        return {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def _parse_vector(text: str, length: int, name: str) -> "Any":
    import numpy as np

    values = [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]
    if len(values) != length:
        raise ValueError(f"{name} must have {length} numeric values")
    return np.asarray(values, dtype=float)


def _parse_matrix3(text: str, name: str) -> "Any":
    import numpy as np

    rows = []
    for row in str(text).split(";"):
        row = row.strip()
        if row:
            rows.append([float(item.strip()) for item in row.split(",") if item.strip()])
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be 3 rows of 3 comma-separated values")
    return matrix


def build_legacy_arm_command(summary: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    top_grasps = summary.get("top_grasps", [])
    if not top_grasps:
        arm_output = {
            "schema_version": "intentgrasp.arm_command.v1",
            "status": "no_pose",
            "reachable": False,
            "reason": "no top grasp",
        }
        (output_dir / "arm_command.json").write_text(json.dumps(arm_output, ensure_ascii=False, indent=2), encoding="utf-8")
        return arm_output

    import importlib
    import numpy as np

    legacy_arm = importlib.import_module("arm_control.grasp_to_arm")
    legacy_arm.R_bc = _parse_matrix3(
        os.getenv("MINDGRASP_CAMERA_TO_BASE_ROTATION", "0,-0.70710678,0.70710678;-1,0,0;0,-0.70710678,-0.70710678"),
        "MINDGRASP_CAMERA_TO_BASE_ROTATION",
    )
    legacy_arm.t_bc = _parse_vector(
        os.getenv("MINDGRASP_CAMERA_TO_BASE_TRANSLATION_M", "0.04,0,0.16"),
        3,
        "MINDGRASP_CAMERA_TO_BASE_TRANSLATION_M",
    )
    legacy_arm.L1 = float(os.getenv("MINDGRASP_ARM_L1_MM", str(legacy_arm.L1)))
    legacy_arm.L2 = float(os.getenv("MINDGRASP_ARM_L2_MM", str(legacy_arm.L2)))

    top = top_grasps[0]
    command = legacy_arm.grasp_to_arm(
        top["translation"],
        top["quaternion"],
        standoff_mm=float(os.getenv("MINDGRASP_ARM_STANDOFF_MM", "40.0")),
        approach_axis=int(os.getenv("MINDGRASP_ARM_APPROACH_AXIS", "0")),
    )
    reachable = bool(command.get("reachable"))
    arm_output = {
        "schema_version": "intentgrasp.arm_command.v1",
        "status": "ready_to_send" if reachable else "not_reachable",
        "packet": str(command.get("packet") or ""),
        "reachable": reachable,
        "reason": str(command.get("reason") or ("ok" if reachable else "not_reachable")),
        "command": {
            "r_mm": float(command.get("r_mm", 0.0)),
            "h_mm": float(command.get("h_mm", 0.0)),
            "yaw_deg": float(command.get("yaw_deg", 0.0)),
            "elbow": int(command.get("elbow", 1)),
            "preferred_elbow": int(command.get("preferred_elbow", command.get("elbow", 1))),
        },
        "kinematics": {
            "l1_mm": float(legacy_arm.L1),
            "l2_mm": float(legacy_arm.L2),
            "joint_model": str(command.get("joint_model") or "arm_control_v3"),
            "joint_angles": command.get("joint_angles"),
            "ik_candidates": command.get("ik_candidates"),
        },
        "transform": {
            "camera_to_base_R": np.asarray(legacy_arm.R_bc, dtype=float).tolist(),
            "camera_to_base_t_m": np.asarray(legacy_arm.t_bc, dtype=float).tolist(),
            "camera_mount": os.getenv("MINDGRASP_CAMERA_MOUNT", "front_of_point_a"),
            "capture_pose": {
                "angle1_deg": float(os.getenv("MINDGRASP_CAPTURE_ANGLE1_DEG", "90.0")),
                "angle2_deg": float(os.getenv("MINDGRASP_CAPTURE_ANGLE2_DEG", "0.0")),
                "camera_height_m": float(os.getenv("MINDGRASP_CAPTURE_CAMERA_HEIGHT_M", "0.16")),
                "camera_forward_offset_m": float(os.getenv("MINDGRASP_CAPTURE_CAMERA_FORWARD_OFFSET_M", "0.04")),
                "camera_pitch_down_deg": float(os.getenv("MINDGRASP_CAPTURE_CAMERA_PITCH_DOWN_DEG", "45.0")),
            },
        },
        "converter": "arm_control.grasp_to_arm.grasp_to_arm",
        "serial_adapter": "arm_control.arm_serial.ArmLink",
        "firmware_protocol": "<C r;h;yaw;elbow>",
    }
    (output_dir / "arm_command.json").write_text(json.dumps(arm_output, ensure_ascii=False, indent=2), encoding="utf-8")
    return arm_output


def execute_legacy_arm_command(arm_output: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    mode = os.getenv("MINDGRASP_ARM_MODE", "serial").strip().lower()
    port = os.getenv("MINDGRASP_ARM_PORT", "COM7").strip() or "COM7"
    baud = int(os.getenv("MINDGRASP_ARM_BAUD", "115200"))
    timeout = float(os.getenv("MINDGRASP_ARM_TIMEOUT", "8.0"))
    ready_timeout = float(os.getenv("MINDGRASP_ARM_READY_TIMEOUT", "3.0"))
    disable_reset = _env_bool("MINDGRASP_ARM_DISABLE_RESET", False)

    if mode != "serial":
        result = {"mode": mode or "command", "sent": False, "reason": "serial execution disabled"}
        (output_dir / "arm_execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    try:
        from arm_control.arm_serial import ArmLink

        if not arm_output.get("reachable"):
            arm = ArmLink(port=port, baud=baud, disable_reset=disable_reset, ready_timeout=ready_timeout)
            try:
                _ok, log = arm.send_and_wait("<PING>", timeout=2.0)
            finally:
                arm.close()
            result = {
                "mode": "serial",
                "port": port,
                "baud": baud,
                "sent": False,
                "success": False,
                "connected": bool(log),
                "reason": f"command not reachable: {arm_output.get('reason')}",
                "probe_log": log,
            }
        else:
            packet = str(arm_output.get("packet") or "")
            arm = ArmLink(port=port, baud=baud, disable_reset=disable_reset, ready_timeout=ready_timeout)
            try:
                ok, log = arm.send_and_wait(packet, timeout=timeout)
            finally:
                arm.close()
            result = {
                "mode": "serial",
                "port": port,
                "baud": baud,
                "sent": True,
                "success": bool(ok),
                "packet": packet,
                "log": log,
                "reason": "done" if ok else "error_or_timeout",
            }
    except Exception as exc:
        result = {
            "mode": "serial",
            "port": port,
            "baud": baud,
            "sent": False,
            "success": False,
            "connected": False,
            "reason": str(exc),
        }

    (output_dir / "arm_execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


class GraspRunWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, choice: str, frame_dir: Path, options_json: Path, output_root: Path, parent=None):
        super().__init__(parent)
        self.choice = choice
        self.frame_dir = frame_dir
        self.options_json = options_json
        self.output_root = output_root
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        output_dir = self.output_root / f"choice_{self.choice.lower()}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_executable(),
            "-B",
            str(PROJECT_ROOT / "run_target_grasp_demo.py"),
            "--frame-dir",
            str(self.frame_dir),
            "--output-dir",
            str(output_dir),
            "--options-json",
            str(self.options_json),
            "--choice",
            self.choice,
            "--localizer",
            "groundingdino",
            "--segmenter",
            "sam",
            "--vis-mode",
            "compare",
            "--top-k",
            "8",
            "--no-vis",
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
        default_cuda = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
        if not cuda_home and default_cuda.exists():
            cuda_home = str(default_cuda)
            env["CUDA_HOME"] = cuda_home
            env["CUDA_PATH"] = cuda_home
        if cuda_home:
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")

        self.status.emit("正在运行 GroundingDINO/SAM -> GraspNet ...")
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                creationflags=creationflags,
            )
            if proc.returncode != 0:
                message = (proc.stderr or proc.stdout or "").strip()
                self.failed.emit(message[-4000:] if message else f"process failed with code {proc.returncode}")
                return
            stdout_text = proc.stdout or ""
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        summary_path = output_dir / "target_grasps.json"
        if not summary_path.exists():
            self.failed.emit(f"未生成 target_grasps.json: {summary_path}")
            return

        pose_overlay = create_pose_overlay(summary_path, output_dir, self.frame_dir)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if self.cancel_requested:
            self.finished.emit(
                {
                    "choice": self.choice,
                    "output_dir": str(output_dir),
                    "summary_path": str(summary_path),
                    "pose_overlay": str(pose_overlay) if pose_overlay else "",
                    "target_overlay": str(output_dir / "target_overlay.png"),
                    "summary": summary,
                    "cancelled": True,
                    "stdout_tail": stdout_text[-2000:],
                }
            )
            return
        self.status.emit("正在转换机械臂指令并连接串口 ...")
        arm_output = build_legacy_arm_command(summary, output_dir)
        if self.cancel_requested:
            self.finished.emit(
                {
                    "choice": self.choice,
                    "output_dir": str(output_dir),
                    "summary_path": str(summary_path),
                    "pose_overlay": str(pose_overlay) if pose_overlay else "",
                    "target_overlay": str(output_dir / "target_overlay.png"),
                    "summary": summary,
                    "arm_output": arm_output,
                    "cancelled": True,
                    "stdout_tail": stdout_text[-2000:],
                }
            )
            return
        arm_execution = execute_legacy_arm_command(arm_output, output_dir)
        self.finished.emit(
            {
                "choice": self.choice,
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
                "pose_overlay": str(pose_overlay) if pose_overlay else "",
                "target_overlay": str(output_dir / "target_overlay.png"),
                "arm_command_path": str(output_dir / "arm_command.json"),
                "arm_execution_path": str(output_dir / "arm_execution.json"),
                "summary": summary,
                "arm_output": arm_output,
                "arm_execution": arm_execution,
                "stdout_tail": stdout_text[-2000:],
            }
        )


class RealSensePrepareWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            from types import SimpleNamespace

            scripts_dir = PROJECT_ROOT / "scripts"
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from capture_realsense_rgbd import capture as capture_realsense
            from run_realsense_grasp_workflow import call_qwen_readable_options
            import run_target_grasp_demo as target_demo

            self.status.emit("正在采集 RealSense RGB-D ...")
            capture_args = SimpleNamespace(
                output_dir=str(PROJECT_ROOT / "captures"),
                width=int(os.getenv("MINDGRASP_RS_WIDTH", "640")),
                height=int(os.getenv("MINDGRASP_RS_HEIGHT", "480")),
                fps=int(os.getenv("MINDGRASP_RS_FPS", "30")),
                serial=os.getenv("MINDGRASP_REALSENSE_SERIAL", "").strip() or None,
                warmup=int(os.getenv("MINDGRASP_RS_WARMUP", "10")),
                depth_frames=int(os.getenv("MINDGRASP_RS_DEPTH_FRAMES", "5")),
                depth_fusion=os.getenv("MINDGRASP_RS_DEPTH_FUSION", "median"),
                depth_preset=os.getenv("MINDGRASP_RS_DEPTH_PRESET", "high_accuracy"),
                emitter=os.getenv("MINDGRASP_RS_EMITTER", "on"),
                laser_power=float(os.getenv("MINDGRASP_RS_LASER_POWER", "360.0")),
                depth_auto_exposure=os.getenv("MINDGRASP_RS_DEPTH_AUTO_EXPOSURE", "on"),
                depth_exposure_us=None,
                enable_filters=os.getenv("MINDGRASP_RS_ENABLE_FILTERS", "1") != "0",
                spatial_magnitude=2.0,
                spatial_alpha=0.5,
                spatial_delta=20.0,
                spatial_holes_fill=2.0,
                temporal_alpha=0.4,
                temporal_delta=20.0,
                hole_filling=True,
                hole_filling_mode=1,
                preview=False,
                timestamp=True,
                overwrite=False,
                workspace_mode=os.getenv("MINDGRASP_RS_WORKSPACE_MODE", "valid"),
                min_depth_m=float(os.getenv("MINDGRASP_RS_MIN_DEPTH_M", "0.20")),
                max_depth_m=float(os.getenv("MINDGRASP_RS_MAX_DEPTH_M", "2.00")),
                workspace_roi=os.getenv("MINDGRASP_RS_WORKSPACE_ROI", "").strip() or None,
                depth_vis_alpha=0.03,
            )
            frame_dir = capture_realsense(capture_args)
            if frame_dir is None:
                self.failed.emit("RealSense 采集未保存帧，请确认相机没有被其它程序占用。")
                return
            frame_dir = Path(frame_dir).resolve()

            api_key = ensure_qwen_environment("QWEN_API_KEY")
            if not api_key:
                self.failed.emit("缺少 QWEN_API_KEY 或 configs/local_secrets.json，无法根据 RealSense RGB 生成目标选项。")
                return

            self.status.emit("正在根据 RealSense RGB 生成目标选项 ...")
            options_result = call_qwen_readable_options(
                frame_dir / "color.png",
                resolve_qwen_base_url(),
                resolve_qwen_model(),
                api_key,
                int(os.getenv("MINDGRASP_MAX_OBJECT_OPTIONS", "3")),
            )
            options_result = build_platform_object_options(options_result)
            if not options_result.get("options"):
                self.failed.emit("当前画面没有生成可选目标。请把相机对准桌面物体后重新采集。")
                return

            REALSENSE_OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S")
            options_json = REALSENSE_OPTIONS_DIR / f"qwen_options_{stem}.json"
            options_json.write_text(json.dumps({"qwen_options": options_result}, ensure_ascii=False, indent=2), encoding="utf-8")
            target_demo.draw_options_overlay(frame_dir / "color.png", options_result, REALSENSE_OPTIONS_DIR / f"qwen_options_overlay_{stem}.png")

            self.finished.emit(
                {
                    "frame_dir": str(frame_dir),
                    "options_json": str(options_json),
                    "options_result": options_result,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class QwenOptionsWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, frame_dir: Path, excluded_options: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.frame_dir = frame_dir
        self.excluded_options = list(excluded_options)

    def run(self) -> None:
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from run_realsense_grasp_workflow import call_qwen_readable_options
            import run_target_grasp_demo as target_demo

            api_key = ensure_qwen_environment("QWEN_API_KEY")
            if not api_key:
                self.failed.emit("缺少 QWEN_API_KEY 或 configs/local_secrets.json，无法重新识别目标选项。")
                return

            excluded_terms = collect_exclusion_terms(self.excluded_options)
            self.status.emit("正在重新识别目标，并排除上一组选项 ...")
            raw_options = call_qwen_readable_options(
                self.frame_dir / "color.png",
                resolve_qwen_base_url(),
                resolve_qwen_model(),
                api_key,
                int(os.getenv("MINDGRASP_MAX_OBJECT_OPTIONS", "3")),
                excluded_targets=excluded_terms,
            )
            options_result = build_platform_object_options(raw_options, excluded_terms=excluded_terms)
            if not options_result.get("options"):
                self.failed.emit("重新识别后没有生成可选目标。")
                return

            REALSENSE_OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S")
            options_json = REALSENSE_OPTIONS_DIR / f"qwen_options_retry_{stem}.json"
            options_json.write_text(json.dumps({"qwen_options": options_result}, ensure_ascii=False, indent=2), encoding="utf-8")
            target_demo.draw_options_overlay(self.frame_dir / "color.png", options_result, REALSENSE_OPTIONS_DIR / f"qwen_options_retry_overlay_{stem}.png")
            self.finished.emit({"options_json": str(options_json), "options_result": options_result, "excluded_terms": excluded_terms})
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.client_socket = HNNKTcpSocketClient()
        self.client_socket.server_connected.connect(self.on_server_connected)
        self.client_socket.server_disconnected.connect(self.on_server_disconnected)
        self.client_socket.recv_from_server.connect(self.on_server_data)

        self.connect_status = False
        self.layout_type = 0
        self.frame_dir = env_project_path("MINDGRASP_FRAME_DIR", DEMO_FRAME_DIR)
        self.options_json_path = env_project_path("MINDGRASP_OPTIONS_JSON", DEFAULT_OPTIONS_JSON)
        self.output_root = PLATFORM_OUTPUT_DIR
        if self.frame_dir != DEMO_FRAME_DIR:
            self.output_root = PLATFORM_OUTPUT_DIR / f"realsense_{self.frame_dir.name}"
        self.options_result = build_platform_object_options(read_options_file(self.options_json_path))
        self.object_options_result = self.options_result
        self.previous_object_options_result: Optional[Dict[str, Any]] = None
        self.pending_choice: Optional[str] = None
        self.pending_option: Optional[Dict[str, Any]] = None
        self.excluded_object_options: List[Dict[str, Any]] = []
        self.bci_active = False
        self.workflow_stopped = False
        self.interaction_stage = OBJECT_SELECTION_STAGE
        self.option_buttons: Dict[str, QPushButton] = {}
        self.image_labels: Dict[str, QLabel] = {}
        self.current_choice: Optional[str] = None
        self.grasp_worker: Optional[GraspRunWorker] = None
        self.realsense_worker: Optional[RealSensePrepareWorker] = None
        self.options_worker: Optional[QThread] = None
        self.log_lines: List[str] = []

        self.init_ui()
        self.show_window_center()
        self.refresh_demo_content()

    def show_window_center(self):
        screen = QDesktopWidget().availableGeometry()
        target_width = min(1720, int(screen.width() * 0.94))
        target_height = min(980, int(screen.height() * 0.92))
        target_width = min(max(1400, target_width), screen.width())
        target_height = min(max(860, target_height), screen.height())
        self.resize(target_width, target_height)
        x = screen.x() + (screen.width() - target_width) // 2
        y = screen.y() + (screen.height() - target_height) // 2
        self.setGeometry(x, y, target_width, target_height)
        self.show()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMinimumSize(1280, 800)
        self.setStyleSheet("background-color: #F3F6F9;")

        central_widget = QWidget(self)
        vbox = QVBoxLayout(central_widget)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.title_bar = CustomTitleBar(self)
        vbox.addWidget(self.title_bar)

        client_widget = QWidget()
        client_widget.setStyleSheet("background: #F3F6F9;")
        vbox.addWidget(client_widget)

        connect_widget = self.build_connect_widget()
        content_widget = self.build_content_widget()

        client_box = QVBoxLayout()
        client_box.setContentsMargins(24, 18, 24, 24)
        client_box.setSpacing(16)
        client_box.addWidget(connect_widget, 1)
        client_box.addWidget(content_widget, 6)

        client_widget.setLayout(client_box)
        self.setCentralWidget(central_widget)

    def build_connect_widget(self) -> QWidget:
        connect_widget = QFrame()
        connect_widget.setObjectName("connectBar")
        connect_widget.setFixedHeight(96)
        connect_widget.setStyleSheet(
            """
            #connectBar {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            """
        )
        connect_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        connect_hbox = QHBoxLayout()
        connect_hbox.setContentsMargins(24, 14, 24, 14)
        connect_hbox.setSpacing(16)

        self.server_ip_lineedit = self.build_labeled_lineedit(connect_hbox, "服务器地址", "127.0.0.1")
        self.server_port_lineedit = self.build_labeled_lineedit(connect_hbox, "端口", "8000")

        server_btn_widget = QWidget()
        server_btn_vbox = QVBoxLayout()
        server_btn_vbox.setContentsMargins(0, 30, 0, 0)
        server_btn_hbox = QHBoxLayout()
        server_btn_hbox.setSpacing(10)
        server_btn_hbox.setContentsMargins(0, 0, 0, 0)

        self.connect_button = QPushButton("连接")
        self.disconnect_button = QPushButton("断开")
        self.capture_button = QPushButton("采集 RealSense")
        self.disconnect_button.setEnabled(False)
        for button in (self.connect_button, self.disconnect_button, self.capture_button):
            button.setStyleSheet(self.button_style())
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.connect_button.clicked.connect(self.connect_server)
        self.disconnect_button.clicked.connect(self.disconnect_server)
        self.capture_button.clicked.connect(self.capture_realsense_frame)

        self.connect_status_label = QLabel("状态: 未连接")
        self.connect_status_label.setFixedHeight(16)
        self.connect_status_label.setAlignment(Qt.AlignCenter)
        self.connect_status_label.setStyleSheet(self.muted_label_style())

        server_btn_hbox.addWidget(self.connect_button, 1)
        server_btn_hbox.addWidget(self.disconnect_button, 1)
        server_btn_hbox.addWidget(self.capture_button, 1)
        server_btn_hbox.addWidget(self.connect_status_label, 1)
        server_btn_vbox.addLayout(server_btn_hbox)
        server_btn_widget.setLayout(server_btn_vbox)

        connect_hbox.addWidget(server_btn_widget, 1)
        connect_widget.setLayout(connect_hbox)
        return connect_widget

    def build_labeled_lineedit(self, parent_layout: QHBoxLayout, label_text: str, value: str) -> QLineEdit:
        wrapper = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        label.setFixedHeight(16)
        label.setStyleSheet(self.muted_label_style())
        lineedit = QLineEdit(value)
        lineedit.setStyleSheet(
            """
            QLineEdit {
                background-color: #F8FAFC;
                border: 1px solid #DDE5EF;
                border-radius: 6px;
                padding-left: 12px;
                font-family: SourceHanSansCN, SourceHanSansCN;
                font-weight: 400;
                font-size: 17px;
                color: #20242A;
            }
            """
        )
        lineedit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(label)
        layout.addWidget(lineedit)
        wrapper.setLayout(layout)
        parent_layout.addWidget(wrapper, 1)
        return lineedit

    def build_content_widget(self) -> QWidget:
        content_widget = QFrame()
        content_widget.setObjectName("contentFrame")
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_widget.setStyleSheet(
            """
            #contentFrame {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            QLabel {
                color: #20242A;
                font-family: SourceHanSansCN, SourceHanSansCN;
            }
            """
        )
        grid = QGridLayout(content_widget)
        grid.setContentsMargins(18, 16, 18, 16)
        grid.setSpacing(12)

        header = QFrame()
        header.setObjectName("workflowHeader")
        header.setStyleSheet(
            """
            #workflowHeader {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            """
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(12)

        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(3)
        title = QLabel("脑机目标选择与抓取姿态预览")
        title.setStyleSheet("font-size: 19px; font-weight: 600; color: #20242A;")
        subtitle = QLabel("RGB 场景 | 目标选项 | SAM mask + GraspNet 位姿")
        subtitle.setStyleSheet("font-size: 13px; color: #6B7280;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addWidget(title_block, 1)

        self.selection_badge = self.build_badge("当前选择", "待选择", "#EAF2FF", "#1F5A96")
        self.chain_badge = self.build_badge("链路", "离线预览", "#EAF7EF", "#19724C")
        self.result_badge = self.build_badge("结果", "未运行", "#FFF4E6", "#9A5A00")
        header_layout.addWidget(self.selection_badge)
        header_layout.addWidget(self.chain_badge)
        header_layout.addWidget(self.result_badge)
        grid.addWidget(header, 0, 0, 1, 3)

        self.rgb_label = self.build_image_label("RGB")
        self.pose_overlay_label = self.build_image_label("Mask + Pose")
        self.image_labels = {
            "rgb": self.rgb_label,
            "pose": self.pose_overlay_label,
        }

        grid.addWidget(self.section("RGB 场景"), 1, 0)
        grid.addWidget(self.rgb_label, 2, 0, 3, 1)

        grid.addWidget(self.section("目标选项"), 1, 1)
        self.option_area = QWidget()
        self.option_grid = QGridLayout(self.option_area)
        self.option_grid.setContentsMargins(0, 0, 0, 0)
        self.option_grid.setSpacing(8)
        grid.addWidget(self.option_area, 2, 1, 3, 1)

        grid.addWidget(self.section("Mask + 位姿"), 1, 2)
        grid.addWidget(self.pose_overlay_label, 2, 2, 3, 1)

        self.status_label = QLabel("等待平台 A/B/C/D 指令")
        self.status_label.setStyleSheet(
            "font-size: 15px; color: #1F5A96; background: #F0F6FF; border: 1px solid #D7E8FF; border-radius: 6px; padding: 8px;"
        )
        self.status_label.setWordWrap(True)
        grid.addWidget(self.status_label, 5, 0, 1, 3)

        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        grid.setRowStretch(2, 1)
        grid.setRowStretch(3, 1)
        grid.setRowStretch(4, 1)
        return content_widget

    def build_badge(self, label: str, value: str, bg: str, fg: str) -> QLabel:
        badge = QLabel(f"{label}\n{value}")
        badge.setAlignment(Qt.AlignCenter)
        badge.setMinimumWidth(138)
        badge.setFixedHeight(50)
        badge.setStyleSheet(
            f"""
            QLabel {{
                background: {bg};
                color: {fg};
                border: 1px solid rgba(0, 0, 0, 0.04);
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
                padding: 4px 10px;
            }}
            """
        )
        return badge

    def set_badge(self, badge: QLabel, label: str, value: str) -> None:
        badge.setText(f"{label}\n{value}")

    def section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setFixedHeight(24)
        label.setStyleSheet("font-size: 15px; font-weight: 600; color: #20242A;")
        return label

    def build_image_label(self, name: str) -> QLabel:
        label = QLabel(name)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(QSize(520, 360))
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet(
            """
            QLabel {
                background: #111827;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                color: #CBD5E1;
                font-size: 14px;
            }
            """
        )
        return label

    def refresh_demo_content(self):
        self.set_image("rgb", self.frame_dir / "color.png")
        pose_path = self.default_pose_overlay_path()
        if pose_path:
            self.set_image("pose", pose_path)
        self.show_object_options(self.options_result, activate_bci=True, reason="initial_options")
        self.log("平台在线输出入口: ipc_algorithm_test -> result_args.data")
        missing = self.missing_formal_chain_parts()
        if missing:
            self.log("正式链路缺少依赖/权重: " + "; ".join(missing))

    def default_pose_overlay_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        if self.current_choice:
            candidates.append(self.output_root / f"choice_{self.current_choice.lower()}" / "grasp_pose_overlay.png")
        if self.frame_dir == DEMO_FRAME_DIR:
            candidates.extend(
                [
                    PROJECT_ROOT / "outputs" / "target_grasp_formal" / "grasp_pose_overlay.png",
                    PROJECT_ROOT / "outputs" / "target_grasp" / "grasp_pose_overlay.png",
                ]
            )
        for path in candidates:
            if path.exists():
                return path
        return None

    def render_options(self):
        while self.option_grid.count():
            item = self.option_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.option_buttons.clear()

        options = self.options_result.get("options", [])
        if not options:
            label = QLabel("未找到 qwen_options.json")
            label.setStyleSheet("font-size: 14px; color: #A33;")
            self.option_grid.addWidget(label, 0, 0)
            return

        for idx, option in enumerate(options):
            key = str(option.get("key", "")).upper()
            label = str(option.get("label", option.get("target_id", "")))
            desc = str(option.get("description", ""))
            short_desc = desc if len(desc) <= 28 else desc[:27] + "..."
            button = QPushButton(f"{key}  {label}\n{short_desc}")
            button.setToolTip(desc)
            button.setMinimumHeight(58)
            button.setStyleSheet(self.option_button_style(selected=False))
            if option.get("enabled", True) is False:
                button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, selected_key=key: self.select_target(selected_key, "本地按钮"))
            self.option_buttons[key] = button
            self.option_grid.addWidget(button, idx // 2, idx % 2)

    def persist_current_options(self) -> None:
        try:
            self.options_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.options_json_path.write_text(json.dumps({"qwen_options": self.options_result}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.log(f"保存选项文件失败: {exc}")

    def publish_bci_gate(self, active: bool, reason: str = "") -> None:
        payload = {
            "msg": "mindgrasp_bci_gate",
            "active": bool(active),
            "mode": "online" if active else "stop",
            "stage": self.interaction_stage,
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            BCI_GATE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            BCI_GATE_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.log(f"写入 BCI gate 状态失败: {exc}")
        if self.connect_status:
            try:
                self.client_socket.send_to_server(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:
                self.log(f"发送 BCI gate 状态失败: {exc}")

    def set_bci_gate(self, active: bool, reason: str = "") -> None:
        self.bci_active = bool(active)
        self.publish_bci_gate(active, reason)
        self.log(f"BCI gate -> {'online' if active else 'stop'} ({self.interaction_stage}; {reason})")

    def show_object_options(self, options_result: Dict[str, Any], activate_bci: bool, reason: str) -> None:
        self.interaction_stage = OBJECT_SELECTION_STAGE
        self.options_result = build_platform_object_options(options_result, collect_exclusion_terms(self.excluded_object_options))
        self.object_options_result = self.options_result
        self.current_choice = None
        self.workflow_stopped = False
        self.render_options()
        self.persist_current_options()
        self.set_badge(self.selection_badge, "当前选择", "待选择")
        self.set_badge(self.chain_badge, "链路", "等待选择")
        self.set_badge(self.result_badge, "结果", "未运行")
        self.status_label.setText("请选择目标 A/B/C；如果没有想要的目标请选择 D。")
        self.set_bci_gate(activate_bci, reason)

    def show_confirm_options(self) -> None:
        self.interaction_stage = CONFIRM_SELECTION_STAGE
        self.options_result = confirm_options()
        self.current_choice = None
        self.render_options()
        label = self.pending_option.get("label", self.pending_choice) if self.pending_option else self.pending_choice
        self.set_badge(self.selection_badge, "待确认", f"{self.pending_choice} {label}")
        self.set_badge(self.chain_badge, "链路", "等待确认")
        self.set_badge(self.result_badge, "结果", "未运行")
        self.status_label.setText(f"已选择 {self.pending_choice}: {label}。请选择 A 重试、B 停止、C 确认。")
        self.set_bci_gate(True, "confirm_options_ready")

    def show_emergency_options(self) -> None:
        self.interaction_stage = EMERGENCY_STAGE
        self.options_result = emergency_options()
        self.current_choice = None
        self.render_options()
        self.set_badge(self.chain_badge, "链路", "运行中")
        self.status_label.setText("已确认目标并开始运行。运行期间仅急停选项会触发中断，其余为干扰项。")
        self.set_bci_gate(True, "emergency_options_ready")

    def show_emergency_confirm_options(self) -> None:
        self.interaction_stage = EMERGENCY_CONFIRM_STAGE
        self.options_result = emergency_confirm_options()
        self.current_choice = None
        self.render_options()
        self.status_label.setText("已选择急停。请选择 A 确认急停，或 B 取消。")
        self.set_bci_gate(True, "emergency_confirm_ready")

    def set_image(self, name: str, path: Path):
        label = self.image_labels.get(name)
        if label is None:
            return
        if not path.exists():
            label.setText(f"缺少文件\n{path.name}")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText(f"无法加载\n{path.name}")
            return
        size = label.size()
        if size.width() < 20 or size.height() < 20:
            size = QSize(360, 220)
        label.setPixmap(pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_image("rgb", self.frame_dir / "color.png")
        pose_path = self.default_pose_overlay_path()
        if pose_path:
            self.set_image("pose", pose_path)

    def capture_realsense_frame(self):
        if self.realsense_worker is not None and self.realsense_worker.isRunning():
            return
        if self.grasp_worker is not None and self.grasp_worker.isRunning():
            self.status_label.setText("抓取姿态正在生成，暂时不能重新采集")
            return

        self.capture_button.setEnabled(False)
        self.interaction_stage = CAPTURING_STAGE
        self.set_bci_gate(False, "capturing_realsense")
        self.current_choice = None
        self.pending_choice = None
        self.pending_option = None
        self.excluded_object_options = []
        self.update_option_styles()
        self.set_badge(self.selection_badge, "当前选择", "等待相机")
        self.set_badge(self.chain_badge, "链路", "采集中")
        self.set_badge(self.result_badge, "结果", "未运行")
        self.status_label.setText("正在采集 RealSense 并生成可选目标 ...")
        self.pose_overlay_label.clear()
        self.pose_overlay_label.setText("等待选择目标")

        self.realsense_worker = RealSensePrepareWorker(self)
        self.realsense_worker.status.connect(self.status_label.setText)
        self.realsense_worker.finished.connect(self.on_realsense_ready)
        self.realsense_worker.failed.connect(self.on_realsense_failed)
        self.realsense_worker.start()

    def on_realsense_ready(self, result: Dict[str, Any]):
        self.capture_button.setEnabled(True)
        self.frame_dir = Path(result["frame_dir"])
        self.options_json_path = Path(result["options_json"])
        self.output_root = PLATFORM_OUTPUT_DIR / f"realsense_{self.frame_dir.name}"
        self.options_result = result["options_result"]
        self.current_choice = None
        self.set_image("rgb", self.frame_dir / "color.png")
        self.pose_overlay_label.clear()
        self.pose_overlay_label.setText("请选择目标")
        self.show_object_options(self.options_result, activate_bci=True, reason="realsense_options_ready")
        self.set_badge(self.chain_badge, "链路", "RealSense 就绪")
        self.status_label.setText(f"RealSense 已采集: {self.frame_dir.name}。请选择目标 A/B/C；没有想要的目标选 D。")
        self.log(f"RealSense frame_dir={self.frame_dir}")
        self.log(f"options_json={self.options_json_path}")

    def on_realsense_failed(self, message: str):
        self.capture_button.setEnabled(True)
        self.set_bci_gate(False, "realsense_failed")
        self.set_badge(self.chain_badge, "链路", "采集失败")
        self.set_badge(self.result_badge, "结果", "未运行")
        self.status_label.setText(message)
        self.log(message)

    def select_target(self, choice: str, source: str):
        choice = choice.upper().strip()
        if choice not in self.option_buttons:
            self.status_label.setText(f"收到无效指令: {choice}")
            self.log(f"忽略无效指令: {choice}")
            return

        self.set_bci_gate(False, f"received_{choice}")
        option = self.get_option(choice)
        action = str(option.get("action", "select_target")) if option else "noop"
        self.current_choice = choice
        self.update_option_styles()
        self.log(f"{source} 指令 {choice}: action={action}, stage={self.interaction_stage}")

        if action == "noop":
            self.status_label.setText(f"{source} 选择 {choice}: 干扰/占位选项，不执行动作。")
            self.set_bci_gate(True, "noop_keep_options")
            return
        if self.interaction_stage == OBJECT_SELECTION_STAGE:
            self.handle_object_choice(choice, option, source)
        elif self.interaction_stage == CONFIRM_SELECTION_STAGE:
            self.handle_confirm_choice(action, source)
        elif self.interaction_stage == EMERGENCY_STAGE:
            self.handle_emergency_choice(action, source)
        elif self.interaction_stage == EMERGENCY_CONFIRM_STAGE:
            self.handle_emergency_confirm_choice(action, source)
        else:
            self.log(f"当前阶段 {self.interaction_stage} 不接受指令 {choice}")

    def handle_object_choice(self, choice: str, option: Optional[Dict[str, Any]], source: str) -> None:
        if not option:
            self.status_label.setText(f"未找到选项: {choice}")
            self.set_bci_gate(True, "missing_option")
            return
        action = str(option.get("action", "select_target"))
        if action == "none_of_these":
            self.excluded_object_options.extend(self.object_options_result.get("options", []))
            self.start_qwen_retry()
            return
        self.pending_choice = choice
        self.pending_option = option
        self.previous_object_options_result = self.object_options_result
        label = option.get("label", choice)
        self.log(f"选择候选目标 {choice}: {label}，进入确认页")
        self.show_confirm_options()

    def handle_confirm_choice(self, action: str, source: str) -> None:
        if action == "retry":
            self.options_result = self.previous_object_options_result or self.object_options_result
            self.show_object_options(self.options_result, activate_bci=True, reason="retry_restore_options")
            self.status_label.setText("已返回上一组目标选项，请重新选择。")
            return
        if action == "stop":
            self.stop_workflow("user_stop")
            return
        if action == "confirm":
            self.confirm_and_run()
            return
        self.set_bci_gate(True, "confirm_keep_options")

    def handle_emergency_choice(self, action: str, source: str) -> None:
        if action == "emergency_request":
            self.show_emergency_confirm_options()
            return
        self.status_label.setText("急停监控页：当前选择为干扰项，流程继续。")
        self.set_bci_gate(True, "emergency_distractor")

    def handle_emergency_confirm_choice(self, action: str, source: str) -> None:
        if action == "confirm_emergency":
            self.stop_workflow("emergency_stop", emergency=True)
            return
        if action == "cancel_emergency":
            self.show_emergency_options()
            return
        self.status_label.setText("急停确认页：当前选择为干扰项，请选择 A 确认或 B 取消。")
        self.set_bci_gate(True, "emergency_confirm_distractor")

    def start_qwen_retry(self) -> None:
        if self.options_worker is not None and self.options_worker.isRunning():
            self.log("重新识别正在进行，忽略重复请求")
            return
        self.interaction_stage = CAPTURING_STAGE
        self.set_bci_gate(False, "retry_recognition")
        self.set_badge(self.selection_badge, "当前选择", "重新识别")
        self.set_badge(self.chain_badge, "链路", "Qwen 重试")
        self.status_label.setText("正在重新识别目标，并排除上一组选项 ...")
        self.options_worker = QwenOptionsWorker(self.frame_dir, self.excluded_object_options, self)
        self.options_worker.status.connect(self.status_label.setText)
        self.options_worker.finished.connect(self.on_qwen_retry_ready)
        self.options_worker.failed.connect(self.on_qwen_retry_failed)
        self.options_worker.start()

    def on_qwen_retry_ready(self, result: Dict[str, Any]) -> None:
        self.options_json_path = Path(result["options_json"])
        self.options_result = result["options_result"]
        self.log("已排除: " + ", ".join(result.get("excluded_terms", [])[:12]))
        self.show_object_options(self.options_result, activate_bci=True, reason="retry_options_ready")

    def on_qwen_retry_failed(self, message: str) -> None:
        self.set_badge(self.chain_badge, "链路", "重试失败")
        self.status_label.setText(message)
        self.log(message)
        self.show_object_options(self.object_options_result, activate_bci=True, reason="retry_failed_restore")

    def confirm_and_run(self) -> None:
        if not self.pending_choice:
            self.status_label.setText("没有待确认目标，请重新选择。")
            self.show_object_options(self.object_options_result, activate_bci=True, reason="confirm_without_target")
            return
        missing = self.missing_formal_chain_parts()
        if missing:
            self.status_label.setText("GroundingDINO/SAM 正式链路未就绪，不能进入 GraspNet")
            self.set_badge(self.chain_badge, "链路", "未就绪")
            self.set_badge(self.result_badge, "结果", "缺依赖")
            self.log("缺少: " + "; ".join(missing))
            self.show_confirm_options()
            return
        label = self.pending_option.get("label", self.pending_choice) if self.pending_option else self.pending_choice
        self.current_choice = self.pending_choice
        self.workflow_stopped = False
        self.set_badge(self.selection_badge, "已确认", f"{self.pending_choice} {label}")
        self.set_badge(self.result_badge, "结果", "等待输出")
        self.show_emergency_options()
        self.run_formal_grasp(self.pending_choice)

    def stop_workflow(self, reason: str, emergency: bool = False) -> None:
        self.workflow_stopped = True
        self.interaction_stage = STOPPED_STAGE
        self.set_bci_gate(False, reason)
        if self.grasp_worker is not None and self.grasp_worker.isRunning():
            self.grasp_worker.cancel()
        self.pending_choice = None
        self.pending_option = None
        self.current_choice = None
        if self.object_options_result:
            self.options_result = self.object_options_result
            self.render_options()
        self.update_option_styles()
        self.set_badge(self.selection_badge, "当前选择", "已停止")
        self.set_badge(self.chain_badge, "链路", "停止")
        self.set_badge(self.result_badge, "结果", "已停止")
        label = "急停已触发" if emergency else "流程已停止"
        self.status_label.setText(f"{label}。BCI 在线识别已关闭，后续不会继续下发机械臂动作。")
        self.log(f"stop_workflow: {reason}, emergency={emergency}")

    def run_formal_grasp(self, choice: str):
        self.grasp_worker = GraspRunWorker(choice, self.frame_dir, self.options_json_path, self.output_root, self)
        self.grasp_worker.status.connect(self.status_label.setText)
        self.grasp_worker.finished.connect(self.on_grasp_finished)
        self.grasp_worker.failed.connect(self.on_grasp_failed)
        self.grasp_worker.start()

    def on_grasp_finished(self, result: Dict[str, Any]):
        if result.get("cancelled") or self.workflow_stopped:
            self.set_bci_gate(False, "grasp_cancelled")
            self.set_badge(self.chain_badge, "链路", "已停止")
            self.set_badge(self.result_badge, "结果", "已取消")
            self.status_label.setText("流程已停止，未继续下发机械臂动作。")
            self.log(f"抓取流程取消，输出目录: {result.get('output_dir')}")
            return
        self.set_bci_gate(False, "grasp_finished")
        summary = result.get("summary", {})
        top_grasps = summary.get("top_grasps", [])
        target_grasps = summary.get("target_grasps", 0)
        arm_output = result.get("arm_output", {})
        arm_execution = result.get("arm_execution", {})
        if result.get("pose_overlay"):
            self.set_image("pose", Path(result["pose_overlay"]))
        elif result.get("target_overlay"):
            self.set_image("pose", Path(result["target_overlay"]))

        if top_grasps:
            top = top_grasps[0]
            position = top.get("translation", [])
            score = float(top.get("score", 0.0) or 0.0)
            arm_status = self.describe_arm_status(arm_output, arm_execution)
            self.status_label.setText(f"抓取姿态已生成: target_grasps={target_grasps}, top_score={score:.3f}; {arm_status}")
            self.set_badge(self.chain_badge, "链路", "完成")
            result_value = "已发机械臂" if arm_execution.get("sent") else f"{target_grasps} 个候选"
            self.set_badge(self.result_badge, "结果", result_value)
            self.log(f"top grasp position={position}, width={top.get('width')}, score={score:.3f}")
            self.log(f"arm packet={arm_output.get('packet')} reachable={arm_output.get('reachable')}")
            self.log(f"arm execution={arm_execution}")
        else:
            self.status_label.setText("流程完成，但没有筛选到目标抓取姿态")
            self.set_badge(self.chain_badge, "链路", "完成")
            self.set_badge(self.result_badge, "结果", "0 个候选")
            self.log("target_grasps=0")
        self.log(f"输出目录: {result.get('output_dir')}")

    def describe_arm_status(self, arm_output: Dict[str, Any], arm_execution: Dict[str, Any]) -> str:
        packet = arm_output.get("packet") or "no packet"
        if arm_execution.get("sent"):
            return f"机械臂已发送 {packet}, success={arm_execution.get('success')}"
        if arm_execution.get("connected"):
            return f"COM7 已连接，命令不可达未发送: {packet}"
        reason = arm_execution.get("reason") or arm_output.get("reason") or "未发送"
        return f"机械臂未发送: {reason}, packet={packet}"

    def on_grasp_failed(self, message: str):
        self.set_bci_gate(False, "grasp_failed")
        self.status_label.setText("正式链路运行失败")
        self.set_badge(self.chain_badge, "链路", "失败")
        self.set_badge(self.result_badge, "结果", "错误")
        self.log(message)

    def get_option(self, choice: str) -> Optional[Dict[str, Any]]:
        for option in self.options_result.get("options", []):
            if str(option.get("key", "")).upper() == choice:
                return option
        return None

    def update_option_styles(self):
        for key, button in self.option_buttons.items():
            button.setStyleSheet(self.option_button_style(selected=key == self.current_choice))

    def missing_formal_chain_parts(self) -> List[str]:
        missing: List[str] = []
        if importlib.util.find_spec("groundingdino") is None:
            missing.append("Python 包 groundingdino")
        if importlib.util.find_spec("segment_anything") is None:
            missing.append("Python 包 segment_anything")
        required_files = [
            PROJECT_ROOT / "weights" / "groundingdino" / "GroundingDINO_SwinT_OGC.py",
            PROJECT_ROOT / "weights" / "groundingdino" / "groundingdino_swint_ogc.pth",
            PROJECT_ROOT / "weights" / "sam" / "sam_vit_b_01ec64.pth",
        ]
        for path in required_files:
            if not path.exists():
                missing.append(str(path.relative_to(PROJECT_ROOT)))
        return missing

    def log(self, text: str):
        self.log_lines.append(text)
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-200:]

    def connect_server(self):
        host = self.server_ip_lineedit.text().strip()
        port_text = self.server_port_lineedit.text().strip()

        ip_pattern = re.compile(r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(\.|$)){4}$")
        if not host:
            QMessageBox.warning(self, "输入错误", "IP 地址不能为空！")
            return
        if not ip_pattern.match(host):
            QMessageBox.warning(self, "输入错误", "请输入合法的 IP 地址！")
            return
        if not port_text:
            QMessageBox.warning(self, "输入错误", "端口号不能为空！")
            return
        if not port_text.isdigit():
            QMessageBox.warning(self, "输入错误", "端口号必须是整数！")
            return

        port = int(port_text)
        if not (0 < port < 65536):
            QMessageBox.warning(self, "输入错误", "端口号必须在 1 到 65535 之间！")
            return

        self.client_socket.connect_server(host, port)

    def disconnect_server(self):
        self.client_socket.close_server()

    def on_server_connected(self):
        self.connect_status = True
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.connect_status_label.setText("状态: 已连接")
        self.set_badge(self.chain_badge, "链路", "平台连接")
        self.publish_bci_gate(self.bci_active, "platform_connected")
        self.log("已连接平台")

    def on_server_disconnected(self):
        self.connect_status = False
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.connect_status_label.setText("状态: 已断开")
        self.set_badge(self.chain_badge, "链路", "离线预览")
        self.log("平台连接已断开")

        if self.layout_type == 1:
            self.exit_server_window()
        else:
            self.show_window_center()

    def on_server_data(self, data):
        ipc_json_data = json.loads(data.data().decode("utf-8"))
        msg = ipc_json_data["msg"]
        if msg == "ipc_algorithm_test":
            result = ipc_json_data["result_args"]["data"]
            choice = command_to_choice(result)
            self.log(f"平台指令: {result}")
            if not self.bci_active:
                self.log(f"BCI gate 已关闭，忽略平台指令: {result}")
                self.publish_bci_gate(False, "ignored_command_gate_closed")
                return
            if choice:
                self.select_target(choice, "平台")
            else:
                self.status_label.setText(f"未识别的平台指令: {result}")
        elif msg == "ipc_user_info":
            self.layout_type = ipc_json_data["layout_type"]
            if self.layout_type == 1:
                self.title_bar.setVisible(False)
                self.ipc_user_info()
        elif msg == "ipc_set_visible":
            self.setVisible(bool(ipc_json_data.get("visible", True)))
        elif msg == "ipc_exit":
            self.close()

    def exit_server_window(self):
        self.title_bar.setVisible(True)
        self.show_window_center()

    def ipc_user_info(self):
        if self.connect_status:
            data = {
                "msg": "ipc_user_info",
                "window": int(self.winId()),
            }
            json_str = json.dumps(data)
            self.client_socket.send_to_server(json_str.encode("utf-8"))

    def muted_label_style(self) -> str:
        return """
        QLabel {
            font-family: "SourceHanSansCN";
            font-weight: 400;
            font-size: 16px;
            color: #9EA0A5;
            line-height: 24px;
        }
        """

    def button_style(self) -> str:
        return """
        QPushButton:enabled {
            font-family: SourceHanSansCN, SourceHanSansCN;
            font-weight: 500;
            font-size: 17px;
            color: #20242A;
        }
        QPushButton:disabled {
            font-family: SourceHanSansCN, SourceHanSansCN;
            font-weight: 400;
            font-size: 17px;
            color: rgba(62,63,66,0.4);
        }
        QPushButton {
            background: #F8FAFC;
            border-radius: 6px;
            border: 1px solid #DDE5EF;
        }
        QPushButton:hover {
            border: 1px solid #2F80ED;
            background: #EEF6FF;
        }
        """

    def option_button_style(self, selected: bool) -> str:
        if selected:
            return """
            QPushButton {
                background: #EAF2FF;
                border: 2px solid #2F80ED;
                border-radius: 8px;
                color: #174A8B;
                font-size: 15px;
                font-weight: 600;
                text-align: left;
                padding-left: 12px;
            }
            """
        return """
        QPushButton {
            background: #FFFFFF;
            border: 1px solid #DDE5EF;
            border-radius: 8px;
            color: #20242A;
            font-size: 15px;
            font-weight: 500;
            text-align: left;
            padding-left: 12px;
        }
        QPushButton:hover {
            border: 1px solid #2F80ED;
            background: #F8FBFF;
        }
        """

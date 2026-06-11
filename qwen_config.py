from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_QWEN_MODEL = "qwen3-vl-flash"


def _candidate_roots() -> Iterable[Path]:
    roots = []
    here = Path(__file__).resolve().parent
    roots.append(here)
    roots.append(Path.cwd())
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)

    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _candidate_config_paths() -> Iterable[Path]:
    for root in _candidate_roots():
        yield root / "configs" / "local_secrets.json"
        yield root / "qwen_config.local.json"
        yield root / ".env.local"
        yield root / ".env"


def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            qwen = data.get("qwen")
            if isinstance(qwen, dict):
                merged = dict(data)
                merged.update(qwen)
                return merged
            return data
        return {}
    return _read_env_file(path)


def load_local_qwen_config() -> Dict[str, Any]:
    for path in _candidate_config_paths():
        if not path.exists():
            continue
        try:
            config = _read_config_file(path)
        except Exception:
            continue
        if config:
            return config
    return {}


def _config_value(*names: str) -> str:
    config = load_local_qwen_config()
    for name in names:
        value = config.get(name)
        if value:
            return str(value).strip()
    return ""


def resolve_qwen_api_key(env_name: str = "QWEN_API_KEY") -> str:
    key = os.getenv(env_name, "").strip()
    if key:
        return key
    return _config_value(env_name, "QWEN_API_KEY", "qwen_api_key", "api_key")


def resolve_qwen_base_url(cli_value: Optional[str] = None) -> str:
    if cli_value:
        return cli_value.strip()
    return (
        os.getenv("QWEN_BASE_URL", "").strip()
        or _config_value("QWEN_BASE_URL", "qwen_base_url", "base_url")
        or DEFAULT_QWEN_BASE_URL
    )


def resolve_qwen_model(cli_value: Optional[str] = None) -> str:
    if cli_value:
        return cli_value.strip()
    return os.getenv("QWEN_MODEL", "").strip() or _config_value("QWEN_MODEL", "qwen_model", "model") or DEFAULT_QWEN_MODEL


def ensure_qwen_environment(env_name: str = "QWEN_API_KEY") -> str:
    key = resolve_qwen_api_key(env_name)
    if key:
        os.environ.setdefault(env_name, key)
        os.environ.setdefault("QWEN_API_KEY", key)
    os.environ.setdefault("QWEN_BASE_URL", resolve_qwen_base_url())
    os.environ.setdefault("QWEN_MODEL", resolve_qwen_model())
    return key

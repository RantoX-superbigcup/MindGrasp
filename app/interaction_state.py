from __future__ import annotations

from typing import Any, Dict, List, Optional

OBJECT_SELECTION_STAGE = "object_selection"
CONFIRM_SELECTION_STAGE = "confirm_selection"
EMERGENCY_STAGE = "emergency_monitor"
EMERGENCY_CONFIRM_STAGE = "emergency_confirm"
CAPTURING_STAGE = "capturing"
RUNNING_STAGE = "running"
STOPPED_STAGE = "stopped"
MAX_OBJECT_OPTIONS = 3


def no_target_option() -> Dict[str, Any]:
    return {
        "key": "D",
        "label": "没有我想要的",
        "target_id": "none_of_these",
        "description": "重新识别，并排除当前这组目标",
        "action": "none_of_these",
        "confidence": 1.0,
    }


def noop_option(key: str, label: str = "干扰项") -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "target_id": f"noop_{key.lower()}",
        "description": "占位选项，不触发机械臂动作",
        "action": "noop",
        "enabled": True,
        "confidence": 1.0,
    }


def fixed_options(stage: str, question: str, options: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"stage": stage, "question": question, "options": options}


def confirm_options() -> Dict[str, Any]:
    return fixed_options(
        CONFIRM_SELECTION_STAGE,
        "已选中目标，请确认是否继续",
        [
            {"key": "A", "label": "重试", "target_id": "retry", "description": "返回刚才的目标列表重新选择", "action": "retry"},
            {"key": "B", "label": "停止", "target_id": "stop", "description": "停止后续流程并关闭脑机识别", "action": "stop"},
            {"key": "C", "label": "确认", "target_id": "confirm", "description": "确认目标，开始定位、位姿预测和机械臂流程", "action": "confirm"},
            noop_option("D", "干扰项"),
        ],
    )


def emergency_options() -> Dict[str, Any]:
    return fixed_options(
        EMERGENCY_STAGE,
        "运行中：如需中断请选择急停",
        [
            {"key": "A", "label": "急停", "target_id": "emergency", "description": "请求停止后续动作，需要二次确认", "action": "emergency_request"},
            noop_option("B", "干扰项 1"),
            noop_option("C", "干扰项 2"),
            noop_option("D", "干扰项 3"),
        ],
    )


def emergency_confirm_options() -> Dict[str, Any]:
    return fixed_options(
        EMERGENCY_CONFIRM_STAGE,
        "请确认是否执行急停",
        [
            {"key": "A", "label": "确认急停", "target_id": "confirm_emergency", "description": "立即停止后续流程", "action": "confirm_emergency"},
            {"key": "B", "label": "取消", "target_id": "cancel_emergency", "description": "返回急停监控页", "action": "cancel_emergency"},
            noop_option("C", "干扰项 1"),
            noop_option("D", "干扰项 2"),
        ],
    )


def option_terms(option: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("label", "target_id", "description"):
        value = str(option.get(key, "")).strip().lower().replace("_", " ")
        if value:
            values.append(value)
    prompts = option.get("grounding_prompts") or []
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        value = str(prompt).strip().lower().replace("_", " ")
        if value:
            values.append(value)
    return values


def collect_exclusion_terms(options: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    terms: List[str] = []
    for option in options:
        action = str(option.get("action", "select_target"))
        if action in {"none_of_these", "noop"}:
            continue
        for term in option_terms(option):
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def option_matches_exclusion(option: Dict[str, Any], excluded_terms: List[str]) -> bool:
    if not excluded_terms:
        return False
    text = " ".join(option_terms(option))
    for term in excluded_terms:
        term = term.strip().lower().replace("_", " ")
        if len(term) >= 3 and (term in text or text in term):
            return True
    return False


def build_platform_object_options(options_result: Dict[str, Any], excluded_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    excluded_terms = excluded_terms or []
    object_options: List[Dict[str, Any]] = []
    for option in options_result.get("options", []):
        action = str(option.get("action", "select_target"))
        if action in {"none_of_these", "noop"}:
            continue
        if option_matches_exclusion(option, excluded_terms):
            continue
        item = dict(option)
        item["key"] = chr(ord("A") + len(object_options))
        item["action"] = "select_target"
        item.setdefault("enabled", True)
        object_options.append(item)
        if len(object_options) >= MAX_OBJECT_OPTIONS:
            break

    while len(object_options) < MAX_OBJECT_OPTIONS:
        key = chr(ord("A") + len(object_options))
        filler = noop_option(key, "空白占位")
        filler["description"] = "当前画面没有更多可选目标"
        object_options.append(filler)

    return {
        **options_result,
        "stage": OBJECT_SELECTION_STAGE,
        "question": "请选择想抓取的目标；若没有请选 D",
        "options": object_options + [no_target_option()],
    }

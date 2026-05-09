import json
import re
from pathlib import Path
from typing import Any

from common import BAD_LABEL, GOOD_LABEL, Prediction, TraceItem, task_signature


DEFAULT_RULE_CONFIG = {
    "version": "fallback",
    "default_label_when_no_bad_rule_matches": GOOD_LABEL,
    "final_result_fields": ["final_result", "answer", "response"],
    "success_task_keywords": [],
    "failure_markers": [
        "error",
        "failed",
        "exception",
        "traceback",
        "null",
        "none",
        "失败",
        "报错",
        "异常",
    ],
    "badcase_rules": {
        "plan_list_required": True,
        "empty_plan_without_final_result": True,
        "all_step_results_empty": True,
        "failed_step_continue_with_new_command": True,
        "repeated_or_loop_tasks": {
            "enabled": True,
            "repeat_threshold": 3,
            "include_command_args": True,
        },
    },
}


def load_rule_config(path: Path | None = None) -> dict[str, Any]:
    rules_path = path or Path(__file__).resolve().parents[1] / "rules.md"
    if not rules_path.is_file():
        return DEFAULT_RULE_CONFIG

    content = rules_path.read_text(encoding="utf-8-sig")
    match = re.search(
        r"<!-- RULE_CONFIG_START -->\s*```json\s*(.*?)\s*```\s*<!-- RULE_CONFIG_END -->",
        content,
        flags=re.DOTALL,
    )
    if not match:
        return DEFAULT_RULE_CONFIG

    config = json.loads(match.group(1))
    return merge_defaults(DEFAULT_RULE_CONFIG, config)


def merge_defaults(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def is_normal_result(result: Any, failure_markers: list[str]) -> bool:
    if result is None:
        return False
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        return not any(marker.lower() in lowered for marker in failure_markers)
    if isinstance(result, (list, dict)):
        return len(result) > 0
    return True


def is_failure_result(result: Any, failure_markers: list[str]) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return True
        lowered = stripped.lower()
        return any(marker.lower() in lowered for marker in failure_markers)
    if isinstance(result, (list, dict)):
        return len(result) == 0
    return False


def has_command(task: Any) -> bool:
    if not isinstance(task, dict):
        return False
    command = task.get("command")
    if not isinstance(command, dict):
        return False
    return bool(str(command.get("name") or "").strip())


def has_failed_step_continue_with_new_command(plan_list: list[Any], failure_markers: list[str]) -> bool:
    for index, task in enumerate(plan_list):
        if not isinstance(task, dict):
            continue
        if not is_failure_result(task.get("result"), failure_markers):
            continue
        if any(has_command(next_task) for next_task in plan_list[index + 1 :]):
            return True
    return False


def has_final_result(trace: dict[str, Any], final_result_fields: list[str]) -> bool:
    return any(is_nonempty_value(trace.get(field)) for field in final_result_fields)


def has_repeated_or_loop_tasks(
    plan_list: list[Any],
    repeat_threshold: int,
    include_command_args: bool,
) -> bool:
    if repeat_threshold <= 1:
        return bool(plan_list)
    signatures = [
        task_signature(task, include_args=include_command_args)
        for task in plan_list
        if isinstance(task, dict)
    ]
    total = len(signatures)
    if total < repeat_threshold:
        return False
    for window_size in range(1, total // repeat_threshold + 1):
        span = window_size * repeat_threshold
        for start in range(0, total - span + 1):
            pattern = signatures[start : start + window_size]
            if all(
                signatures[start + offset * window_size : start + (offset + 1) * window_size] == pattern
                for offset in range(1, repeat_threshold)
            ):
                return True
    return False


def classify_trace(
    trace: dict[str, Any],
    config: dict[str, Any],
    repeat_threshold_override: int | None = None,
) -> tuple[str, str]:
    badcase_rules = config.get("badcase_rules") or {}
    failure_markers = list(config.get("failure_markers") or [])
    final_result_fields = list(config.get("final_result_fields") or [])
    success_task_keywords = list(config.get("success_task_keywords") or [])

    plan_list = trace.get("plan_list")
    if badcase_rules.get("plan_list_required", True) and not isinstance(plan_list, list):
        return BAD_LABEL, "规则法：plan_list 缺失或不是 list"
    if not isinstance(plan_list, list):
        plan_list = []

    if (
        badcase_rules.get("empty_plan_without_final_result", True)
        and not plan_list
        and not has_final_result(trace, final_result_fields)
    ):
        return BAD_LABEL, "规则法：plan_list 为空且没有最终结果"

    task_results = [
        task.get("result")
        for task in plan_list
        if isinstance(task, dict)
    ]
    if (
        badcase_rules.get("all_step_results_empty", True)
        and task_results
        and not any(is_nonempty_value(result) for result in task_results)
    ):
        return BAD_LABEL, "规则法：所有步骤 result 均为空"

    if (
        badcase_rules.get("failed_step_continue_with_new_command", True)
        and has_failed_step_continue_with_new_command(plan_list, failure_markers)
    ):
        return BAD_LABEL, "规则法：工具失败或空结果后继续执行新 command"

    repeat_rule = badcase_rules.get("repeated_or_loop_tasks") or {}
    if repeat_rule.get("enabled", True):
        repeat_threshold = repeat_threshold_override or int(repeat_rule.get("repeat_threshold", 3))
        include_command_args = bool(repeat_rule.get("include_command_args", True))
        if has_repeated_or_loop_tasks(plan_list, repeat_threshold, include_command_args):
            return BAD_LABEL, f"规则法：命中重复/循环任务阈值 {repeat_threshold}"

    if success_task_keywords:
        for task in plan_list:
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("task_name", ""))
            if any(keyword in task_name for keyword in success_task_keywords) and is_normal_result(
                task.get("result"),
                failure_markers,
            ):
                return GOOD_LABEL, "规则法：命中配置的成功任务关键词且结果有效"

    default_label = str(config.get("default_label_when_no_bad_rule_matches") or GOOD_LABEL)
    if default_label not in {GOOD_LABEL, BAD_LABEL}:
        default_label = GOOD_LABEL
    return default_label, "规则法：未命中高置信 badcase 规则"


def classify(items: list[TraceItem], repeat_threshold: int | None = None) -> list[Prediction]:
    config = load_rule_config()
    predictions: list[Prediction] = []
    for item in items:
        predicted_label, reason = classify_trace(item.trace, config, repeat_threshold)
        predictions.append(
            Prediction(
                name=item.meta.name,
                source=item.meta.source,
                split=item.meta.split or "unknown",
                actual_label=item.meta.label or "unknown",
                predicted_label=predicted_label,
                detail={"reason": reason},
            )
        )
    return predictions

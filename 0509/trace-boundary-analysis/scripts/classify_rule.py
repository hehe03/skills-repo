import json
import re
from pathlib import Path
from typing import Any

from common import BAD_LABEL, GOOD_LABEL, Prediction, TraceItem, task_signature


RULE_LAYER_ORDER = ["general", "trace_format", "domain_prior"]
RULE_LAYER_ALIASES = {
    "1": "general",
    "general": "general",
    "generic": "general",
    "2": "trace_format",
    "trace": "trace_format",
    "trace_format": "trace_format",
    "format": "trace_format",
    "3": "domain_prior",
    "domain": "domain_prior",
    "domain_prior": "domain_prior",
    "prior": "domain_prior",
    "all": "domain_prior",
}


DEFAULT_RULE_CONFIG = {
    "version": "rule-v1-layered",
    "default_label_when_no_bad_rule_matches": GOOD_LABEL,
    "rule_layers": {
        "general": {
            "description": "No trace-specific prior knowledge; only generic JSON/result-quality checks.",
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
                "root_object_required": True,
                "non_empty_root_required": True,
                "all_leaf_values_empty": True,
            },
        },
        "trace_format": {
            "description": "Knows the basic trace schema: query, plan_list, task_name, command, result.",
            "final_result_fields": ["final_result", "answer", "response"],
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
        },
        "domain_prior": {
            "description": "Knows domain-specific success evidence such as task-name keywords.",
            "success_task_keywords": [],
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
    return normalize_config(merge_defaults(DEFAULT_RULE_CONFIG, config))


def merge_defaults(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Accept both the layered config and the older flat rules.md config."""
    layers = config.setdefault("rule_layers", {})
    general = layers.setdefault("general", {})
    trace_format = layers.setdefault("trace_format", {})
    domain_prior = layers.setdefault("domain_prior", {})

    if "failure_markers" in config:
        general["failure_markers"] = config["failure_markers"]
    if "final_result_fields" in config:
        trace_format["final_result_fields"] = config["final_result_fields"]
    if "badcase_rules" in config:
        trace_format["badcase_rules"] = merge_defaults(
            trace_format.get("badcase_rules") or {},
            config["badcase_rules"],
        )
    if "success_task_keywords" in config:
        domain_prior["success_task_keywords"] = config["success_task_keywords"]
    return config


def normalize_rule_layer(rule_layer: str | None) -> str:
    key = (rule_layer or "domain_prior").strip().lower().replace("-", "_")
    if key not in RULE_LAYER_ALIASES:
        raise ValueError(
            f"Unsupported rule layer: {rule_layer}. "
            f"Expected one of: {', '.join(RULE_LAYER_ORDER)}"
        )
    return RULE_LAYER_ALIASES[key]


def enabled_layers(rule_layer: str | None) -> list[str]:
    normalized = normalize_rule_layer(rule_layer)
    return RULE_LAYER_ORDER[: RULE_LAYER_ORDER.index(normalized) + 1]


def is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def iter_leaf_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_leaf_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_leaf_values(child)
    else:
        yield value


def all_leaf_values_empty(trace: dict[str, Any]) -> bool:
    leaves = list(iter_leaf_values(trace))
    return bool(leaves) and not any(is_nonempty_value(value) for value in leaves)


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


def classify_with_general_rules(trace: dict[str, Any], layer_config: dict[str, Any]) -> tuple[str, str] | None:
    badcase_rules = layer_config.get("badcase_rules") or {}
    if badcase_rules.get("root_object_required", True) and not isinstance(trace, dict):
        return BAD_LABEL, "rule[general]: JSON root is not an object"
    if not isinstance(trace, dict):
        return None
    if badcase_rules.get("non_empty_root_required", True) and not trace:
        return BAD_LABEL, "rule[general]: JSON root object is empty"
    if badcase_rules.get("all_leaf_values_empty", True) and all_leaf_values_empty(trace):
        return BAD_LABEL, "rule[general]: all JSON leaf values are empty"
    return None


def classify_with_trace_format_rules(
    trace: dict[str, Any],
    layer_config: dict[str, Any],
    failure_markers: list[str],
    repeat_threshold_override: int | None,
) -> tuple[str, str] | None:
    badcase_rules = layer_config.get("badcase_rules") or {}
    final_result_fields = list(layer_config.get("final_result_fields") or [])

    plan_list = trace.get("plan_list")
    if badcase_rules.get("plan_list_required", True) and not isinstance(plan_list, list):
        return BAD_LABEL, "rule[trace_format]: plan_list is missing or is not a list"
    if not isinstance(plan_list, list):
        plan_list = []

    if (
        badcase_rules.get("empty_plan_without_final_result", True)
        and not plan_list
        and not has_final_result(trace, final_result_fields)
    ):
        return BAD_LABEL, "rule[trace_format]: plan_list is empty and no final result field is populated"

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
        return BAD_LABEL, "rule[trace_format]: all step result fields are empty"

    if (
        badcase_rules.get("failed_step_continue_with_new_command", True)
        and has_failed_step_continue_with_new_command(plan_list, failure_markers)
    ):
        return BAD_LABEL, "rule[trace_format]: a failed/empty step is followed by another command"

    repeat_rule = badcase_rules.get("repeated_or_loop_tasks") or {}
    if repeat_rule.get("enabled", True):
        repeat_threshold = repeat_threshold_override or int(repeat_rule.get("repeat_threshold", 3))
        include_command_args = bool(repeat_rule.get("include_command_args", True))
        if has_repeated_or_loop_tasks(plan_list, repeat_threshold, include_command_args):
            return BAD_LABEL, f"rule[trace_format]: repeated/looped task pattern reaches threshold {repeat_threshold}"

    return None


def classify_with_domain_prior_rules(
    trace: dict[str, Any],
    layer_config: dict[str, Any],
    failure_markers: list[str],
) -> tuple[str, str] | None:
    success_task_keywords = list(layer_config.get("success_task_keywords") or [])
    if not success_task_keywords:
        return None

    plan_list = trace.get("plan_list")
    if not isinstance(plan_list, list):
        return None

    for task in plan_list:
        if not isinstance(task, dict):
            continue
        task_name = str(task.get("task_name", ""))
        if any(keyword in task_name for keyword in success_task_keywords) and is_normal_result(
            task.get("result"),
            failure_markers,
        ):
            return GOOD_LABEL, "rule[domain_prior]: success task keyword matched with a valid result"
    return None


def classify_trace(
    trace: dict[str, Any],
    config: dict[str, Any],
    repeat_threshold_override: int | None = None,
    rule_layer: str = "domain_prior",
) -> tuple[str, str]:
    active_layers = enabled_layers(rule_layer)
    layers = config.get("rule_layers") or {}
    general_config = layers.get("general") or {}
    trace_format_config = layers.get("trace_format") or {}
    domain_prior_config = layers.get("domain_prior") or {}
    failure_markers = list(general_config.get("failure_markers") or [])

    if "general" in active_layers:
        result = classify_with_general_rules(trace, general_config)
        if result is not None:
            return result

    if "trace_format" in active_layers:
        result = classify_with_trace_format_rules(
            trace,
            trace_format_config,
            failure_markers,
            repeat_threshold_override,
        )
        if result is not None:
            return result

    if "domain_prior" in active_layers:
        result = classify_with_domain_prior_rules(trace, domain_prior_config, failure_markers)
        if result is not None:
            return result

    default_label = str(config.get("default_label_when_no_bad_rule_matches") or GOOD_LABEL)
    if default_label not in {GOOD_LABEL, BAD_LABEL}:
        default_label = GOOD_LABEL
    return default_label, f"rule[{normalize_rule_layer(rule_layer)}]: no high-confidence badcase rule matched"


def classify(
    items: list[TraceItem],
    repeat_threshold: int | None = None,
    rule_layer: str = "domain_prior",
    config_override: dict[str, Any] | None = None,
) -> list[Prediction]:
    config = (
        normalize_config(merge_defaults(DEFAULT_RULE_CONFIG, config_override))
        if config_override is not None
        else load_rule_config()
    )
    normalized_layer = normalize_rule_layer(rule_layer)
    predictions: list[Prediction] = []
    for item in items:
        predicted_label, reason = classify_trace(item.trace, config, repeat_threshold, normalized_layer)
        predictions.append(
            Prediction(
                name=item.meta.name,
                source=item.meta.source,
                split=item.meta.split or "unknown",
                actual_label=item.meta.label or "unknown",
                predicted_label=predicted_label,
                detail={"reason": reason, "rule_layer": normalized_layer},
            )
        )
    return predictions

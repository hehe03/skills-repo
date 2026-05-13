from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from trace_io import TraceRecord


ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|timeout|timed out|permission denied|"
    r"not found|invalid|abort|cancelled|unauthorized|forbidden)\b",
    re.IGNORECASE,
)
FINAL_KEYS = {
    "final",
    "final_answer",
    "final_response",
    "answer",
    "response",
    "output",
    "result",
}
ACTION_KEYS = {"tool", "tool_name", "name", "action", "operation", "task_name"}
RESULT_KEYS = {"result", "output", "observation", "content", "response", "final_answer"}


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _stringify(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:limit]
    except TypeError:
        return str(value)[:limit]


def _all_strings(trace: Any) -> List[str]:
    strings: List[str] = []
    for node in _walk(trace):
        if isinstance(node, str):
            strings.append(node)
    return strings


def _candidate_steps(trace: Any) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []

    def add_from_list(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    steps.append(item)

    if isinstance(trace, dict):
        for key in ("plan_list", "steps", "events", "messages", "turns", "spans", "actions"):
            add_from_list(trace.get(key))

    if not steps:
        for node in _walk(trace):
            if not isinstance(node, dict):
                continue
            keys = set(node.keys())
            if keys & (ACTION_KEYS | RESULT_KEYS | {"role", "command", "function_call"}):
                steps.append(node)
    return steps


def _action_name(step: Dict[str, Any]) -> str:
    command = step.get("command")
    if isinstance(command, dict):
        for key in ("name", "tool", "tool_name", "action"):
            if command.get(key):
                return str(command[key])
    function_call = step.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return str(function_call["name"])
    for key in ACTION_KEYS:
        if step.get(key):
            return str(step[key])
    if step.get("role"):
        return str(step["role"])
    return "unknown"


def _result_text(step: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in RESULT_KEYS:
        if key in step:
            chunks.append(_stringify(step[key]))
    return "\n".join(chunk for chunk in chunks if chunk)


def _has_final_answer(trace: Any, strings: List[str]) -> Tuple[bool, int]:
    if isinstance(trace, dict):
        for key, value in trace.items():
            if key in FINAL_KEYS and _stringify(value).strip():
                return True, len(_stringify(value).strip())
    final_like = []
    for node in _walk(trace):
        if not isinstance(node, dict):
            continue
        role = str(node.get("role", "")).lower()
        if role == "assistant" and _stringify(node.get("content")).strip():
            final_like.append(_stringify(node.get("content")).strip())
        for key in ("final_answer", "final_response", "answer"):
            if _stringify(node.get(key)).strip():
                final_like.append(_stringify(node.get(key)).strip())
    if final_like:
        text = final_like[-1]
        return True, len(text)
    long_strings = [s.strip() for s in strings if len(s.strip()) >= 80]
    if long_strings:
        return True, len(long_strings[-1])
    return False, 0


def _max_consecutive(values: List[str]) -> int:
    if not values:
        return 0
    best = 1
    current = 1
    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def extract_features(record: TraceRecord) -> Dict[str, Any]:
    trace = record.trace
    strings = _all_strings(trace)
    full_text = "\n".join(strings)
    steps = _candidate_steps(trace)
    actions = [_action_name(step).strip().lower() for step in steps]
    result_texts = [_result_text(step).strip() for step in steps]
    nonempty_results = [text for text in result_texts if text]
    action_counts = Counter(actions)
    has_final_answer, final_answer_chars = _has_final_answer(trace, strings)
    error_count = len(ERROR_RE.findall(full_text))
    parse_error = bool(record.parse_error or (isinstance(trace, dict) and trace.get("_parse_error")))
    step_count = len(steps)
    empty_result_count = max(0, step_count - len(nonempty_results))
    empty_result_ratio = empty_result_count / step_count if step_count else 1.0
    repeated_action_count = sum(count - 1 for count in action_counts.values() if count > 1)
    max_consecutive_same_action = _max_consecutive(actions)
    unique_action_ratio = len(action_counts) / step_count if step_count else 0.0

    return {
        "parse_error": parse_error,
        "is_empty_trace": not bool(trace) or trace == [] or trace == {},
        "has_steps": step_count > 0,
        "step_count": step_count,
        "action_count": step_count,
        "unique_action_count": len(action_counts),
        "unique_action_ratio": round(unique_action_ratio, 4),
        "repeated_action_count": repeated_action_count,
        "max_consecutive_same_action": max_consecutive_same_action,
        "error_count": error_count,
        "has_error_text": error_count > 0 or parse_error,
        "empty_result_count": empty_result_count,
        "empty_result_ratio": round(empty_result_ratio, 4),
        "nonempty_result_ratio": round(1.0 - empty_result_ratio, 4) if step_count else 0.0,
        "has_final_answer": has_final_answer,
        "final_answer_chars": final_answer_chars,
        "text_chars": len(full_text),
        "source": record.source or "",
        "split": record.split or "",
    }

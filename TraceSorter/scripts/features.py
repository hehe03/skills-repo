from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from trace_io import TraceRecord


ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|timeout|timed out|permission denied|"
    r"not found|invalid|abort|cancelled|unauthorized|forbidden)\b",
    re.IGNORECASE,
)
DEFAULT_TOP_LEVEL_FINAL_KEYS = {
    "final",
    "final_answer",
    "final_response",
    "answer",
    "response",
    "output",
    "result",
}
DEFAULT_NESTED_FINAL_KEYS = {
    "final_answer",
    "final_response",
    "answer",
}
DEFAULT_ASSISTANT_ROLES = {"assistant"}
DEFAULT_ASSISTANT_CONTENT_KEYS = {"content"}
ACTION_KEYS = {"tool", "tool_name", "name", "action", "operation", "task_name"}
RESULT_KEYS = {"result", "output", "observation", "content", "response", "final_answer"}


@dataclass(frozen=True)
class FinalAnswerConfig:
    top_level_keys: frozenset[str]
    nested_keys: frozenset[str]
    assistant_roles: frozenset[str]
    assistant_content_keys: frozenset[str]
    min_chars: int = 1
    evidence_enabled: bool = False
    evidence_strength: str = "none"
    evidence_source: str = "none"
    adopted_fields: tuple[str, ...] = ()


def default_final_answer_config() -> FinalAnswerConfig:
    return FinalAnswerConfig(
        top_level_keys=frozenset(DEFAULT_TOP_LEVEL_FINAL_KEYS),
        nested_keys=frozenset(DEFAULT_NESTED_FINAL_KEYS),
        assistant_roles=frozenset(DEFAULT_ASSISTANT_ROLES),
        assistant_content_keys=frozenset(DEFAULT_ASSISTANT_CONTENT_KEYS),
        min_chars=1,
        evidence_enabled=False,
        evidence_strength="none",
        evidence_source="none",
        adopted_fields=(),
    )


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def load_final_answer_config(
    config_path: str | Path | None = None,
    extra_keys: str | List[str] | None = None,
) -> FinalAnswerConfig:
    config = default_final_answer_config()
    data: Dict[str, Any] = {}
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

    top_level_keys = set(config.top_level_keys)
    nested_keys = set(config.nested_keys)
    assistant_roles = set(config.assistant_roles)
    assistant_content_keys = set(config.assistant_content_keys)

    if "top_level_keys" in data:
        top_level_keys = _as_string_set(data.get("top_level_keys"))
    if "nested_keys" in data:
        nested_keys = _as_string_set(data.get("nested_keys"))
    if "assistant_roles" in data:
        assistant_roles = {item.lower() for item in _as_string_set(data.get("assistant_roles"))}
    if "assistant_content_keys" in data:
        assistant_content_keys = _as_string_set(data.get("assistant_content_keys"))

    simple_extra_keys = _as_string_set(extra_keys)
    if simple_extra_keys and not config_path:
        top_level_keys = set(simple_extra_keys)
        nested_keys = set(simple_extra_keys)
        assistant_content_keys = set()
    top_level_keys.update(simple_extra_keys)
    nested_keys.update(simple_extra_keys)
    config_source = str(data.get("evidence_source") or "").strip().lower()
    has_user_override = bool(config_path or simple_extra_keys)
    evidence_source = config_source or ("user" if has_user_override else "none")
    evidence_strength = "strong" if evidence_source == "user" else "medium" if evidence_source == "llm" else "none"
    evidence_enabled = has_user_override or evidence_source in {"user", "llm"}
    if "evidence_enabled" in data:
        evidence_enabled = bool(data["evidence_enabled"])
    if "evidence_strength" in data:
        evidence_strength = str(data["evidence_strength"]).strip().lower()

    return FinalAnswerConfig(
        top_level_keys=frozenset(top_level_keys),
        nested_keys=frozenset(nested_keys),
        assistant_roles=frozenset(assistant_roles),
        assistant_content_keys=frozenset(assistant_content_keys),
        min_chars=max(1, int(data.get("min_chars", config.min_chars))),
        evidence_enabled=evidence_enabled,
        evidence_strength=evidence_strength if evidence_enabled else "none",
        evidence_source=evidence_source if evidence_enabled else "none",
        adopted_fields=tuple(sorted(simple_extra_keys)) if simple_extra_keys else tuple(data.get("adopted_fields", ())),
    )


def _replace_final_answer_config(
    config: FinalAnswerConfig,
    *,
    top_level_keys: Iterable[str] | None = None,
    nested_keys: Iterable[str] | None = None,
    assistant_content_keys: Iterable[str] | None = None,
    evidence_enabled: bool,
    evidence_strength: str,
    evidence_source: str,
    adopted_fields: Iterable[str],
) -> FinalAnswerConfig:
    return FinalAnswerConfig(
        top_level_keys=frozenset(top_level_keys if top_level_keys is not None else config.top_level_keys),
        nested_keys=frozenset(nested_keys if nested_keys is not None else config.nested_keys),
        assistant_roles=config.assistant_roles,
        assistant_content_keys=frozenset(
            assistant_content_keys if assistant_content_keys is not None else config.assistant_content_keys
        ),
        min_chars=config.min_chars,
        evidence_enabled=evidence_enabled,
        evidence_strength=evidence_strength,
        evidence_source=evidence_source,
        adopted_fields=tuple(sorted(set(adopted_fields))),
    )


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _walk_descendants(value: Any) -> Iterable[Any]:
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


def _valid_final_text(value: Any, config: FinalAnswerConfig) -> str:
    text = _stringify(value).strip()
    return text if len(text) >= config.min_chars else ""


def _has_final_answer(trace: Any, config: FinalAnswerConfig) -> Tuple[bool, int, str]:
    if not config.evidence_enabled:
        return False, 0, ""
    if isinstance(trace, dict):
        for key, value in trace.items():
            if key in config.top_level_keys:
                text = _valid_final_text(value, config)
                if text:
                    return True, len(text), f"top_level:{key}"
    final_like = []
    for node in _walk_descendants(trace):
        if not isinstance(node, dict):
            continue
        role = str(node.get("role", "")).lower()
        if role in config.assistant_roles:
            for key in config.assistant_content_keys:
                text = _valid_final_text(node.get(key), config)
                if text:
                    final_like.append((text, f"assistant:{key}"))
        for key in config.nested_keys:
            text = _valid_final_text(node.get(key), config)
            if text:
                final_like.append((text, f"nested:{key}"))
    if final_like:
        text, source = final_like[-1]
        return True, len(text), source
    return False, 0, ""


def discover_default_final_answer_config(
    records: Iterable[TraceRecord],
    config: FinalAnswerConfig,
) -> FinalAnswerConfig:
    if config.evidence_enabled:
        return config
    top_hits: set[str] = set()
    nested_hits: set[str] = set()
    assistant_hits: set[str] = set()
    for record in records:
        trace = record.trace
        if isinstance(trace, dict):
            for key, value in trace.items():
                if key in config.top_level_keys and _valid_final_text(value, config):
                    top_hits.add(key)
        for node in _walk_descendants(trace):
            if not isinstance(node, dict):
                continue
            role = str(node.get("role", "")).lower()
            if role in config.assistant_roles:
                for key in config.assistant_content_keys:
                    if _valid_final_text(node.get(key), config):
                        assistant_hits.add(f"assistant:{key}")
            for key in config.nested_keys:
                if _valid_final_text(node.get(key), config):
                    nested_hits.add(key)
    adopted_fields = [f"top_level:{key}" for key in top_hits]
    adopted_fields.extend(f"nested:{key}" for key in nested_hits)
    adopted_fields.extend(sorted(assistant_hits))
    if not adopted_fields:
        return config
    return _replace_final_answer_config(
        config,
        top_level_keys=top_hits,
        nested_keys=nested_hits,
        assistant_content_keys={item.split(":", 1)[1] for item in assistant_hits},
        evidence_enabled=True,
        evidence_strength="medium",
        evidence_source="default",
        adopted_fields=adopted_fields,
    )


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


def extract_features(
    record: TraceRecord,
    final_answer_config: FinalAnswerConfig | None = None,
) -> Dict[str, Any]:
    trace = record.trace
    final_answer_config = final_answer_config or default_final_answer_config()
    strings = _all_strings(trace)
    full_text = "\n".join(strings)
    steps = _candidate_steps(trace)
    actions = [_action_name(step).strip().lower() for step in steps]
    result_texts = [_result_text(step).strip() for step in steps]
    nonempty_results = [text for text in result_texts if text]
    action_counts = Counter(actions)
    has_final_answer, final_answer_chars, final_answer_source = _has_final_answer(
        trace,
        final_answer_config,
    )
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
        "final_answer_evidence_enabled": final_answer_config.evidence_enabled,
        "final_answer_evidence_strength": final_answer_config.evidence_strength,
        "final_answer_evidence_source": final_answer_config.evidence_source,
        "final_answer_adopted_fields": ",".join(final_answer_config.adopted_fields),
        "final_answer_chars": final_answer_chars,
        "final_answer_source": final_answer_source,
        "text_chars": len(full_text),
        "source": record.source or "",
        "split": record.split or "",
    }

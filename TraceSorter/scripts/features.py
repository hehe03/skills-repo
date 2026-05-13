from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
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
MAX_DYNAMIC_FIELD_TEXT = 4000


@dataclass(frozen=True)
class FinalAnswerItem:
    key_pattern: str
    value_pattern: str
    raw: str


@dataclass(frozen=True)
class FinalAnswerConfig:
    top_level_keys: frozenset[str]
    nested_keys: frozenset[str]
    assistant_roles: frozenset[str]
    assistant_content_keys: frozenset[str]
    item_patterns: tuple[FinalAnswerItem, ...] = ()
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
        item_patterns=(),
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


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _parse_final_answer_item(raw: str) -> FinalAnswerItem:
    text = raw.strip()
    if ":" not in text:
        raise ValueError(f"final-answer item must use key:value format: {raw}")
    key, value = text.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"final-answer item has empty key: {raw}")
    if not value:
        value = "*"
    return FinalAnswerItem(key_pattern=key, value_pattern=value, raw=f"{key}:{value}")


def _parse_final_answer_items(value: Any) -> tuple[FinalAnswerItem, ...]:
    items: List[FinalAnswerItem] = []
    for raw in _as_string_list(value):
        chunks = [raw]
        if "\n" in raw or ";" in raw:
            chunks = re.split(r"[;\n]+", raw)
        for chunk in chunks:
            if chunk.strip():
                items.append(_parse_final_answer_item(chunk))
    return tuple(items)


def _wildcard_fullmatch(pattern: str, value: Any) -> bool:
    text = _stringify(value).strip()
    regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
    return re.fullmatch(regex, text, flags=re.IGNORECASE | re.DOTALL) is not None


def load_final_answer_config(
    config_path: str | Path | None = None,
    final_answer_items: str | List[str] | None = None,
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

    item_patterns = _parse_final_answer_items(final_answer_items)
    if "final_answer_items" in data:
        item_patterns = item_patterns + _parse_final_answer_items(data.get("final_answer_items"))
    config_source = str(data.get("evidence_source") or "").strip().lower()
    has_user_override = bool(config_path or item_patterns)
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
        item_patterns=item_patterns,
        min_chars=max(1, int(data.get("min_chars", config.min_chars))),
        evidence_enabled=evidence_enabled,
        evidence_strength=evidence_strength if evidence_enabled else "none",
        evidence_source=evidence_source if evidence_enabled else "none",
        adopted_fields=tuple(item.raw for item in item_patterns) if item_patterns else tuple(data.get("adopted_fields", ())),
    )


def _replace_final_answer_config(
    config: FinalAnswerConfig,
    *,
    top_level_keys: Iterable[str] | None = None,
    nested_keys: Iterable[str] | None = None,
    assistant_content_keys: Iterable[str] | None = None,
    item_patterns: Iterable[FinalAnswerItem] | None = None,
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
        item_patterns=tuple(item_patterns if item_patterns is not None else config.item_patterns),
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


def _field_path_segment(key: Any) -> str:
    text = str(key).strip()
    text = re.sub(r"\s+", "_", text)
    return text or "empty_key"


def _flatten_leaf_fields(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value and path:
            yield path, ""
        for key, child in value.items():
            segment = _field_path_segment(key)
            next_path = f"{path}.{segment}" if path else segment
            yield from _flatten_leaf_fields(child, next_path)
        return
    if isinstance(value, list):
        if not value and path:
            yield path, ""
        next_path = f"{path}[]" if path else "[]"
        for child in value:
            yield from _flatten_leaf_fields(child, next_path)
        return
    if path:
        yield path, value


def _dynamic_field_features(trace: Any) -> Dict[str, Any]:
    values_by_path: Dict[str, List[Any]] = defaultdict(list)
    for path, value in _flatten_leaf_fields(trace):
        values_by_path[path].append(value)

    features: Dict[str, Any] = {
        "trace_field_paths": "\n".join(sorted(values_by_path)),
    }
    for path, values in values_by_path.items():
        texts = [_stringify(value, limit=500).strip() for value in values]
        nonempty_texts = [text for text in texts if text]
        features[f"field_exists:{path}"] = True
        features[f"field_count:{path}"] = len(values)
        features[f"field_text:{path}"] = "\n".join(nonempty_texts)[:MAX_DYNAMIC_FIELD_TEXT]
        features[f"field_nonempty_ratio:{path}"] = round(len(nonempty_texts) / len(values), 4) if values else 0.0

        numbers: List[float] = []
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numbers.append(float(value))
            else:
                try:
                    numbers.append(float(str(value)))
                except (TypeError, ValueError):
                    pass
        if numbers and len(numbers) == len(values):
            features[f"field_number_min:{path}"] = min(numbers)
            features[f"field_number_max:{path}"] = max(numbers)
            features[f"field_number_mean:{path}"] = round(sum(numbers) / len(numbers), 4)
            if len(numbers) == 1:
                features[f"field_number:{path}"] = numbers[0]

        bools = [value for value in values if isinstance(value, bool)]
        if bools and len(bools) == len(values):
            features[f"field_bool_true_ratio:{path}"] = round(sum(1 for value in bools if value) / len(bools), 4)

    return features


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


def _match_final_answer_items(node: Dict[str, Any], config: FinalAnswerConfig) -> tuple[str, str, str] | None:
    for key, value in node.items():
        for item in config.item_patterns:
            if not _wildcard_fullmatch(item.key_pattern, key):
                continue
            if not _wildcard_fullmatch(item.value_pattern, value):
                continue
            text = _valid_final_text(value, config)
            if text:
                return text, key, item.raw
    return None


def _has_final_answer(trace: Any, config: FinalAnswerConfig) -> Tuple[bool, int, str]:
    if not config.evidence_enabled:
        return False, 0, ""
    if isinstance(trace, dict):
        item_match = _match_final_answer_items(trace, config)
        if item_match:
            text, key, raw = item_match
            return True, len(text), f"top_level_item:{key}~{raw}"
        for key, value in trace.items():
            if key in config.top_level_keys:
                text = _valid_final_text(value, config)
                if text:
                    return True, len(text), f"top_level:{key}"
    final_like = []
    for node in _walk_descendants(trace):
        if not isinstance(node, dict):
            continue
        item_match = _match_final_answer_items(node, config)
        if item_match:
            text, key, raw = item_match
            final_like.append((text, f"nested_item:{key}~{raw}"))
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
        item_patterns=(),
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

    features = {
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
    features.update(_dynamic_field_features(trace))
    return features

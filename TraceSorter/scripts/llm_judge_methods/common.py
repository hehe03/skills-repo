from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import default_final_answer_config, extract_features  # noqa: E402
from metrics import confusion_and_scores  # noqa: E402
from trace_io import TraceRecord, load_records, records_with_labels, split_records  # noqa: E402


LABELS = {"goodcase", "badcase"}


def load_eval_records(
    trace_path: str,
    metadata: str | None,
    *,
    split: str | None = None,
) -> List[TraceRecord]:
    records = load_records(trace_path, metadata)
    if split:
        records = split_records(records, split)
        if not records:
            raise ValueError(f"metadata split column has no samples for split: {split}")
    return records


def load_train_eval_records(
    trace_path: str,
    metadata: str | None,
    *,
    train_trace_path: str | None = None,
    train_metadata: str | None = None,
    train_split: str | None = None,
    eval_split: str | None = None,
) -> tuple[List[TraceRecord], List[TraceRecord], str, str]:
    all_records = load_records(trace_path, metadata)
    if eval_split:
        eval_records = split_records(all_records, eval_split)
        if not eval_records:
            raise ValueError(f"metadata split column has no samples for eval split: {eval_split}")
        eval_source = f"trace_path split={eval_split}"
    else:
        eval_records = all_records
        eval_source = "trace_path all samples"

    if train_trace_path:
        train_records = load_records(train_trace_path, train_metadata or metadata)
        if train_split:
            train_records = split_records(train_records, train_split)
            if not train_records:
                raise ValueError(f"training metadata split column has no samples for split: {train_split}")
        train_source = f"train_trace_path={train_trace_path}" + (f", split={train_split}" if train_split else "")
    elif train_split:
        train_records = split_records(all_records, train_split)
        if not train_records:
            raise ValueError(f"metadata split column has no samples for train split: {train_split}")
        train_source = f"trace_path split={train_split}"
    else:
        raise ValueError("rubric iteration requires --train-split or --train-trace-path with labeled records")

    train_records = records_with_labels(train_records)
    labels = {record.label for record in train_records}
    if not {"goodcase", "badcase"}.issubset(labels):
        raise ValueError("training records must contain both goodcase and badcase labels")
    return train_records, eval_records, train_source, eval_source


def timestamped_path(output_dir: str | Path, prefix: str, suffix: str = ".md") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{prefix}_{timestamp}{suffix}"


def json_excerpt(value: Any, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _first_text_field(trace: Any, keys: Iterable[str]) -> str:
    if not isinstance(trace, dict):
        return ""
    for key in keys:
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_tool_name(value: Any) -> str:
    text = str(value or "unknown_tool").strip()
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text[:64] or "unknown_tool"


def _command_name(command: Any) -> str:
    if isinstance(command, dict):
        for key in ("name", "tool", "tool_name", "action", "operation"):
            if command.get(key):
                return _safe_tool_name(command[key])
    return "unknown_tool"


def _command_args(command: Any) -> Dict[str, Any]:
    if not isinstance(command, dict):
        return {}
    for key in ("args", "arguments", "params", "input"):
        value = command.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            return {"input": value}
    return {}


def _step_result_text(step: Dict[str, Any], max_chars: int) -> str:
    chunks = []
    for key in ("result", "output", "observation", "content", "response"):
        if key in step:
            chunks.append(f"{key}: {json_excerpt(step[key], max_chars)}")
    return "\n".join(chunks)[:max_chars]


def trace_to_openai_messages(record: TraceRecord, *, max_trace_chars: int = 4000) -> List[Dict[str, Any]]:
    """Convert arbitrary project traces into a best-effort OpenAI-style trajectory.

    AgentEvals expects message trajectories with optional tool_calls. The project traces are
    not guaranteed to be LangChain/OpenAI messages, so this adapter preserves tool-like
    steps when possible and otherwise falls back to a compact JSON excerpt.
    """
    trace = record.trace
    if isinstance(trace, dict):
        raw_messages = trace.get("messages")
        if isinstance(raw_messages, list) and all(isinstance(item, dict) for item in raw_messages):
            normalized: List[Dict[str, Any]] = []
            for item in raw_messages:
                role = item.get("role") or "assistant"
                normalized.append(
                    {
                        "role": "assistant" if role == "ai" else role,
                        "content": item.get("content") if item.get("content") is not None else "",
                        **({"tool_calls": item["tool_calls"]} if item.get("tool_calls") else {}),
                    }
                )
            return normalized

    goal = _first_text_field(
        trace,
        ("input", "question", "task", "prompt", "user_request", "goal", "instruction"),
    )
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": goal or f"Agent trace record: {record.name}. Infer the task goal from the trajectory when possible.",
        }
    ]

    steps = trace.get("plan_list") if isinstance(trace, dict) else None
    if not isinstance(steps, list):
        steps = trace.get("steps") if isinstance(trace, dict) else None
    if not isinstance(steps, list):
        messages.append({"role": "assistant", "content": json_excerpt(trace, max_trace_chars)})
        return messages

    remaining_chars = max_trace_chars
    for index, step in enumerate(steps):
        if remaining_chars <= 0:
            break
        if not isinstance(step, dict):
            content = f"Step {index}: {json_excerpt(step, min(remaining_chars, 600))}"
            messages.append({"role": "assistant", "content": content})
            remaining_chars -= len(content)
            continue
        command = step.get("command") or {
            key: step.get(key)
            for key in ("name", "tool", "tool_name", "action", "operation", "args")
            if key in step
        }
        if command:
            name = _command_name(command)
            args = _command_args(command)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Step {index}: call tool `{name}`.",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            result_text = _step_result_text(step, min(remaining_chars, 1200))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": result_text or "",
                }
            )
            remaining_chars -= len(result_text) + 100
        else:
            content = f"Step {index}: {json_excerpt(step, min(remaining_chars, 800))}"
            messages.append({"role": "assistant", "content": content})
            remaining_chars -= len(content)
    return messages


def compact_record_payload(record: TraceRecord, *, max_trace_chars: int = 1600) -> Dict[str, Any]:
    features = extract_features(record, default_final_answer_config())
    keep_features = {
        key: features.get(key)
        for key in (
            "step_count",
            "error_count",
            "has_error_text",
            "empty_result_ratio",
            "nonempty_result_ratio",
            "repeated_action_count",
            "max_consecutive_same_action",
            "has_final_answer",
            "text_chars",
        )
        if key in features
    }
    field_paths = str(features.get("trace_field_paths", "")).splitlines()
    return {
        "name": record.name,
        "label": record.label,
        "source": record.source,
        "split": record.split,
        "features": keep_features,
        "top_field_paths": field_paths[:80],
        "trace_excerpt": json_excerpt(record.trace, max_trace_chars),
    }


def sample_labeled_records(
    records: List[TraceRecord],
    *,
    per_label: int,
) -> List[TraceRecord]:
    selected: List[TraceRecord] = []
    for label in ("goodcase", "badcase"):
        label_records = [record for record in records if record.label == label]
        selected.extend(label_records[:per_label])
    return selected


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def result_score_parts(raw_result: Any, *, threshold: float) -> tuple[Any, str, str, float, float]:
    score: Any = None
    reasoning = ""
    raw_text = repr(raw_result)
    if isinstance(raw_result, dict):
        score = raw_result.get("score")
        reasoning = str(raw_result.get("reasoning") or raw_result.get("comment") or "")
    else:
        for attr in ("score", "value"):
            if hasattr(raw_result, attr):
                score = getattr(raw_result, attr)
                break
        for attr in ("reasoning", "comment", "commentary"):
            if hasattr(raw_result, attr):
                reasoning = str(getattr(raw_result, attr) or "")
                break
    if isinstance(score, tuple) and score:
        score, reasoning = score[0], str(score[1] if len(score) > 1 else reasoning)

    if isinstance(score, bool):
        predicted = "goodcase" if score else "badcase"
        good_score = 1.0 if score else 0.0
    else:
        try:
            good_score = float(score)
        except (TypeError, ValueError):
            good_score = 0.5
        predicted = "goodcase" if good_score >= threshold else "badcase"
    bad_score = round(1.0 - good_score, 4)
    return score, predicted, reasoning, round(good_score, 4), bad_score


def write_llm_judge_report(
    path: str | Path,
    *,
    title: str,
    method: str,
    trace_path: str,
    metadata_path: str | None,
    model: str,
    results: List[Dict[str, Any]],
    notes: List[str],
    max_rows: int,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = confusion_and_scores(results)
    lines = [
        f"# {title}",
        "",
        f"- Method: `{method}`",
        f"- Trace path: `{trace_path}`",
        f"- Metadata: `{metadata_path}`" if metadata_path else "- Metadata: none",
        f"- Model: `{model}`",
        f"- Samples evaluated: {len(results)}",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in notes],
        "",
    ]
    if metrics["count"]:
        lines.extend(
            [
                "## Metrics",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| accuracy | {metrics['accuracy']} |",
                f"| precision(badcase) | {metrics['precision']} |",
                f"| recall(badcase) | {metrics['recall']} |",
                f"| f1(badcase) | {metrics['f1']} |",
                "",
                "| confusion | count |",
                "|---|---:|",
                f"| tp | {metrics['tp']} |",
                f"| fp | {metrics['fp']} |",
                f"| tn | {metrics['tn']} |",
                f"| fn | {metrics['fn']} |",
                "",
            ]
        )
    lines.extend(
        [
            "## Predictions",
            "",
            "| name | label | predicted | raw_score | bad_score | good_score | reason |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in results[:max_rows]:
        reason = str(row.get("reason", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{row.get('name')}` | {row.get('label') or ''} | {row.get('predicted_label')} | "
            f"{row.get('raw_score')} | {row.get('bad_score')} | {row.get('good_score')} | {reason} |"
        )
    if len(results) > max_rows:
        lines.append(f"| ... | ... | ... | ... | ... | ... | truncated at {max_rows} rows |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output

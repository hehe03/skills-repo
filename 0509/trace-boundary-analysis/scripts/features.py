import json
import math
import re
import statistics
from typing import Any

from common import OUTLINE_TASK_KEYWORD, normalize_text, task_signature


FEATURE_NAMES = [
    "step_count_log",
    "unique_task_ratio",
    "unique_command_ratio",
    "result_nonempty_ratio",
    "avg_result_log",
    "final_result_log",
    "max_same_run",
    "max_loop_repeats",
    "revisit_ratio",
    "missing_command_ratio",
    "has_outline",
    "outline_count_log",
    "query_task_overlap",
    "task_entropy",
    "command_entropy",
    "unique_result_ratio",
    "max_same_result_run",
    "final_result_nonempty",
    "result_entropy",
]


def tokenize(text: str) -> set[str]:
    text = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", text))
    chinese_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    return words | chinese_chars


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    return -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())


def longest_same_run(values: list[str]) -> int:
    if not values:
        return 0
    longest = 1
    current = 1
    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def max_loop_repeats(values: list[str]) -> int:
    total = len(values)
    best = 1 if total else 0
    for window_size in range(1, total // 2 + 1):
        for start in range(0, total - window_size + 1):
            pattern = values[start : start + window_size]
            repeats = 1
            cursor = start + window_size
            while cursor + window_size <= total and values[cursor : cursor + window_size] == pattern:
                repeats += 1
                cursor += window_size
            best = max(best, repeats)
    return best


def extract_features(trace: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    plan_list = trace.get("plan_list")
    if not isinstance(plan_list, list):
        plan_list = []

    tasks = [task for task in plan_list if isinstance(task, dict)]
    step_count = len(tasks)
    task_names = [normalize_text(task.get("task_name")) for task in tasks]
    result_texts = [normalize_text(task.get("result")) for task in tasks]
    result_lengths = [len(result_text) for result_text in result_texts]

    command_names: list[str] = []
    missing_command_count = 0
    for task in tasks:
        command = task.get("command")
        if isinstance(command, dict):
            command_names.append(normalize_text(command.get("name")))
        else:
            command_names.append("")
            missing_command_count += 1

    signatures = [task_signature(task) for task in tasks]
    unique_task_count = len(set(task_names))
    unique_command_count = len(set(command_names))
    unique_result_count = len(set(result_texts))
    nonempty_results = sum(1 for result_text in result_texts if result_text.strip())
    outline_count = sum(1 for task_name in task_names if OUTLINE_TASK_KEYWORD in task_name)

    query_tokens = tokenize(normalize_text(trace.get("query")))
    task_tokens = tokenize(" ".join(task_names))
    overlap = len(query_tokens & task_tokens) / len(query_tokens | task_tokens) if query_tokens and task_tokens else 0.0
    revisit_count = max(0, step_count - unique_task_count)
    avg_result_log = statistics.mean(math.log1p(length) for length in result_lengths) if result_lengths else 0.0
    final_result_log = math.log1p(result_lengths[-1]) if result_lengths else 0.0

    features = {
        "step_count_log": math.log1p(step_count),
        "unique_task_ratio": unique_task_count / step_count if step_count else 0.0,
        "unique_command_ratio": unique_command_count / step_count if step_count else 0.0,
        "result_nonempty_ratio": nonempty_results / step_count if step_count else 0.0,
        "avg_result_log": avg_result_log,
        "final_result_log": final_result_log,
        "max_same_run": float(longest_same_run(signatures)),
        "max_loop_repeats": float(max_loop_repeats(signatures)),
        "revisit_ratio": revisit_count / step_count if step_count else 0.0,
        "missing_command_ratio": missing_command_count / step_count if step_count else 0.0,
        "has_outline": 1.0 if outline_count else 0.0,
        "outline_count_log": math.log1p(outline_count),
        "query_task_overlap": overlap,
        "task_entropy": entropy(task_names),
        "command_entropy": entropy(command_names),
        "unique_result_ratio": unique_result_count / step_count if step_count else 0.0,
        "max_same_result_run": float(longest_same_run(result_texts)),
        "final_result_nonempty": 1.0 if result_texts and result_texts[-1].strip() else 0.0,
        "result_entropy": entropy(result_texts),
    }
    summary = {
        "step_count": step_count,
        "outline_count": outline_count,
        "max_same_run": int(features["max_same_run"]),
        "max_loop_repeats": int(features["max_loop_repeats"]),
        "max_same_result_run": int(features["max_same_result_run"]),
        "result_nonempty_ratio": features["result_nonempty_ratio"],
        "unique_task_ratio": features["unique_task_ratio"],
        "unique_result_ratio": features["unique_result_ratio"],
        "missing_command_ratio": features["missing_command_ratio"],
        "final_result_length": result_lengths[-1] if result_lengths else 0,
    }
    return features, summary

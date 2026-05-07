import argparse
import csv
import json
import math
import re
import sys
import statistics
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_eval_utils import (  # noqa: E402
    BAD_LABEL,
    GOOD_LABEL,
    Prediction,
    filter_metadata_by_split,
    load_metadata,
    load_trace_records,
    print_metrics_summary,
    print_predictions,
    write_metrics_markdown,
    write_predictions_tsv,
)

OUTLINE_TASK_KEYWORD = "生成大纲"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised trace classifier for good case / bad case."
    )
    parser.add_argument("input_dir", help="Directory that contains trace JSON files.")
    parser.add_argument(
        "metadata_csv",
        help="Metadata CSV with columns: name,label,source,split.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Use only this split as evaluation set. Default: all samples.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Good-case threshold for final score. Default: 0.55.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV file for per-sample predictions.",
    )
    parser.add_argument(
        "--fields",
        help="Comma-separated top-level trace fields used for classification, for example: query,plan_list. Default: all fields.",
    )
    parser.add_argument(
        "--bad-risk-threshold",
        type=float,
        default=0.55,
        help="Mark as bad case when bad-risk score reaches this threshold. Default: 0.55.",
    )
    parser.add_argument(
        "--bad-risk-weight",
        type=float,
        default=0.45,
        help="Penalty weight applied to bad-risk score in final score. Default: 0.45.",
    )
    parser.add_argument(
        "--centrality-weight",
        type=float,
        default=0.10,
        help="Weight for batch centrality adjustment. Default: 0.10.",
    )
    parser.add_argument(
        "--metrics-output",
        default="unsupervised_metrics.md",
        help="Markdown metrics output path. Default: unsupervised_metrics.md.",
    )
    return parser.parse_args()


def parse_field_names(fields: str | None) -> list[str] | None:
    if fields is None:
        return None

    field_names = [field.strip() for field in fields.split(",") if field.strip()]
    if not field_names:
        raise ValueError("--fields must contain at least one top-level field name.")

    invalid_fields = [
        field
        for field in field_names
        if "." in field or "[" in field or "]" in field
    ]
    if invalid_fields:
        raise ValueError(
            "--fields only supports top-level fields: " + ", ".join(invalid_fields)
        )

    return field_names


def filter_trace_fields(trace: dict[str, Any], field_names: list[str] | None) -> dict[str, Any]:
    if field_names is None:
        return trace
    return {field_name: trace[field_name] for field_name in field_names if field_name in trace}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def log_len(value: Any) -> float:
    return math.log1p(len(normalize_text(value)))


def task_signature(task: dict[str, Any]) -> str:
    task_name = normalize_text(task.get("task_name"))
    command = task.get("command")
    command_name = ""
    command_args = ""
    if isinstance(command, dict):
        command_name = normalize_text(command.get("name"))
        command_args = normalize_text(command.get("args"))
    return f"{task_name}||{command_name}||{command_args}"


def longest_same_run(signatures: list[str]) -> int:
    if not signatures:
        return 0
    longest = 1
    current = 1
    for index in range(1, len(signatures)):
        if signatures[index] == signatures[index - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def max_loop_repeats(signatures: list[str]) -> int:
    total = len(signatures)
    best = 1 if total else 0
    for window_size in range(1, total // 2 + 1):
        for start in range(0, total - window_size + 1):
            pattern = signatures[start : start + window_size]
            repeats = 1
            cursor = start + window_size
            while cursor + window_size <= total and signatures[cursor : cursor + window_size] == pattern:
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
    results = [task.get("result") for task in tasks]
    result_texts = [normalize_text(result) for result in results]
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
    overlap = 0.0
    if query_tokens and task_tokens:
        overlap = len(query_tokens & task_tokens) / len(query_tokens | task_tokens)

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
        "final_result_nonempty": bool(features["final_result_nonempty"]),
    }
    return features, summary


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def behavior_prior(features: dict[str, float]) -> float:
    length_penalty = abs(features["step_count_log"] - math.log1p(6)) / 3.0
    repeat_penalty = min((features["max_same_run"] - 1.0) / 3.0, 1.0)
    loop_penalty = min((features["max_loop_repeats"] - 1.0) / 3.0, 1.0)

    raw = 0.0
    raw += 1.15 * features["result_nonempty_ratio"]
    raw += 0.75 * features["unique_task_ratio"]
    raw += 0.55 * features["unique_command_ratio"]
    raw += 0.55 * min(features["avg_result_log"] / 5.0, 1.0)
    raw += 0.35 * min(features["final_result_log"] / 5.0, 1.0)
    raw += 0.50 * features["has_outline"]
    raw += 0.25 * features["query_task_overlap"]
    raw -= 1.15 * loop_penalty
    raw -= 0.95 * repeat_penalty
    raw -= 0.55 * features["missing_command_ratio"]
    raw -= 0.35 * min(length_penalty, 1.0)
    raw -= 1.35
    return sigmoid(raw)


def bad_risk_score(features: dict[str, float]) -> float:
    repeat_penalty = clamp((features["max_same_run"] - 1.0) / 2.0)
    loop_penalty = clamp((features["max_loop_repeats"] - 1.0) / 2.0)
    same_result_penalty = clamp((features["max_same_result_run"] - 1.0) / 2.0)
    low_final_result = 1.0 - clamp(features["final_result_log"] / math.log1p(80))
    low_avg_result = 1.0 - clamp(features["avg_result_log"] / math.log1p(60))
    low_task_diversity = 1.0 - features["unique_task_ratio"]
    low_result_diversity = 1.0 - features["unique_result_ratio"]
    no_outline = 1.0 - features["has_outline"]

    raw = 0.0
    raw += 1.50 * loop_penalty
    raw += 1.25 * repeat_penalty
    raw += 1.00 * same_result_penalty
    raw += 0.90 * (1.0 - features["result_nonempty_ratio"])
    raw += 0.75 * low_final_result
    raw += 0.50 * low_avg_result
    raw += 0.65 * low_task_diversity
    raw += 0.45 * low_result_diversity
    raw += 0.55 * features["revisit_ratio"]
    raw += 0.55 * features["missing_command_ratio"]
    raw += 0.55 * no_outline
    raw -= 0.35 * features["query_task_overlap"]
    raw -= 1.65
    return sigmoid(raw)


def robust_values(records: list[dict[str, float]]) -> list[list[float]]:
    columns: dict[str, list[float]] = {
        name: [record[name] for record in records] for name in FEATURE_NAMES
    }
    medians = {name: statistics.median(values) for name, values in columns.items()}
    iqrs: dict[str, float] = {}
    for name, values in columns.items():
        sorted_values = sorted(values)
        if len(sorted_values) < 4:
            iqrs[name] = max(statistics.pstdev(sorted_values), 1.0)
        else:
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[(len(sorted_values) * 3) // 4]
            iqrs[name] = max(q3 - q1, 1e-6)

    scaled: list[list[float]] = []
    for record in records:
        scaled.append([(record[name] - medians[name]) / iqrs[name] for name in FEATURE_NAMES])
    return scaled


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def centrality_scores(records: list[dict[str, float]]) -> list[float]:
    if len(records) < 3:
        return [0.5 for _ in records]

    scaled = robust_values(records)
    raw_scores: list[float] = []
    for index, vector in enumerate(scaled):
        distances = [
            euclidean(vector, other)
            for other_index, other in enumerate(scaled)
            if other_index != index
        ]
        median_distance = statistics.median(distances)
        raw_scores.append(1.0 / (1.0 + median_distance))

    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score - min_score < 1e-9:
        return [0.5 for _ in raw_scores]
    return [(score - min_score) / (max_score - min_score) for score in raw_scores]


def build_reason(summary: dict[str, Any]) -> str:
    reasons: list[str] = []
    if summary["max_loop_repeats"] >= 2:
        reasons.append(f"连续循环次数={summary['max_loop_repeats']}")
    if summary["max_same_run"] >= 2:
        reasons.append(f"连续重复次数={summary['max_same_run']}")
    if summary["outline_count"]:
        reasons.append(f"生成大纲任务数={summary['outline_count']}")
    reasons.append(f"result非空比例={summary['result_nonempty_ratio']:.2f}")
    reasons.append(f"任务数={summary['step_count']}")
    return "；".join(reasons)


def build_reason(summary: dict[str, Any]) -> str:
    reasons: list[str] = []
    if summary["max_loop_repeats"] >= 2:
        reasons.append(f"连续循环次数={summary['max_loop_repeats']}")
    if summary["max_same_run"] >= 2:
        reasons.append(f"连续重复次数={summary['max_same_run']}")
    if summary["max_same_result_run"] >= 2:
        reasons.append(f"连续重复结果次数={summary['max_same_result_run']}")
    if summary["outline_count"]:
        reasons.append(f"生成大纲任务数={summary['outline_count']}")
    if summary["missing_command_ratio"] > 0:
        reasons.append(f"缺失command比例={summary['missing_command_ratio']:.2f}")
    reasons.append(f"result非空比例={summary['result_nonempty_ratio']:.2f}")
    reasons.append(f"结果多样性={summary['unique_result_ratio']:.2f}")
    reasons.append(f"最终结果长度={summary['final_result_length']}")
    reasons.append(f"任务数={summary['step_count']}")
    return "；".join(reasons)


def main() -> int:
    args = parse_args()
    try:
        field_names = parse_field_names(args.fields)
    except ValueError as exc:
        print(exc)
        return 1

    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return 1

    if not metadata_csv.is_file():
        print(f"Metadata CSV does not exist: {metadata_csv}")
        return 1

    metadata = filter_metadata_by_split(load_metadata(metadata_csv), args.split)
    if not metadata:
        print("No metadata rows selected.")
        return 0

    records = load_trace_records(input_dir, metadata)
    items: list[tuple[str, dict[str, float], dict[str, Any]]] = []
    for record in records:
        trace = filter_trace_fields(record.trace, field_names)
        features, summary = extract_features(trace)
        items.append((record.meta.name, features, summary))

    centralities = centrality_scores([features for _, features, _ in items])
    use_centrality = len(items) >= 3
    predictions: list[Prediction] = []
    for (sample_name, features, summary), centrality in zip(items, centralities):
        prior = behavior_prior(features)
        risk = bad_risk_score(features)
        centrality_adjustment = args.centrality_weight * (centrality - 0.5) if use_centrality else 0.0
        score = clamp(prior + centrality_adjustment - args.bad_risk_weight * risk)
        predicted_label = BAD_LABEL if risk >= args.bad_risk_threshold else GOOD_LABEL if score >= args.threshold else BAD_LABEL
        meta = next(record.meta for record in records if record.meta.name == sample_name)
        predictions.append(
            Prediction(
                name=sample_name,
                source=meta.source,
                split=meta.split,
                actual_label=meta.label,
                predicted_label=predicted_label,
                detail={
                    "score": f"{score:.3f}",
                    "behavior_prior": f"{prior:.3f}",
                    "bad_risk": f"{risk:.3f}",
                    "centrality": f"{centrality:.3f}" if use_centrality else "N/A",
                    "reason": build_reason(summary),
                },
            )
        )

    print_predictions(predictions)
    print_metrics_summary(predictions)
    if args.output:
        write_predictions_tsv(predictions, Path(args.output))
    write_metrics_markdown(predictions, Path(args.metrics_output), "Unsupervised Trace Classification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

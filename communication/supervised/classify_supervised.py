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
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot supervised trace classifier for good case / bad case."
    )
    parser.add_argument("input_dir", help="Directory that contains trace JSON files.")
    parser.add_argument(
        "metadata_csv",
        help="Metadata CSV with columns: name,label,source,split.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Good-case threshold for supervised score. Default: 0.5.",
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
        "--metrics-output",
        default="supervised_metrics.md",
        help="Markdown metrics output path. Default: supervised_metrics.md.",
    )
    return parser.parse_args(argv)


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


def extract_features(trace: dict[str, Any]) -> dict[str, float]:
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

    return {
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
    }


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]


def fit_scaler(records: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = sorted(record[name] for record in records)
        medians[name] = statistics.median(values)
        if len(values) < 4:
            scales[name] = max(statistics.pstdev(values), 1.0)
        else:
            q1 = values[len(values) // 4]
            q3 = values[(len(values) * 3) // 4]
            scales[name] = max(q3 - q1, 1e-6)
    return medians, scales


def scale_features(
    features: dict[str, float], medians: dict[str, float], scales: dict[str, float]
) -> list[float]:
    return [(features[name] - medians[name]) / scales[name] for name in FEATURE_NAMES]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    return [
        statistics.mean(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def nearest_example(
    vector: list[float], examples: list[tuple[str, list[float]]]
) -> tuple[str, float]:
    nearest_name = ""
    nearest_distance = float("inf")
    for name, example_vector in examples:
        distance = euclidean(vector, example_vector)
        if distance < nearest_distance:
            nearest_name = name
            nearest_distance = distance
    return nearest_name, nearest_distance


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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

    metadata = load_metadata(metadata_csv)
    train_metadata = [row for row in metadata if row.split == "train"]
    test_metadata = [row for row in metadata if row.split == "test"]
    if not train_metadata:
        print("No train rows found in metadata.")
        return 1
    if not test_metadata:
        print("No test rows found in metadata.")
        return 1

    train_records = load_trace_records(input_dir, train_metadata)
    test_records = load_trace_records(input_dir, test_metadata)

    train_features: list[tuple[str, str, dict[str, float]]] = []
    for record in train_records:
        trace = filter_trace_fields(record.trace, field_names)
        features = extract_features(trace)
        train_features.append((record.meta.name, record.meta.label, features))

    good_count = sum(1 for _, label, _ in train_features if label == GOOD_LABEL)
    bad_count = sum(1 for _, label, _ in train_features if label == BAD_LABEL)
    if good_count == 0 or bad_count == 0:
        print("Train split must include at least one goodcase and one badcase sample.")
        return 1

    medians, scales = fit_scaler([features for _, _, features in train_features])
    scaled_labeled = [
        (name, label, scale_features(features, medians, scales))
        for name, label, features in train_features
    ]
    good_examples = [
        (name, vector) for name, label, vector in scaled_labeled if label == GOOD_LABEL
    ]
    bad_examples = [
        (name, vector) for name, label, vector in scaled_labeled if label == BAD_LABEL
    ]
    good_centroid = mean_vector([vector for _, vector in good_examples])
    bad_centroid = mean_vector([vector for _, vector in bad_examples])

    distance_scale = max(
        statistics.mean(
            euclidean(vector, good_centroid if label == GOOD_LABEL else bad_centroid)
            for _, label, vector in scaled_labeled
        ),
        1.0,
    )

    predictions: list[Prediction] = []
    for record in test_records:
        trace = filter_trace_fields(record.trace, field_names)
        features = extract_features(trace)
        vector = scale_features(features, medians, scales)
        distance_to_good = euclidean(vector, good_centroid)
        distance_to_bad = euclidean(vector, bad_centroid)
        score = sigmoid((distance_to_bad - distance_to_good) / distance_scale)
        predicted_label = GOOD_LABEL if score >= args.threshold else BAD_LABEL
        nearest_good, _ = nearest_example(vector, good_examples)
        nearest_bad, _ = nearest_example(vector, bad_examples)
        predictions.append(
            Prediction(
                name=record.meta.name,
                source=record.meta.source,
                split=record.meta.split,
                actual_label=record.meta.label,
                predicted_label=predicted_label,
                detail={
                    "score": f"{score:.3f}",
                    "confidence": f"{abs(score - 0.5) * 2:.3f}",
                    "nearest_good": nearest_good,
                    "nearest_bad": nearest_bad,
                },
            )
        )

    print_predictions(predictions)
    print_metrics_summary(predictions)
    if args.output:
        write_predictions_tsv(predictions, Path(args.output))
    write_metrics_markdown(predictions, Path(args.metrics_output), "Supervised Trace Classification")
    return 0


if __name__ == "__main__":
    # Set INLINE_ARGS to run from an editor without command-line arguments.
    # Example:
    # INLINE_ARGS = [r".\traces", r".\metadata.csv"]
    INLINE_ARGS: list[str] | None = None
    INLINE_ARGS = [r"D:\Data\agent\trace\all", r"..\..\高交all.csv", "--threshold", '0.85']
    raise SystemExit(main(INLINE_ARGS))

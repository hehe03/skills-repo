import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GOOD_LABEL = "good case"
BAD_LABEL = "bad case"
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


@dataclass(frozen=True)
class TraceSample:
    name: str
    trace: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot supervised trace classifier for good case / bad case."
    )
    parser.add_argument(
        "input_path",
        help="JSON/JSONL file or directory that contains trace files.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="CSV/TSV label file with columns: file,label.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSON files recursively.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Good-case threshold for supervised score. Default: 0.5.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV output path. Uses utf-8-sig for Windows Excel.",
    )
    parser.add_argument(
        "--fields",
        help="Comma-separated top-level trace fields used for classification, for example: query,plan_list. Default: all fields.",
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


def normalize_label(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned in {"good", "good case", "goodcase", "1", "true", "是"}:
        return GOOD_LABEL
    if cleaned in {"bad", "bad case", "badcase", "0", "false", "否"}:
        return BAD_LABEL
    raise ValueError(f"Unknown label: {value}")


def load_labels(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        sample = file.read(2048)
        file.seek(0)
        delimiter = "\t" if "\t" in sample and "," not in sample else ","
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames and {"file", "label"}.issubset(set(reader.fieldnames)):
            return {
                normalize_text(row["file"]): normalize_label(normalize_text(row["label"]))
                for row in reader
                if row.get("file") and row.get("label")
            }

        file.seek(0)
        plain_reader = csv.reader(file, delimiter=delimiter)
        labels: dict[str, str] = {}
        for row in plain_reader:
            if len(row) >= 2:
                labels[normalize_text(row[0])] = normalize_label(normalize_text(row[1]))
        return labels


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("trace JSON root must be an object")
    return data


def iter_trace_files(input_dir: Path, recursive: bool) -> list[Path]:
    json_pattern = "**/*.json" if recursive else "*.json"
    jsonl_pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted([*input_dir.glob(json_pattern), *input_dir.glob(jsonl_pattern)])


def make_error_row(sample_name: str, message: str) -> dict[str, Any]:
    return {
        "file": sample_name,
        "label": BAD_LABEL,
        "score": "0.000",
        "confidence": "1.000",
        "nearest_good": "",
        "nearest_bad": "",
        "known_label": f"load_error={message}",
    }


def append_jsonl_samples(
    path: Path,
    sample_prefix: str | None,
    samples: list[TraceSample],
    error_rows: list[dict[str, Any]],
) -> None:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            fallback_name = (
                str(line_number)
                if sample_prefix is None
                else f"{sample_prefix}:{line_number}"
            )
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("trace JSONL line root must be an object")
                sample_name = normalize_text(data.get("sample_name")) or fallback_name
                samples.append(TraceSample(sample_name, data))
            except Exception as exc:
                error_rows.append(make_error_row(fallback_name, str(exc)))


def load_trace_samples(input_path: Path, recursive: bool) -> tuple[list[TraceSample], list[dict[str, Any]]]:
    samples: list[TraceSample] = []
    error_rows: list[dict[str, Any]] = []

    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            try:
                samples.append(TraceSample(input_path.name, load_json(input_path)))
            except Exception as exc:
                error_rows.append(make_error_row(input_path.name, str(exc)))
        elif suffix == ".jsonl":
            append_jsonl_samples(input_path, None, samples, error_rows)
        else:
            error_rows.append(make_error_row(input_path.name, "input file must be .json or .jsonl"))
        return samples, error_rows

    for trace_file in iter_trace_files(input_path, recursive):
        sample_prefix = trace_file.relative_to(input_path).as_posix()
        suffix = trace_file.suffix.lower()
        if suffix == ".json":
            try:
                samples.append(TraceSample(sample_prefix, load_json(trace_file)))
            except Exception as exc:
                error_rows.append(make_error_row(sample_prefix, str(exc)))
        elif suffix == ".jsonl":
            append_jsonl_samples(trace_file, sample_prefix, samples, error_rows)

    return samples, error_rows


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


def matched_label_key(sample_name: str, labels: dict[str, str]) -> str | None:
    rel = normalize_text(sample_name).replace("\\", "/")
    candidates = [
        rel,
        Path(rel).name,
    ]
    for candidate in candidates:
        if candidate in labels:
            return candidate
    return None


def write_rows(rows: list[dict[str, Any]], output: str | None) -> None:
    fieldnames = [
        "file",
        "label",
        "score",
        "confidence",
        "nearest_good",
        "nearest_bad",
        "known_label",
    ]
    if output:
        with Path(output).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return

    print("\t".join(fieldnames))
    for row in rows:
        print("\t".join(str(row[name]) for name in fieldnames))


def main() -> int:
    args = parse_args()
    try:
        field_names = parse_field_names(args.fields)
    except ValueError as exc:
        print(exc)
        return 1

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return 1

    labels = load_labels(Path(args.labels))
    samples, rows = load_trace_samples(input_path, args.recursive)
    if not samples and not rows:
        print("No JSON/JSONL samples found.")
        return 0

    records: list[tuple[str, dict[str, float], str | None]] = []
    labeled_features: list[tuple[str, str, dict[str, float]]] = []

    for sample in samples:
        try:
            trace = filter_trace_fields(sample.trace, field_names)
            features = extract_features(trace)
            label_key = matched_label_key(sample.name, labels)
            known_label = labels[label_key] if label_key else None
            records.append((sample.name, features, known_label))
            if known_label:
                labeled_features.append((sample.name, known_label, features))
        except Exception as exc:
            rows.append(make_error_row(sample.name, str(exc)))

    good_count = sum(1 for _, label, _ in labeled_features if label == GOOD_LABEL)
    bad_count = sum(1 for _, label, _ in labeled_features if label == BAD_LABEL)
    if good_count == 0 or bad_count == 0:
        print("Label file must include at least one good case and one bad case sample.")
        return 1
    if good_count > 5 or bad_count > 5:
        print("Warning: recommended labels are no more than 5 samples per class.")

    medians, scales = fit_scaler([features for _, _, features in labeled_features])
    scaled_labeled = [
        (name, label, scale_features(features, medians, scales))
        for name, label, features in labeled_features
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

    for sample_name, features, known_label in records:
        vector = scale_features(features, medians, scales)
        distance_to_good = euclidean(vector, good_centroid)
        distance_to_bad = euclidean(vector, bad_centroid)
        score = sigmoid((distance_to_bad - distance_to_good) / distance_scale)
        label = GOOD_LABEL if score >= args.threshold else BAD_LABEL
        nearest_good, _ = nearest_example(vector, good_examples)
        nearest_bad, _ = nearest_example(vector, bad_examples)
        rows.append(
            {
                "file": sample_name,
                "label": label,
                "score": f"{score:.3f}",
                "confidence": f"{abs(score - 0.5) * 2:.3f}",
                "nearest_good": nearest_good,
                "nearest_bad": nearest_bad,
                "known_label": known_label or "",
            }
        )

    write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

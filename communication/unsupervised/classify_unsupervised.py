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
        description="Unsupervised trace classifier for good case / bad case."
    )
    parser.add_argument(
        "input_path",
        help="JSON/JSONL file or directory that contains trace files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSON files recursively.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Good-case threshold for final score. Default: 0.55.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV output path. Uses utf-8-sig for Windows Excel.",
    )
    return parser.parse_args()


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
    }

    summary = {
        "step_count": step_count,
        "outline_count": outline_count,
        "max_same_run": int(features["max_same_run"]),
        "max_loop_repeats": int(features["max_loop_repeats"]),
        "result_nonempty_ratio": features["result_nonempty_ratio"],
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
        "behavior_prior": "0.000",
        "centrality": "0.000",
        "reason": f"load_error={message}",
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

            sample_name = (
                str(line_number)
                if sample_prefix is None
                else f"{sample_prefix}:{line_number}"
            )
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("trace JSONL line root must be an object")
                samples.append(TraceSample(sample_name, data))
            except Exception as exc:
                error_rows.append(make_error_row(sample_name, str(exc)))


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


def write_rows(rows: list[dict[str, Any]], output: str | None) -> None:
    fieldnames = ["file", "label", "score", "behavior_prior", "centrality", "reason"]
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
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return 1

    samples, rows = load_trace_samples(input_path, args.recursive)
    if not samples and not rows:
        print("No JSON/JSONL samples found.")
        return 0

    items: list[tuple[str, dict[str, float], dict[str, Any]]] = []
    for sample in samples:
        try:
            features, summary = extract_features(sample.trace)
            items.append((sample.name, features, summary))
        except Exception as exc:
            rows.append(make_error_row(sample.name, str(exc)))

    centralities = centrality_scores([features for _, features, _ in items])
    use_centrality = len(items) >= 3
    for (sample_name, features, summary), centrality in zip(items, centralities):
        prior = behavior_prior(features)
        score = 0.75 * prior + 0.25 * centrality if use_centrality else prior
        label = GOOD_LABEL if score >= args.threshold else BAD_LABEL
        rows.append(
            {
                "file": sample_name,
                "label": label,
                "score": f"{score:.3f}",
                "behavior_prior": f"{prior:.3f}",
                "centrality": f"{centrality:.3f}" if use_centrality else "N/A",
                "reason": build_reason(summary),
            }
        )

    write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from pathlib import Path
from typing import Any

from trace_eval_utils import (
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify trace JSON files into good case or bad case."
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
        "--repeat-threshold",
        type=int,
        default=None,
        help="Enable repeated/loop task detection and mark as badcase when the pattern appears this many times. Default: disabled.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV file for per-sample predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default="classify_traces_metrics.md",
        help="Markdown metrics output path. Default: classify_traces_metrics.md.",
    )
    return parser.parse_args(argv)


def is_normal_result(result: Any) -> bool:
    if result is None:
        return False

    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        failure_markers = [
            "error",
            "failed",
            "exception",
            "traceback",
            "null",
            "none",
            "失败",
            "报错",
            "异常",
        ]
        return not any(marker in lowered for marker in failure_markers)

    if isinstance(result, (list, dict)):
        return len(result) > 0

    return True


def build_task_signature(task: dict[str, Any]) -> str:
    task_name = str(task.get("task_name", "")).strip()
    command = task.get("command")
    command_name = ""

    if isinstance(command, dict):
        command_name = str(command.get("name", "")).strip()

    return f"{task_name}||{command_name}"


def has_repeated_or_loop_tasks(plan_list: list[Any], repeat_threshold: int) -> bool:
    if repeat_threshold <= 1:
        return len(plan_list) > 0

    signatures = [
        build_task_signature(task)
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
            if not pattern:
                continue

            if all(
                signatures[start + offset * window_size : start + (offset + 1) * window_size]
                == pattern
                for offset in range(1, repeat_threshold)
            ):
                return True

    return False


def classify_trace(trace: dict[str, Any], repeat_threshold: int | None) -> str:
    plan_list = trace.get("plan_list")
    if not isinstance(plan_list, list):
        return BAD_LABEL

    if repeat_threshold is not None and has_repeated_or_loop_tasks(plan_list, repeat_threshold):
        return BAD_LABEL

    for task in plan_list:
        if not isinstance(task, dict):
            continue

        task_name = str(task.get("task_name", ""))
        result = task.get("result")

        if OUTLINE_TASK_KEYWORD in task_name and is_normal_result(result):
            return GOOD_LABEL

    return BAD_LABEL


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)
    repeat_threshold = args.repeat_threshold

    if repeat_threshold is not None and repeat_threshold < 2:
        print("--repeat-threshold must be greater than or equal to 2.")
        return 1

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return 1

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
    predictions: list[Prediction] = []
    for record in records:
        predicted_label = classify_trace(record.trace, repeat_threshold)
        predictions.append(
            Prediction(
                name=record.meta.name,
                source=record.meta.source,
                split=record.meta.split,
                actual_label=record.meta.label,
                predicted_label=predicted_label,
                detail={},
            )
        )

    print_predictions(predictions)
    print_metrics_summary(predictions)
    if args.output:
        write_predictions_tsv(predictions, Path(args.output))
    write_metrics_markdown(predictions, Path(args.metrics_output), "Rule-based Trace Classification")
    return 0


if __name__ == "__main__":
    # Set INLINE_ARGS to run from an editor without command-line arguments.
    # Example:
    # INLINE_ARGS = [r".\traces", r".\metadata.csv", "--split", "test"]
    INLINE_ARGS: list[str] | None = None
    raise SystemExit(main(INLINE_ARGS))

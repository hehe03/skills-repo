import argparse
import json
from pathlib import Path
from typing import Any


GOOD_LABEL = "good case"
BAD_LABEL = "bad case"
OUTLINE_TASK_KEYWORD = "生成大纲"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify trace JSON files into good case or bad case."
    )
    parser.add_argument("input_dir", help="Directory that contains trace JSON files.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSON files recursively.",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=2,
        help="Mark as bad case when a consecutive repeated task or loop appears this many times. Default: 2.",
    )
    return parser.parse_args()


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


def classify_trace(trace: dict[str, Any], repeat_threshold: int) -> str:
    plan_list = trace.get("plan_list")
    if not isinstance(plan_list, list):
        return BAD_LABEL

    if has_repeated_or_loop_tasks(plan_list, repeat_threshold):
        return BAD_LABEL

    for task in plan_list:
        if not isinstance(task, dict):
            continue

        task_name = str(task.get("task_name", ""))
        result = task.get("result")

        if OUTLINE_TASK_KEYWORD in task_name and is_normal_result(result):
            return GOOD_LABEL

    return BAD_LABEL


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(input_dir.glob(pattern))


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    repeat_threshold = args.repeat_threshold

    if repeat_threshold < 2:
        print("--repeat-threshold must be greater than or equal to 2.")
        return 1

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return 1

    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return 1

    json_files = iter_json_files(input_dir, args.recursive)
    if not json_files:
        print("No JSON files found.")
        return 0

    for json_file in json_files:
        try:
            trace = load_json(json_file)
            label = classify_trace(trace, repeat_threshold)
            print(f"{json_file.name}\t{label}")
        except Exception as exc:
            print(f"{json_file.name}\t{BAD_LABEL}\tload_error={exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

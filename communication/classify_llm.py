import argparse
import json
import re
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


DEFAULT_PROMPT = """你是一个 Agent trace 质量评估器。

请判断每个 trace 是 goodcase 还是 badcase。

判定标准：
- goodcase：Agent 的执行过程能够有效完成用户请求，关键任务有合理结果，整体 trace 没有明显无效循环、重复执行、答非所问或产出不足。
- badcase：Agent 没有有效完成用户请求，或者存在明显循环/重复、关键结果缺失、输出过短、步骤混乱、答非所问等问题。

输出要求：
- 只能输出 JSON。
- 如果只有一个样本，输出：{"label":"goodcase"} 或 {"label":"badcase"}。
- 如果有多个样本，输出：{"样本名":"goodcase","样本名2":"badcase"}。
- label 只能是 goodcase 或 badcase。
"""


def llm_generate(query: str) -> str:
    # TODO: Fill in your LLM call here. Input is the complete query, output is the raw model result.
    return ""


def parse_args(argv: list[Any] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based trace classifier for goodcase / badcase."
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
        "--batch",
        type=int,
        default=1,
        help="Number of trace samples per LLM query. Default: 1.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional UTF-8 prompt file. If omitted, uses the built-in prompt.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV file for per-sample predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default="llm_metrics.md",
        help="Markdown metrics output path. Default: llm_metrics.md.",
    )
    return parser.parse_args([str(item) for item in argv] if argv is not None else None)


def load_prompt(prompt_file: str | None) -> str:
    if not prompt_file:
        return DEFAULT_PROMPT
    return Path(prompt_file).read_text(encoding="utf-8-sig")


def normalize_llm_label(value: Any) -> str | None:
    if isinstance(value, dict) and "label" in value:
        value = value.get("label")

    text = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if text in {"good", "goodcase", "good_case", "1", "true"}:
        return GOOD_LABEL
    if text in {"bad", "badcase", "bad_case", "0", "false"}:
        return BAD_LABEL
    if "badcase" in text or "bad_case" in text:
        return BAD_LABEL
    if "goodcase" in text or "good_case" in text:
        return GOOD_LABEL
    return None


def extract_json_text(raw_result: str) -> str | None:
    stripped = raw_result.strip()
    if not stripped:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return stripped[object_start : object_end + 1]

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end > array_start:
        return stripped[array_start : array_end + 1]

    return None


def labels_from_json(parsed: Any, sample_names: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}

    if isinstance(parsed, dict):
        if len(sample_names) == 1 and "label" in parsed:
            label = normalize_llm_label(parsed.get("label"))
            if label:
                labels[sample_names[0]] = label
            return labels

        for sample_name in sample_names:
            if sample_name in parsed:
                label = normalize_llm_label(parsed[sample_name])
                if label:
                    labels[sample_name] = label
        return labels

    if isinstance(parsed, list):
        for index, item in enumerate(parsed):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("sample_name") or "").strip()
                if not name and len(sample_names) == len(parsed):
                    name = sample_names[index]
                label = normalize_llm_label(item.get("label"))
                if name in sample_names and label:
                    labels[name] = label
            elif index < len(sample_names):
                label = normalize_llm_label(item)
                if label:
                    labels[sample_names[index]] = label
    return labels


def labels_from_text(raw_result: str, sample_names: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    if len(sample_names) == 1:
        label = normalize_llm_label(raw_result)
        if label:
            labels[sample_names[0]] = label
        return labels

    for line in raw_result.splitlines():
        if ":" in line:
            name_part, label_part = line.split(":", 1)
        elif "," in line:
            name_part, label_part = line.split(",", 1)
        elif "\t" in line:
            name_part, label_part = line.split("\t", 1)
        else:
            continue

        name = name_part.strip().strip("\"'`")
        label = normalize_llm_label(label_part)
        if name in sample_names and label:
            labels[name] = label
    return labels


def parse_llm_result(raw_result: str, sample_names: list[str]) -> tuple[dict[str, str], str]:
    json_text = extract_json_text(raw_result)
    if json_text:
        try:
            labels = labels_from_json(json.loads(json_text), sample_names)
            if labels:
                return labels, "json"
        except json.JSONDecodeError:
            pass

    labels = labels_from_text(raw_result, sample_names)
    if labels:
        return labels, "text"

    return {}, "parse_error"


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_query(prompt: str, batch_records: list[Any]) -> str:
    samples = [
        {
            "name": record.meta.name,
            "trace": record.trace,
        }
        for record in batch_records
    ]
    return (
        prompt.rstrip()
        + "\n\n下面是待判断的 trace 样本：\n"
        + json.dumps(samples[0] if len(samples) == 1 else samples, ensure_ascii=False, indent=2)
    )


def main(argv: list[Any] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)

    if args.batch < 1:
        print("--batch must be greater than or equal to 1.")
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

    prompt = load_prompt(args.prompt_file)
    records = load_trace_records(input_dir, metadata)
    predictions: list[Prediction] = []

    for batch_records in chunks(records, args.batch):
        sample_names = [record.meta.name for record in batch_records]
        query = build_query(prompt, batch_records)
        raw_result = llm_generate(query)
        parsed_labels, parse_method = parse_llm_result(str(raw_result or ""), sample_names)

        for record in batch_records:
            predicted_label = parsed_labels.get(record.meta.name, BAD_LABEL)
            predictions.append(
                Prediction(
                    name=record.meta.name,
                    source=record.meta.source,
                    split=record.meta.split,
                    actual_label=record.meta.label,
                    predicted_label=predicted_label,
                    detail={
                        "parse_method": parse_method if record.meta.name in parsed_labels else "parse_error",
                        "raw_result": str(raw_result or ""),
                    },
                )
            )

    print_predictions(predictions)
    print_metrics_summary(predictions)
    if args.output:
        write_predictions_tsv(predictions, Path(args.output))
    write_metrics_markdown(predictions, Path(args.metrics_output), "LLM Trace Classification")
    return 0


if __name__ == "__main__":
    # Set INLINE_ARGS to run from an editor without command-line arguments.
    # Example:
    # INLINE_ARGS = [r".\traces", r".\metadata.csv", "--split", "test", "--batch", 1]
    INLINE_ARGS: list[Any] | None = None
    raise SystemExit(main(INLINE_ARGS))

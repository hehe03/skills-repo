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


HYBRID_AGENT_PROMPT = """你是一个严格的 Agent trace 审查员。你的任务是审查未被结构规则拦截的 trace 是否真正完成了用户请求。

请按以下流程分析：
1. 理解用户 query 的目标。
2. 检查 plan_list 是否覆盖该目标。
3. 检查关键 task 的 result 是否真实有效，而不只是非空。
4. 检查是否存在隐性 badcase：答非所问、结果过短、泛泛而谈、任务和结果不匹配、没有形成最终可用答案、执行路径偏离 query。
5. 检查是否存在 goodcase 证据：关键步骤完整、结果具体、最终产出能满足 query。
6. 如果证据不足以证明任务完成，倾向判 badcase。

注意：
- 连续重复和循环任务已经由前置规则处理；但如果你仍看到类似失败证据，也应判 badcase。
- 不要因为 task_name 合理或出现“生成大纲”就直接判 goodcase，必须检查 result 的质量和相关性。

输出要求：
- 只能输出 JSON，不要输出 Markdown。
- 如果只有一个样本，输出：
  {
    "label": "goodcase 或 badcase",
    "badcase_type": "none/irrelevant/low_quality/incomplete/missing_result/repetition/loop/other",
    "evidence": ["证据1", "证据2"],
    "confidence": 0.0到1.0
  }
- 如果有多个样本，输出以样本名为 key 的对象：
  {
    "case_001.json": {
      "label": "badcase",
      "badcase_type": "low_quality",
      "evidence": ["最终结果过短，无法满足 query"],
      "confidence": 0.86
    }
  }
"""


def llm_agent_generate(query: str) -> str:
    # TODO: Fill in your Agent LLM call here. Input is the complete query, output is raw model result.
    return ""


def parse_args(argv: list[Any] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid rule + Agent trace classifier for goodcase / badcase."
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
        default=2,
        help="Hard-rule threshold for consecutive repeated/loop tasks. Default: 2.",
    )
    parser.add_argument(
        "--disable-hard-rule",
        action="store_true",
        help="Disable repeated/loop hard rule and send every sample to Agent LLM.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Number of non-rule-hit samples per Agent query. Default: 1.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional UTF-8 prompt file. If omitted, uses the built-in hybrid Agent prompt.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV file for per-sample predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default="hybrid_metrics.md",
        help="Markdown metrics output path. Default: hybrid_metrics.md.",
    )
    return parser.parse_args([str(item) for item in argv] if argv is not None else None)


def load_prompt(prompt_file: str | None) -> str:
    if not prompt_file:
        return HYBRID_AGENT_PROMPT
    return Path(prompt_file).read_text(encoding="utf-8-sig")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def task_signature(task: dict[str, Any]) -> str:
    task_name = normalize_text(task.get("task_name"))
    command = task.get("command")
    command_name = ""
    command_args = ""
    if isinstance(command, dict):
        command_name = normalize_text(command.get("name"))
        command_args = normalize_text(command.get("args"))
    return f"{task_name}||{command_name}||{command_args}"


def detect_repeated_or_loop_tasks(trace: dict[str, Any], repeat_threshold: int) -> tuple[bool, str]:
    plan_list = trace.get("plan_list")
    if not isinstance(plan_list, list) or repeat_threshold < 2:
        return False, ""

    signatures = [
        task_signature(task)
        for task in plan_list
        if isinstance(task, dict)
    ]
    total = len(signatures)
    if total < repeat_threshold:
        return False, ""

    for index in range(0, total - repeat_threshold + 1):
        window = signatures[index : index + repeat_threshold]
        if len(set(window)) == 1:
            return True, f"连续重复任务：位置 {index + 1}-{index + repeat_threshold}"

    for window_size in range(1, total // repeat_threshold + 1):
        span = window_size * repeat_threshold
        for start in range(0, total - span + 1):
            pattern = signatures[start : start + window_size]
            if all(
                signatures[start + offset * window_size : start + (offset + 1) * window_size] == pattern
                for offset in range(1, repeat_threshold)
            ):
                return True, f"连续循环任务：pattern长度={window_size}，重复次数={repeat_threshold}"

    return False, ""


def normalize_agent_label(value: Any) -> str | None:
    if isinstance(value, dict) and "label" in value:
        value = value.get("label")
    text = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if text in {"good", "goodcase", "1", "true"}:
        return GOOD_LABEL
    if text in {"bad", "badcase", "0", "false"}:
        return BAD_LABEL
    if "badcase" in text:
        return BAD_LABEL
    if "goodcase" in text:
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

    return None


def normalize_result_item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        label = normalize_agent_label(value.get("label"))
        return {
            "label": label,
            "badcase_type": str(value.get("badcase_type") or "").strip(),
            "evidence": value.get("evidence") if isinstance(value.get("evidence"), list) else [],
            "confidence": value.get("confidence"),
        }

    label = normalize_agent_label(value)
    return {
        "label": label,
        "badcase_type": "",
        "evidence": [],
        "confidence": None,
    }


def parse_agent_result(raw_result: str, sample_names: list[str]) -> tuple[dict[str, dict[str, Any]], str]:
    json_text = extract_json_text(raw_result)
    if json_text:
        try:
            parsed = json.loads(json_text)
            results: dict[str, dict[str, Any]] = {}
            if isinstance(parsed, dict):
                if len(sample_names) == 1 and "label" in parsed:
                    item = normalize_result_item(parsed)
                    if item["label"]:
                        results[sample_names[0]] = item
                    return results, "json"

                for sample_name in sample_names:
                    if sample_name in parsed:
                        item = normalize_result_item(parsed[sample_name])
                        if item["label"]:
                            results[sample_name] = item
                if results:
                    return results, "json"
        except json.JSONDecodeError:
            pass

    label = normalize_agent_label(raw_result)
    if label and len(sample_names) == 1:
        return {
            sample_names[0]: {
                "label": label,
                "badcase_type": "",
                "evidence": [],
                "confidence": None,
            }
        }, "text"

    return {}, "parse_error"


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def print_agent_progress(batch_index: int, batch_records: list[Any], total: int) -> None:
    if total <= 1:
        return
    start = batch_index + 1
    end = batch_index + len(batch_records)
    names = ", ".join(record.meta.name for record in batch_records)
    print(f"[hybrid-agent] analyzing {start}-{end}/{total}: {names}", flush=True)


def build_agent_query(prompt: str, batch_records: list[Any]) -> str:
    samples = [
        {
            "name": record.meta.name,
            "trace": record.trace,
        }
        for record in batch_records
    ]
    return (
        prompt.rstrip()
        + "\n\n以下样本未命中前置循环/重复 hard rule，请做语义和质量审查：\n"
        + json.dumps(samples[0] if len(samples) == 1 else samples, ensure_ascii=False, indent=2)
    )


def main(argv: list[Any] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)

    if args.batch < 1:
        print("--batch must be greater than or equal to 1.")
        return 1
    if args.repeat_threshold < 2:
        print("--repeat-threshold must be greater than or equal to 2.")
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
    prompt = load_prompt(args.prompt_file)
    predictions: list[Prediction] = []
    agent_records: list[Any] = []

    for record in records:
        rule_hit = False
        rule_reason = ""
        if not args.disable_hard_rule:
            rule_hit, rule_reason = detect_repeated_or_loop_tasks(record.trace, args.repeat_threshold)

        if rule_hit:
            predictions.append(
                Prediction(
                    name=record.meta.name,
                    source=record.meta.source,
                    split=record.meta.split,
                    actual_label=record.meta.label,
                    predicted_label=BAD_LABEL,
                    detail={
                        "stage": "hard_rule",
                        "badcase_type": "repetition_or_loop",
                        "evidence": rule_reason,
                        "confidence": "1.0",
                        "raw_result": "",
                    },
                )
            )
        else:
            agent_records.append(record)

    print(
        f"[hybrid] hard-rule badcase={len(predictions)}, agent-review={len(agent_records)}, total={len(records)}",
        flush=True,
    )

    for batch_start, batch_records in enumerate(chunks(agent_records, args.batch), start=0):
        batch_index = batch_start * args.batch
        print_agent_progress(batch_index, batch_records, len(agent_records))
        sample_names = [record.meta.name for record in batch_records]
        raw_result = llm_agent_generate(build_agent_query(prompt, batch_records))
        parsed_results, parse_method = parse_agent_result(str(raw_result or ""), sample_names)

        for record in batch_records:
            result = parsed_results.get(record.meta.name)
            predicted_label = result["label"] if result else BAD_LABEL
            predictions.append(
                Prediction(
                    name=record.meta.name,
                    source=record.meta.source,
                    split=record.meta.split,
                    actual_label=record.meta.label,
                    predicted_label=predicted_label,
                    detail={
                        "stage": "agent",
                        "parse_method": parse_method if result else "parse_error",
                        "badcase_type": result.get("badcase_type", "") if result else "",
                        "confidence": result.get("confidence", "") if result else "",
                        "evidence": json.dumps(result.get("evidence", []), ensure_ascii=False) if result else "[]",
                        "raw_result": str(raw_result or ""),
                    },
                )
            )

    order = {record.meta.name: index for index, record in enumerate(records)}
    predictions.sort(key=lambda prediction: order[prediction.name])

    print_predictions(predictions)
    print_metrics_summary(predictions)
    if args.output:
        write_predictions_tsv(predictions, Path(args.output))
    write_metrics_markdown(predictions, Path(args.metrics_output), "Hybrid Rule + Agent Trace Classification")
    return 0


if __name__ == "__main__":
    # Set INLINE_ARGS to run from an editor without command-line arguments.
    # Example:
    # INLINE_ARGS = [r".\traces", r".\metadata.csv", "--split", "test", "--batch", 1]
    INLINE_ARGS: list[Any] | None = None
    raise SystemExit(main(INLINE_ARGS))

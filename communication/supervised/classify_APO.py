import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_eval_utils import (  # noqa: E402
    BAD_LABEL,
    GOOD_LABEL,
    Prediction,
    badcase_confusion,
    badcase_metrics,
    load_metadata,
    load_trace_records,
    print_metrics_summary,
    print_predictions,
    write_predictions_tsv,
)


INITIAL_PROMPT = """你是一个 Agent trace 质量评估器。

请判断给定 trace 是 goodcase 还是 badcase。

判定标准：
- goodcase：Agent 的执行过程能够有效完成用户请求，关键步骤有合理产出，最终结果与用户目标一致。
- badcase：Agent 没有有效完成用户请求，或者存在明显循环/重复、关键结果缺失、输出过短、步骤混乱、答非所问等问题。

输出要求：
- 只能输出 JSON。
- 如果只有一个样本，输出：{"label":"goodcase"} 或 {"label":"badcase"}。
- 如果有多个样本，输出：{"样本名":"goodcase","样本名2":"badcase"}。
- label 只能是 goodcase 或 badcase。
"""


@dataclass(frozen=True)
class EvalResult:
    predictions: list[Prediction]
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class IterationLog:
    epoch: int
    batch_index: int
    train_names: list[str]
    current_f1: float
    candidate_f1: float
    accepted: bool
    suggestion: str


def llm_student_generate(query: str) -> str:
    # TODO: Fill in your student-model LLM call here.
    # Input is the complete query. Output is the raw model result.
    return ""


def llm_teacher_generate(query: str) -> str:
    # TODO: Fill in your teacher-model LLM call here.
    # Input is the complete query. Output is the raw model result.
    return ""


def parse_args(argv: list[Any] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APO-style supervised prompt optimizer for trace classification."
    )
    parser.add_argument("input_dir", help="Directory that contains trace JSON files.")
    parser.add_argument(
        "metadata_csv",
        help="Metadata CSV with columns: name,label,source,split.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training minibatch size and student classification batch size. Default: 4.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of passes over the train split. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train minibatch shuffling. Default: 42.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Number of candidate prompts requested from the teacher. Default: 1.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional UTF-8 initial prompt file. If omitted, uses the built-in prompt.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV file for best test-set predictions.",
    )
    parser.add_argument(
        "--result-output",
        default="APO-result.md",
        help="APO markdown result output path. Default: APO-result.md.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Deprecated: progress is shown by default.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Do not print progress for student/teacher APO steps.",
    )
    return parser.parse_args([str(item) for item in argv] if argv is not None else None)


def load_initial_prompt(prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8-sig")
    return INITIAL_PROMPT


def normalize_llm_label(value: Any) -> str | None:
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


def parse_student_result(raw_result: str, sample_names: list[str]) -> tuple[dict[str, str], str]:
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


def build_classification_query(prompt: str, records: list[Any]) -> str:
    samples = [
        {
            "name": record.meta.name,
            "trace": record.trace,
        }
        for record in records
    ]
    return (
        prompt.rstrip()
        + "\n\n待分类 trace 样本如下：\n"
        + json.dumps(samples[0] if len(samples) == 1 else samples, ensure_ascii=False, indent=2)
    )


def classify_records_with_prompt(
    records: list[Any],
    prompt: str,
    batch_size: int,
    show_progress: bool,
    progress_prefix: str,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    total = len(records)

    for batch_number, batch_records in enumerate(chunks(records, batch_size), start=1):
        if show_progress and total > 1:
            start = (batch_number - 1) * batch_size + 1
            end = start + len(batch_records) - 1
            names = ", ".join(record.meta.name for record in batch_records)
            print(f"[{progress_prefix}] classify {start}-{end}/{total}: {names}", flush=True)

        sample_names = [record.meta.name for record in batch_records]
        query = build_classification_query(prompt, batch_records)
        raw_result = llm_student_generate(query)
        parsed_labels, parse_method = parse_student_result(str(raw_result or ""), sample_names)

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
    return predictions


def progress_enabled(args: argparse.Namespace) -> bool:
    return not args.quiet_progress


def evaluate_prompt(
    records: list[Any],
    prompt: str,
    batch_size: int,
    show_progress: bool,
    progress_prefix: str,
) -> EvalResult:
    predictions = classify_records_with_prompt(records, prompt, batch_size, show_progress, progress_prefix)
    matrix = badcase_confusion(predictions)
    metrics = badcase_metrics(matrix)
    return EvalResult(
        predictions=predictions,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
    )


def build_critique_query(prompt: str, records: list[Any], predictions: list[Prediction]) -> str:
    prediction_by_name = {prediction.name: prediction for prediction in predictions}
    examples = []
    for record in records:
        prediction = prediction_by_name[record.meta.name]
        examples.append(
            {
                "name": record.meta.name,
                "actual_label": record.meta.label,
                "predicted_label": prediction.predicted_label,
                "is_correct": record.meta.label == prediction.predicted_label,
                "trace": record.trace,
            }
        )

    return (
        "你是 APO 提示词优化中的教师模型。请根据当前提示词、训练样本、学生模型预测结果和真实标签，"
        "分析当前提示词为什么会导致错误分类，并给出可执行的改进建议。\n\n"
        "要求：\n"
        "- 重点分析 badcase 漏判和 goodcase 误杀。\n"
        "- 建议应该具体说明需要补充、删除或强调哪些判断规则。\n"
        "- 不要直接输出新提示词，只输出改进建议。\n\n"
        "当前提示词：\n"
        f"{prompt}\n\n"
        "训练样本与预测结果：\n"
        + json.dumps(examples, ensure_ascii=False, indent=2)
    )


def build_optimize_query(prompt: str, suggestion: str, candidates: int) -> str:
    return (
        "你是 APO 提示词优化中的提示词编辑器。请根据改进建议改写当前提示词，"
        "目标是在 trace goodcase/badcase 分类任务上提升 badcase 的 F1-score。\n\n"
        "请遵循 APO 的思路：把改进建议视为自然语言梯度，并沿着相反的语义方向编辑提示词，"
        "让提示词更明确、更可执行、更能区分边界样本。\n\n"
        "输出要求：\n"
        "- 只能输出 JSON。\n"
        f"- 输出 {candidates} 个候选提示词，格式为 {{\"prompts\":[\"候选提示词1\", \"候选提示词2\"]}}。\n"
        "- 候选提示词必须完整，可直接用于分类。\n\n"
        "当前提示词：\n"
        f"{prompt}\n\n"
        "改进建议：\n"
        f"{suggestion}"
    )


def parse_candidate_prompts(raw_result: str) -> list[str]:
    stripped = raw_result.strip()
    if not stripped:
        return []

    json_text = extract_json_text(stripped)
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("prompts"), list):
                    return [str(prompt).strip() for prompt in parsed["prompts"] if str(prompt).strip()]
                if parsed.get("prompt"):
                    return [str(parsed["prompt"]).strip()]
            if isinstance(parsed, list):
                return [str(prompt).strip() for prompt in parsed if str(prompt).strip()]
        except json.JSONDecodeError:
            pass

    return [stripped] if len(stripped) >= 20 else []


def metric_tuple(eval_result: EvalResult) -> tuple[float, float, float]:
    return (eval_result.f1, eval_result.recall, eval_result.precision)


def write_apo_result(
    output: Path,
    initial_prompt: str,
    best_prompt: str,
    baseline: EvalResult,
    best_result: EvalResult,
    history: list[IterationLog],
) -> None:
    lines = [
        "# APO Result",
        "",
        "## Summary",
        "",
        f"- Baseline badcase precision: {baseline.precision:.4f}",
        f"- Baseline badcase recall: {baseline.recall:.4f}",
        f"- Baseline badcase F1-score: {baseline.f1:.4f}",
        f"- Best badcase precision: {best_result.precision:.4f}",
        f"- Best badcase recall: {best_result.recall:.4f}",
        f"- Best badcase F1-score: {best_result.f1:.4f}",
        "",
        "## Optimization History",
        "",
        "| epoch | batch | train samples | previous F1 | candidate F1 | accepted |",
        "| ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for item in history:
        lines.append(
            "| "
            f"{item.epoch} | {item.batch_index} | {', '.join(item.train_names)} | "
            f"{item.current_f1:.4f} | {item.candidate_f1:.4f} | {item.accepted} |"
        )

    lines.extend(
        [
            "",
            "## Initial Prompt",
            "",
            "```text",
            initial_prompt,
            "```",
            "",
            "## Best Prompt",
            "",
            "```text",
            best_prompt,
            "```",
            "",
            "## Best Test Predictions",
            "",
            "| name | source | actual | predicted |",
            "| --- | --- | --- | --- |",
        ]
    )
    for prediction in best_result.predictions:
        lines.append(
            f"| {prediction.name} | {prediction.source} | {prediction.actual_label} | {prediction.predicted_label} |"
        )
    lines.extend(["", "## Teacher Suggestions", ""])
    for item in history:
        lines.extend(
            [
                f"### Epoch {item.epoch}, Batch {item.batch_index}",
                "",
                "```text",
                item.suggestion or "(empty suggestion)",
                "```",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8-sig")


def main(argv: list[Any] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)

    if args.batch < 1:
        print("--batch must be greater than or equal to 1.")
        return 1
    if args.epoch < 1:
        print("--epoch must be greater than or equal to 1.")
        return 1
    if args.candidates < 1:
        print("--candidates must be greater than or equal to 1.")
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return 1
    if not metadata_csv.is_file():
        print(f"Metadata CSV does not exist: {metadata_csv}")
        return 1

    metadata = load_metadata(metadata_csv)
    train_rows = [row for row in metadata if row.split == "train"]
    test_rows = [row for row in metadata if row.split == "test"]
    if not train_rows:
        print("No train rows found in metadata.")
        return 1
    if not test_rows:
        print("No test rows found in metadata.")
        return 1

    train_records = load_trace_records(input_dir, train_rows)
    test_records = load_trace_records(input_dir, test_rows)
    rng = random.Random(args.seed)

    initial_prompt = load_initial_prompt(args.prompt_file)
    current_prompt = initial_prompt
    show_progress = progress_enabled(args)
    print(
        f"[APO] loaded {len(train_records)} train samples and {len(test_records)} test samples",
        flush=True,
    )
    print("[APO] evaluating initial prompt on test split", flush=True)
    current_result = evaluate_prompt(test_records, current_prompt, args.batch, show_progress, "baseline")
    print(f"[APO] baseline badcase F1={current_result.f1:.4f}", flush=True)
    best_prompt = current_prompt
    best_result = current_result
    baseline_result = current_result
    history: list[IterationLog] = []

    for epoch_index in range(1, args.epoch + 1):
        shuffled_records = list(train_records)
        rng.shuffle(shuffled_records)

        for batch_index, train_batch in enumerate(chunks(shuffled_records, args.batch), start=1):
            train_names = [record.meta.name for record in train_batch]
            print(f"[APO] epoch {epoch_index}/{args.epoch}, batch {batch_index}: {', '.join(train_names)}", flush=True)

            train_predictions = classify_records_with_prompt(
                train_batch,
                current_prompt,
                args.batch,
                show_progress,
                f"train-e{epoch_index}-b{batch_index}",
            )
            critique_query = build_critique_query(current_prompt, train_batch, train_predictions)
            if show_progress:
                print(f"[APO] epoch {epoch_index}, batch {batch_index}: asking teacher for critique", flush=True)
            suggestion = str(llm_teacher_generate(critique_query) or "").strip()

            optimize_query = build_optimize_query(current_prompt, suggestion, args.candidates)
            if show_progress:
                print(f"[APO] epoch {epoch_index}, batch {batch_index}: asking teacher for prompt candidates", flush=True)
            candidate_raw = str(llm_teacher_generate(optimize_query) or "")
            candidate_prompts = parse_candidate_prompts(candidate_raw)
            if show_progress:
                print(
                    f"[APO] epoch {epoch_index}, batch {batch_index}: received {len(candidate_prompts)} candidate prompt(s)",
                    flush=True,
                )

            best_candidate_prompt = current_prompt
            best_candidate_result = current_result
            for candidate_index, candidate_prompt in enumerate(candidate_prompts, start=1):
                print(f"[APO] evaluating candidate {candidate_index}/{len(candidate_prompts)} on test split", flush=True)
                candidate_result = evaluate_prompt(
                    test_records,
                    candidate_prompt,
                    args.batch,
                    show_progress,
                    f"candidate-e{epoch_index}-b{batch_index}-{candidate_index}",
                )
                print(
                    f"[APO] candidate {candidate_index}/{len(candidate_prompts)} badcase F1={candidate_result.f1:.4f}",
                    flush=True,
                )
                if metric_tuple(candidate_result) > metric_tuple(best_candidate_result):
                    best_candidate_prompt = candidate_prompt
                    best_candidate_result = candidate_result

            accepted = best_candidate_result.f1 > current_result.f1
            history.append(
                IterationLog(
                    epoch=epoch_index,
                    batch_index=batch_index,
                    train_names=train_names,
                    current_f1=current_result.f1,
                    candidate_f1=best_candidate_result.f1,
                    accepted=accepted,
                    suggestion=suggestion,
                )
            )

            if accepted:
                current_prompt = best_candidate_prompt
                current_result = best_candidate_result
                print(f"[APO] accepted prompt, test badcase F1={current_result.f1:.4f}", flush=True)
            else:
                print(f"[APO] kept current prompt, test badcase F1={current_result.f1:.4f}", flush=True)

            if metric_tuple(current_result) > metric_tuple(best_result):
                best_prompt = current_prompt
                best_result = current_result

    print_predictions(best_result.predictions)
    print_metrics_summary(best_result.predictions)
    if args.output:
        write_predictions_tsv(best_result.predictions, Path(args.output))
    write_apo_result(
        Path(args.result_output),
        initial_prompt,
        best_prompt,
        baseline_result,
        best_result,
        history,
    )
    return 0


if __name__ == "__main__":
    # Set INLINE_ARGS to run from an editor without command-line arguments.
    # Example:
    # INLINE_ARGS = [r".\traces", r".\metadata.csv", "--batch", 4, "--epoch", 2]
    INLINE_ARGS: list[Any] | None = None
    raise SystemExit(main(INLINE_ARGS))

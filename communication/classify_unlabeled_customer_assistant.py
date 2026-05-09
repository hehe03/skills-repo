import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from trace_eval_utils import (
    BAD_LABEL,
    GOOD_LABEL,
    Prediction,
    badcase_confusion,
    badcase_metrics,
    print_metrics_summary,
    print_predictions,
    write_predictions_tsv,
)


UNKNOWN_LABEL = "unknown"
ALL_SPLIT = "all"

DECISION_BAD_HIGH = "badcase_high_confidence"
DECISION_GOOD_LIKELY = "goodcase_likely"
DECISION_NEEDS_REVIEW = "needs_review"

ENTRY_CLARIFICATION = "clarification_entry"
ENTRY_RECOMMENDATION = "recommendation_entry"
ENTRY_OUTLINE = "outline_entry"
ENTRY_UNKNOWN = "unknown_entry"

FAILURE_MARKERS = [
    "error",
    "failed",
    "exception",
    "traceback",
    "timeout",
    "permission denied",
    "not found",
    "null",
    "none",
    "失败",
    "报错",
    "异常",
    "超时",
    "无权限",
    "未找到",
    "无法获取",
    "无结果",
    "没有结果",
    "未达目标",
    "未完成",
    "为空",
]

CONFIRMATION_MARKERS = [
    "确认",
    "请确认",
    "是否",
    "要不要",
    "是否继续",
    "是否进入",
    "下一任务",
    "下一步",
    "用户同意",
    "用户确认",
    "等待用户",
    "询问用户",
]

OUTLINE_MARKERS = ["大纲", "交流大纲", "生成大纲", "输出大纲", "ppt大纲", "PPT大纲"]

RECOMMENDATION_MARKERS = [
    "要素推荐",
    "历史交流",
    "交流话题",
    "客户痛点",
    "痛点话题",
    "相似客户",
    "客户案例",
    "案例推荐",
    "合作信息",
    "客户合作",
    "补充与确认",
]

CLARIFICATION_MARKERS = [
    "澄清",
    "要素澄清",
    "要素检查",
    "检查要素",
    "完备性",
    "关键要素",
    "补充信息",
]

DIRECT_OUTLINE_QUERY_MARKERS = [
    "直接生成大纲",
    "直接生成交流大纲",
    "直接出大纲",
    "帮我写大纲",
    "输出交流大纲",
    "生成交流大纲",
    "生成大纲",
]

DIRECT_RECOMMENDATION_QUERY_MARKERS = [
    "要素推荐",
    "推荐要素",
    "补充话题",
    "历史交流",
    "客户痛点",
    "相似客户",
    "客户案例",
    "合作信息",
]


@dataclass(frozen=True)
class FlexibleMetadataRow:
    name: str
    label: str
    source: str
    split: str


@dataclass(frozen=True)
class FlexibleTraceRecord:
    meta: FlexibleMetadataRow
    trace: dict[str, Any]


@dataclass(frozen=True)
class TraceStep:
    index: int
    task_name: str
    command_name: str
    command_args_text: str
    result_text: str
    combined_text: str
    has_command: bool
    result_empty: bool
    result_failed: bool
    is_confirmation: bool
    is_outline: bool
    is_recommendation: bool
    is_clarification: bool


@dataclass(frozen=True)
class TraceSummary:
    query: str
    plan_list_exists: bool
    tasks: list[TraceStep]
    final_result: str
    entry_mode: str
    outline_steps: list[int]
    recommendation_steps: list[int]
    clarification_steps: list[int]
    failed_tool_steps: list[int]
    empty_result_steps: list[int]
    confirmation_steps: list[int]
    repeat_tool_runs: list[list[int]]
    max_loop_repeats: int
    query_final_overlap: float
    final_has_outline_structure: bool


@dataclass(frozen=True)
class ClassificationResult:
    decision_label: str
    predicted_label: str
    confidence: float
    risk_score: float
    risk_factors: list[str]
    evidence: list[str]
    decision_stage: str
    failure_summary: str
    auto_failure_cluster: str
    entry_mode: str
    risk_breakdown: dict[str, float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-precision unlabeled trace classifier for customer communication assistant traces."
    )
    parser.add_argument("input_dir", help="Directory that contains trace JSON files.")
    parser.add_argument(
        "metadata_csv",
        help="Metadata CSV. Compatible with name,label,source,split; label/source/split are optional.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Use only this split as evaluation set. Default: all samples.",
    )
    parser.add_argument(
        "--output",
        help="Optional TSV file for per-sample predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default="unlabeled_customer_assistant_metrics.md",
        help="Markdown report output path. Default: unlabeled_customer_assistant_metrics.md.",
    )
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=0.85,
        help="Risk threshold for high-confidence badcase when at least one hard rule is hit. Default: 0.85.",
    )
    parser.add_argument(
        "--review-risk-threshold",
        type=float,
        default=0.70,
        help="Risk threshold for needs_review when no hard rule is hit. Default: 0.70.",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=3,
        help="Consecutive repeated tool calls required for direct high-confidence badcase. Default: 3.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip labeled precision/recall metrics even when labels are present.",
    )
    return parser.parse_args(argv)


def normalize_label(value: str | None) -> str:
    if value is None:
        return UNKNOWN_LABEL
    cleaned = value.strip().lower().replace(" ", "").replace("_", "")
    if not cleaned:
        return UNKNOWN_LABEL
    if cleaned in {"good", "goodcase", "1", "true"}:
        return GOOD_LABEL
    if cleaned in {"bad", "badcase", "0", "false"}:
        return BAD_LABEL
    return UNKNOWN_LABEL


def normalize_split(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    if cleaned in {"train", "test"}:
        return cleaned
    return ALL_SPLIT


def load_flexible_metadata(path: Path) -> list[FlexibleMetadataRow]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if "name" not in fieldnames:
            raise ValueError("Metadata CSV missing required column: name")

        rows: list[FlexibleMetadataRow] = []
        for index, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                raise ValueError(f"Metadata row {index} has empty name.")
            rows.append(
                FlexibleMetadataRow(
                    name=name,
                    label=normalize_label(row.get("label")),
                    source=(row.get("source") or "").strip() or "unknown",
                    split=normalize_split(row.get("split")),
                )
            )
    return rows


def filter_metadata_by_split(
    rows: Iterable[FlexibleMetadataRow], split: str | None
) -> list[FlexibleMetadataRow]:
    if split is None:
        return list(rows)
    return [row for row in rows if row.split == split]


def load_json_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("trace JSON root must be an object")
    return data


def load_flexible_trace_records(
    trace_dir: Path, rows: Iterable[FlexibleMetadataRow]
) -> list[FlexibleTraceRecord]:
    records: list[FlexibleTraceRecord] = []
    for row in rows:
        trace_path = trace_dir / row.name
        if not trace_path.is_file():
            raise FileNotFoundError(f"Trace JSON not found for metadata name: {row.name}")
        records.append(FlexibleTraceRecord(meta=row, trace=load_json_trace(trace_path)))
    return records


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def contains_any(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def tokenize(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    chinese_chars = set(re.findall(r"[\u4e00-\u9fff]", lowered))
    return words | chinese_chars


def token_overlap(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def compact_text(text: str, limit: int = 80) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def command_signature(step: TraceStep) -> str:
    return f"{step.command_name}||{step.command_args_text}"


def task_signature(step: TraceStep) -> str:
    return f"{step.task_name}||{step.command_name}||{step.command_args_text}"


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


def max_loop_repeats(signatures: list[str]) -> int:
    total = len(signatures)
    best = 1 if total else 0
    for window_size in range(1, total // 2 + 1):
        for start in range(0, total - window_size + 1):
            pattern = signatures[start : start + window_size]
            repeats = 1
            cursor = start + window_size
            while (
                cursor + window_size <= total
                and signatures[cursor : cursor + window_size] == pattern
            ):
                repeats += 1
                cursor += window_size
            best = max(best, repeats)
    return best


def build_step(index: int, task: dict[str, Any]) -> TraceStep:
    task_name = normalize_text(task.get("task_name"))
    command = task.get("command")
    command_name = ""
    command_args_text = ""
    has_command = isinstance(command, dict)
    if isinstance(command, dict):
        command_name = normalize_text(command.get("name"))
        command_args_text = normalize_text(command.get("args"))

    result_text = normalize_text(task.get("result"))
    combined_text = " ".join([task_name, command_name, command_args_text, result_text])
    result_empty = not result_text.strip()
    result_failed = result_empty or contains_any(result_text, FAILURE_MARKERS)
    is_confirmation = contains_any(combined_text, CONFIRMATION_MARKERS)
    is_outline = contains_any(combined_text, OUTLINE_MARKERS)
    is_recommendation = contains_any(combined_text, RECOMMENDATION_MARKERS)
    is_clarification = contains_any(combined_text, CLARIFICATION_MARKERS)

    return TraceStep(
        index=index,
        task_name=task_name,
        command_name=command_name,
        command_args_text=command_args_text,
        result_text=result_text,
        combined_text=combined_text,
        has_command=has_command,
        result_empty=result_empty,
        result_failed=result_failed,
        is_confirmation=is_confirmation,
        is_outline=is_outline,
        is_recommendation=is_recommendation,
        is_clarification=is_clarification,
    )


def infer_entry_mode(query: str) -> str:
    if contains_any(query, DIRECT_OUTLINE_QUERY_MARKERS):
        return ENTRY_OUTLINE
    if contains_any(query, DIRECT_RECOMMENDATION_QUERY_MARKERS):
        return ENTRY_RECOMMENDATION

    stripped = query.strip()
    if not stripped:
        return ENTRY_UNKNOWN
    if re.search(r"(file|文件|报告|id|ID|客户|公司)", stripped) and len(stripped) <= 80:
        return ENTRY_CLARIFICATION
    if len(stripped) <= 40:
        return ENTRY_CLARIFICATION
    return ENTRY_UNKNOWN


def repeated_tool_runs(steps: list[TraceStep]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous_signature = ""

    for step in steps:
        signature = command_signature(step) if step.has_command and step.command_name else ""
        if signature and signature == previous_signature:
            if not current:
                current = [step.index - 1]
            current.append(step.index)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
        previous_signature = signature

    if len(current) >= 2:
        runs.append(current)
    return runs


def has_later_confirmation(steps: list[TraceStep], failed_index: int, next_index: int) -> bool:
    return any(step.is_confirmation for step in steps if failed_index < step.index < next_index)


def build_summary(trace: dict[str, Any]) -> TraceSummary:
    query = normalize_text(trace.get("query"))
    plan_list = trace.get("plan_list")
    plan_list_exists = isinstance(plan_list, list)
    raw_tasks = plan_list if isinstance(plan_list, list) else []
    tasks = [
        build_step(index, task)
        for index, task in enumerate(raw_tasks, start=1)
        if isinstance(task, dict)
    ]

    final_result = tasks[-1].result_text if tasks else normalize_text(trace.get("result"))
    outline_steps = [step.index for step in tasks if step.is_outline]
    recommendation_steps = [step.index for step in tasks if step.is_recommendation]
    clarification_steps = [step.index for step in tasks if step.is_clarification]
    failed_tool_steps = [
        step.index for step in tasks if step.has_command and step.result_failed
    ]
    empty_result_steps = [step.index for step in tasks if step.result_empty]
    confirmation_steps = [step.index for step in tasks if step.is_confirmation]
    signatures = [task_signature(step) for step in tasks]

    return TraceSummary(
        query=query,
        plan_list_exists=plan_list_exists,
        tasks=tasks,
        final_result=final_result,
        entry_mode=infer_entry_mode(query),
        outline_steps=outline_steps,
        recommendation_steps=recommendation_steps,
        clarification_steps=clarification_steps,
        failed_tool_steps=failed_tool_steps,
        empty_result_steps=empty_result_steps,
        confirmation_steps=confirmation_steps,
        repeat_tool_runs=repeated_tool_runs(tasks),
        max_loop_repeats=max_loop_repeats(signatures),
        query_final_overlap=token_overlap(query, final_result),
        final_has_outline_structure=has_outline_structure(final_result),
    )


def has_outline_structure(text: str) -> bool:
    if not text.strip():
        return False
    heading_markers = [
        r"\n\s*[一二三四五六七八九十]+[、.．]",
        r"\n\s*\d+[、.．)]",
        r"\n\s*[-*•]",
        r"第[一二三四五六七八九十\d]+部分",
        r"章节",
        r"要点",
        r"议程",
    ]
    return any(re.search(pattern, text) for pattern in heading_markers)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def score_summary(summary: TraceSummary) -> dict[str, float]:
    tasks = summary.tasks
    task_count = len(tasks)
    result_nonempty_ratio = (
        sum(1 for step in tasks if not step.result_empty) / task_count if task_count else 0.0
    )
    max_repeat_run = max((len(run) for run in summary.repeat_tool_runs), default=0)
    failed_continue_count = count_tool_failure_continuations(summary)

    schema_risk = 0.0
    if not summary.plan_list_exists:
        schema_risk = 1.0
    elif not tasks and not summary.final_result.strip():
        schema_risk = 0.95
    elif not summary.query.strip():
        schema_risk = 0.45

    tool_risk = 0.0
    tool_risk += 0.60 if failed_continue_count else 0.0
    tool_risk += 0.35 if max_repeat_run >= 3 else 0.20 if max_repeat_run >= 2 else 0.0
    tool_risk += 0.20 if summary.max_loop_repeats >= 2 else 0.0
    tool_risk = clamp(tool_risk)

    result_risk = 1.0 - result_nonempty_ratio
    if summary.final_result.strip():
        final_len = len(summary.final_result)
        if final_len < 20:
            result_risk = max(result_risk, 0.70)
        elif final_len < 80:
            result_risk = max(result_risk, 0.45)
    else:
        result_risk = max(result_risk, 0.90)

    flow_risk = compute_flow_risk(summary)
    semantic_risk = 0.0
    if summary.final_result.strip() and summary.query.strip():
        semantic_risk = 1.0 - clamp(summary.query_final_overlap / 0.18)
    elif not summary.final_result.strip():
        semantic_risk = 0.80

    confirmation_risk = 0.0
    if failed_continue_count:
        confirmation_risk = 1.0
    elif summary.entry_mode in {ENTRY_CLARIFICATION, ENTRY_RECOMMENDATION}:
        if not summary.confirmation_steps and summary.recommendation_steps:
            confirmation_risk = 0.35

    return {
        "schema_risk": clamp(schema_risk),
        "flow_risk": clamp(flow_risk),
        "tool_risk": clamp(tool_risk),
        "result_risk": clamp(result_risk),
        "semantic_risk": clamp(semantic_risk),
        "confirmation_risk": clamp(confirmation_risk),
    }


def weighted_risk_score(risks: dict[str, float]) -> float:
    return clamp(
        0.25 * risks["tool_risk"]
        + 0.20 * risks["flow_risk"]
        + 0.20 * risks["result_risk"]
        + 0.15 * risks["semantic_risk"]
        + 0.10 * risks["confirmation_risk"]
        + 0.10 * risks["schema_risk"]
    )


def count_tool_failure_continuations(summary: TraceSummary) -> int:
    count = 0
    steps = summary.tasks
    for position, step in enumerate(steps[:-1]):
        next_step = steps[position + 1]
        if not step.has_command or not step.result_failed or not next_step.has_command:
            continue
        if not has_later_confirmation(steps, step.index, next_step.index):
            count += 1
    return count


def compute_flow_risk(summary: TraceSummary) -> float:
    if not summary.tasks:
        return 0.80

    risk = 0.0
    if summary.entry_mode == ENTRY_CLARIFICATION:
        if summary.outline_steps and not summary.clarification_steps:
            risk = max(risk, 0.85)
        if summary.recommendation_steps and not summary.confirmation_steps:
            risk = max(risk, 0.35)
    elif summary.entry_mode == ENTRY_RECOMMENDATION:
        if summary.outline_steps and not summary.recommendation_steps:
            risk = max(risk, 0.85)
    elif summary.entry_mode == ENTRY_OUTLINE:
        if not summary.outline_steps and not contains_any(summary.final_result, OUTLINE_MARKERS):
            risk = max(risk, 0.55)
    else:
        risk = max(risk, 0.20)

    if summary.outline_steps and not summary.final_result.strip():
        risk = max(risk, 0.75)
    return clamp(risk)


def hard_rule_evidence(
    summary: TraceSummary, repeat_threshold: int
) -> tuple[list[str], list[str], str]:
    risk_factors: list[str] = []
    evidence: list[str] = []
    decision_stage = "risk_score"

    if not summary.plan_list_exists:
        risk_factors.append("missing_plan_list")
        evidence.append("plan_list 缺失或不是 list")
        decision_stage = "schema_rule"
    elif not summary.tasks and not summary.final_result.strip():
        risk_factors.append("empty_plan_list")
        evidence.append("plan_list 为空且没有最终结果")
        decision_stage = "schema_rule"
    elif summary.tasks and len(summary.empty_result_steps) == len(summary.tasks):
        risk_factors.append("all_results_empty")
        evidence.append("所有 task result 均为空")
        decision_stage = "schema_rule"

    for position, step in enumerate(summary.tasks[:-1]):
        next_step = summary.tasks[position + 1]
        if (
            step.has_command
            and step.result_failed
            and next_step.has_command
            and not has_later_confirmation(summary.tasks, step.index, next_step.index)
        ):
            risk_factors.append("tool_failure_continue_without_user_confirmation")
            evidence.append(
                f"第{step.index}步工具未达目标，第{next_step.index}步直接调用新工具，未检测到用户确认"
            )
            decision_stage = "tool_compliance_rule"
            break

    max_repeat_run = max((len(run) for run in summary.repeat_tool_runs), default=0)
    if max_repeat_run >= repeat_threshold:
        run = max(summary.repeat_tool_runs, key=len)
        risk_factors.append("self_repeated_tool_call")
        evidence.append(
            f"连续{len(run)}次调用同一工具且参数相同，步骤={','.join(str(item) for item in run)}"
        )
        decision_stage = "tool_compliance_rule"

    if summary.entry_mode == ENTRY_CLARIFICATION:
        if summary.outline_steps and not summary.clarification_steps:
            risk_factors.append("clarification_skipped_before_outline")
            evidence.append("用户输入倾向需要要素澄清，但 trace 未出现澄清或要素检查即进入大纲生成")
            decision_stage = "flow_rule"
    elif summary.entry_mode == ENTRY_RECOMMENDATION:
        if summary.outline_steps and not summary.recommendation_steps:
            risk_factors.append("recommendation_skipped_before_outline")
            evidence.append("用户直接要求要素推荐，但 trace 未出现要素推荐任务即进入大纲生成")
            decision_stage = "flow_rule"

    if summary.outline_steps and not summary.final_result.strip():
        risk_factors.append("outline_missing_final_result")
        evidence.append("trace 已进入大纲生成阶段，但最终结果为空")
        decision_stage = "result_rule"

    if summary.final_result.strip() and contains_any(summary.final_result, FAILURE_MARKERS):
        risk_factors.append("final_result_is_failure")
        evidence.append("最终结果包含失败或异常信号")
        decision_stage = "result_rule"

    return risk_factors, evidence, decision_stage


def review_risk_factors(summary: TraceSummary, risks: dict[str, float]) -> tuple[list[str], list[str]]:
    factors: list[str] = []
    evidence: list[str] = []

    if summary.entry_mode == ENTRY_UNKNOWN:
        factors.append("unknown_entry_mode")
        evidence.append("无法稳定识别用户入口意图")
    if summary.max_loop_repeats >= 2:
        factors.append("loop_pattern")
        evidence.append(f"检测到循环任务模式，最大重复轮数={summary.max_loop_repeats}")
    if summary.repeat_tool_runs:
        max_run = max(summary.repeat_tool_runs, key=len)
        factors.append("repeated_tool_call_risk")
        evidence.append(f"存在重复工具调用风险，步骤={','.join(str(item) for item in max_run)}")
    if summary.final_result.strip() and not summary.final_has_outline_structure and summary.outline_steps:
        factors.append("weak_outline_structure")
        evidence.append("已进入大纲生成阶段，但最终结果缺少明显大纲结构")
    if risks["semantic_risk"] >= 0.70 and summary.final_result.strip():
        factors.append("low_query_final_overlap")
        evidence.append("用户输入与最终结果的关键词重合较低")
    if risks["result_risk"] >= 0.70:
        factors.append("weak_or_missing_result")
        evidence.append("结果为空、过短或非空比例偏低")
    return factors, evidence


def choose_failure_cluster(risk_factors: list[str], decision_label: str) -> str:
    if decision_label == DECISION_GOOD_LIKELY:
        return "cluster_00: 未发现明显失败模式"
    if "tool_failure_continue_without_user_confirmation" in risk_factors:
        return "cluster_01: 工具失败后继续推进流程"
    if "self_repeated_tool_call" in risk_factors or "repeated_tool_call_risk" in risk_factors:
        return "cluster_02: 重复调用同一工具"
    if "clarification_skipped_before_outline" in risk_factors:
        return "cluster_03: 未经要素澄清直接生成大纲"
    if "recommendation_skipped_before_outline" in risk_factors:
        return "cluster_04: 要素推荐缺失但进入大纲生成"
    if "outline_missing_final_result" in risk_factors or "weak_outline_structure" in risk_factors:
        return "cluster_05: 大纲生成结果缺失或结构不足"
    if "low_query_final_overlap" in risk_factors:
        return "cluster_06: 最终结果与用户目标弱相关"
    if "missing_plan_list" in risk_factors or "empty_plan_list" in risk_factors:
        return "cluster_07: trace 结构不可用"
    return "cluster_99: 其它高风险模式"


def hard_rule_score_floor(risk_factors: list[str]) -> float:
    floors = {
        "missing_plan_list": 0.95,
        "empty_plan_list": 0.93,
        "all_results_empty": 0.92,
        "tool_failure_continue_without_user_confirmation": 0.95,
        "self_repeated_tool_call": 0.92,
        "clarification_skipped_before_outline": 0.88,
        "recommendation_skipped_before_outline": 0.88,
        "outline_missing_final_result": 0.90,
        "final_result_is_failure": 0.90,
    }
    return max((floors.get(factor, 0.0) for factor in risk_factors), default=0.0)


def build_failure_summary(summary: TraceSummary, evidence: list[str]) -> str:
    flow = []
    if summary.clarification_steps:
        flow.append("要素澄清")
    if summary.recommendation_steps:
        flow.append("要素推荐")
    if summary.outline_steps:
        flow.append("大纲生成")
    if not flow:
        flow.append("未识别到关键业务阶段")
    final_preview = compact_text(summary.final_result, 60) if summary.final_result else "无最终结果"
    evidence_preview = "；".join(evidence[:3]) if evidence else "未发现强证据"
    return (
        f"入口={summary.entry_mode}；实际阶段={' -> '.join(flow)}；"
        f"最终结果={final_preview}；证据={evidence_preview}"
    )


def classify_summary(
    summary: TraceSummary,
    high_risk_threshold: float,
    review_risk_threshold: float,
    repeat_threshold: int,
) -> ClassificationResult:
    risks = score_summary(summary)
    risk_score = weighted_risk_score(risks)
    hard_factors, hard_evidence, decision_stage = hard_rule_evidence(
        summary, repeat_threshold
    )
    risk_score = max(risk_score, hard_rule_score_floor(hard_factors))
    review_factors, review_evidence = review_risk_factors(summary, risks)

    if hard_factors and risk_score >= high_risk_threshold:
        decision_label = DECISION_BAD_HIGH
        predicted_label = BAD_LABEL
        confidence = clamp(0.80 + 0.20 * risk_score)
        risk_factors = hard_factors + [
            factor for factor in review_factors if factor not in hard_factors
        ]
        evidence = hard_evidence + [
            item for item in review_evidence if item not in hard_evidence
        ]
    elif hard_factors:
        decision_label = DECISION_NEEDS_REVIEW
        predicted_label = GOOD_LABEL
        confidence = clamp(0.50 + 0.30 * risk_score)
        risk_factors = hard_factors + [
            factor for factor in review_factors if factor not in hard_factors
        ]
        evidence = hard_evidence + [
            item for item in review_evidence if item not in hard_evidence
        ]
        decision_stage = "hard_rule_below_high_risk_threshold"
    elif risk_score >= review_risk_threshold or review_factors:
        decision_label = DECISION_NEEDS_REVIEW
        predicted_label = GOOD_LABEL
        confidence = clamp(0.45 + 0.35 * risk_score)
        risk_factors = review_factors or ["risk_score_review"]
        evidence = review_evidence or [f"风险分达到复核阈值：{risk_score:.3f}"]
        decision_stage = "risk_score_review"
    else:
        decision_label = DECISION_GOOD_LIKELY
        predicted_label = GOOD_LABEL
        confidence = clamp(1.0 - risk_score)
        risk_factors = []
        evidence = ["未命中高置信 badcase 规则，风险分低于复核阈值"]
        decision_stage = "low_risk"

    return ClassificationResult(
        decision_label=decision_label,
        predicted_label=predicted_label,
        confidence=confidence,
        risk_score=risk_score,
        risk_factors=risk_factors,
        evidence=evidence,
        decision_stage=decision_stage,
        failure_summary=build_failure_summary(summary, evidence),
        auto_failure_cluster=choose_failure_cluster(risk_factors, decision_label),
        entry_mode=summary.entry_mode,
        risk_breakdown=risks,
    )


def has_labels(predictions: list[Prediction]) -> bool:
    return all(prediction.actual_label in {GOOD_LABEL, BAD_LABEL} for prediction in predictions)


def build_unlabeled_report(predictions: list[Prediction], title: str) -> str:
    label_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    factor_counts: dict[str, int] = {}
    for prediction in predictions:
        decision_label = str(prediction.detail.get("decision_label", ""))
        label_counts[decision_label] = label_counts.get(decision_label, 0) + 1
        cluster = str(prediction.detail.get("auto_failure_cluster", ""))
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        factors = [
            factor.strip()
            for factor in str(prediction.detail.get("risk_factors", "")).split("|")
            if factor.strip()
        ]
        for factor in factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    total = len(predictions)
    lines = [
        f"# {title}",
        "",
        f"- 样本数：{total}",
        f"- high-confidence badcase：{label_counts.get(DECISION_BAD_HIGH, 0)}",
        f"- needs_review：{label_counts.get(DECISION_NEEDS_REVIEW, 0)}",
        f"- likely goodcase：{label_counts.get(DECISION_GOOD_LIKELY, 0)}",
        f"- high-confidence coverage：{safe_ratio(label_counts.get(DECISION_BAD_HIGH, 0), total):.4f}",
        f"- review coverage：{safe_ratio(label_counts.get(DECISION_NEEDS_REVIEW, 0), total):.4f}",
        "",
        "## 自动失败模式分布",
        "",
        "| cluster | count |",
        "| --- | ---: |",
    ]
    for cluster, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {cluster or 'N/A'} | {count} |")

    lines.extend(["", "## 风险因子分布", "", "| risk_factor | count |", "| --- | ---: |"])
    for factor, count in sorted(factor_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {factor} | {count} |")

    bad_candidates = [
        prediction
        for prediction in predictions
        if prediction.detail.get("decision_label") == DECISION_BAD_HIGH
    ][:20]
    lines.extend(["", "## 高置信 badcase 样本", "", "| name | cluster | evidence |", "| --- | --- | --- |"])
    for prediction in bad_candidates:
        lines.append(
            "| "
            f"{prediction.name} | "
            f"{prediction.detail.get('auto_failure_cluster', '')} | "
            f"{prediction.detail.get('evidence', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def append_labeled_metrics(report: str, predictions: list[Prediction]) -> str:
    matrix = badcase_confusion(predictions)
    metrics = badcase_metrics(matrix)
    lines = [
        report.rstrip(),
        "",
        "## 有标注评估",
        "",
        f"- badcase precision：{metrics['precision']:.4f}",
        f"- badcase recall：{metrics['recall']:.4f}",
        f"- badcase F1：{metrics['f1']:.4f}",
        "",
        "|  | 预测 badcase | 预测 goodcase |",
        "| --- | ---: | ---: |",
        f"| 实际 badcase | {matrix['tp']} | {matrix['fn']} |",
        f"| 实际 goodcase | {matrix['fp']} | {matrix['tn']} |",
        "",
    ]
    return "\n".join(lines)


def print_unlabeled_summary(predictions: list[Prediction]) -> None:
    counts: dict[str, int] = {}
    for prediction in predictions:
        decision_label = str(prediction.detail.get("decision_label", ""))
        counts[decision_label] = counts.get(decision_label, 0) + 1
    total = len(predictions)
    print("")
    print("Unlabeled decision summary:")
    for label in [DECISION_BAD_HIGH, DECISION_NEEDS_REVIEW, DECISION_GOOD_LIKELY]:
        count = counts.get(label, 0)
        print(f"{label}\t{count}/{total}\t{safe_ratio(count, total):.4f}")


def build_prediction(record: FlexibleTraceRecord, result: ClassificationResult) -> Prediction:
    return Prediction(
        name=record.meta.name,
        source=record.meta.source,
        split=record.meta.split,
        actual_label=record.meta.label,
        predicted_label=result.predicted_label,
        detail={
            "decision_label": result.decision_label,
            "confidence": f"{result.confidence:.3f}",
            "risk_score": f"{result.risk_score:.3f}",
            "entry_mode": result.entry_mode,
            "decision_stage": result.decision_stage,
            "risk_factors": "|".join(result.risk_factors),
            "evidence": "|".join(result.evidence),
            "failure_summary": result.failure_summary,
            "auto_failure_cluster": result.auto_failure_cluster,
            "schema_risk": f"{result.risk_breakdown['schema_risk']:.3f}",
            "flow_risk": f"{result.risk_breakdown['flow_risk']:.3f}",
            "tool_risk": f"{result.risk_breakdown['tool_risk']:.3f}",
            "result_risk": f"{result.risk_breakdown['result_risk']:.3f}",
            "semantic_risk": f"{result.risk_breakdown['semantic_risk']:.3f}",
            "confirmation_risk": f"{result.risk_breakdown['confirmation_risk']:.3f}",
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeat_threshold < 2:
        print("--repeat-threshold must be greater than or equal to 2.")
        return 1
    if args.review_risk_threshold > args.high_risk_threshold:
        print("--review-risk-threshold must be less than or equal to --high-risk-threshold.")
        return 1

    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return 1
    if not metadata_csv.is_file():
        print(f"Metadata CSV does not exist: {metadata_csv}")
        return 1

    metadata = filter_metadata_by_split(load_flexible_metadata(metadata_csv), args.split)
    if not metadata:
        print("No metadata rows selected.")
        return 0

    records = load_flexible_trace_records(input_dir, metadata)
    predictions: list[Prediction] = []
    for record in records:
        summary = build_summary(record.trace)
        result = classify_summary(
            summary,
            high_risk_threshold=args.high_risk_threshold,
            review_risk_threshold=args.review_risk_threshold,
            repeat_threshold=args.repeat_threshold,
        )
        predictions.append(build_prediction(record, result))

    print_predictions(predictions)
    print_unlabeled_summary(predictions)
    labeled = has_labels(predictions)
    if labeled and not args.no_metrics:
        print_metrics_summary(predictions)

    if args.output:
        write_predictions_tsv(predictions, Path(args.output))

    report = build_unlabeled_report(
        predictions, "Unlabeled Customer Assistant Trace Classification"
    )
    if labeled and not args.no_metrics:
        report = append_labeled_metrics(report, predictions)
    Path(args.metrics_output).write_text(report, encoding="utf-8-sig")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

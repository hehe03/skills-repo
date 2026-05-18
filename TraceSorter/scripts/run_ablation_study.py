from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from experiment_components.ablation_plans import build_ablation_variants
from experiment_components.contribution import summarize_result_components, summarize_rule_components
from experiment_components.rule_filters import component_counts, filter_rules
from llm_config import apply_llm_config, DEFAULT_LLM_CONFIG
from metrics import confusion_and_scores
from run_experiments import (
    DatasetBundle,
    discover_default_final_answer_config,
    load_final_answer_config,
    load_rules,
    parse_methods,
    predict_with_method,
    prepare_datasets,
    rule_paths_for_method,
    train_method,
)


FINAL_ANSWER_COMPONENTS = {
    "static_final_answer",
    "unlabeled_final_answer",
    "labeled_final_answer_diff",
}
DYNAMIC_FIELD_COMPONENTS = {
    "unlabeled_field_presence",
    "unlabeled_field_stats",
    "labeled_field_presence",
    "labeled_field_value",
    "labeled_field_numeric",
    "labeled_field_stats",
}
GENERATED_DYNAMIC_PREFIXES = ("unlabeled_", "labeled_", "llm_")
NUMERIC_COMPONENTS = {
    "unlabeled_numeric_quantile",
    "unlabeled_behavior_quantile",
    "unlabeled_schema_profile",
    "unlabeled_field_stats",
    "labeled_numeric_diff",
    "labeled_behavior_diff",
    "labeled_schema_profile",
    "labeled_field_numeric",
    "labeled_field_stats",
}
FIELD_COMPONENTS = {
    "unlabeled_field_presence",
    "unlabeled_field_stats",
    "labeled_field_presence",
    "labeled_field_value",
    "labeled_field_numeric",
    "labeled_field_stats",
}


@dataclass
class StudyVariant:
    method: str
    name: str
    category: str
    rules: List[Dict[str, Any]]
    enabled_components: tuple[str, ...] = ()
    disabled_components: tuple[str, ...] = ()


@dataclass
class StudyResult:
    variant: StudyVariant
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    changed: int
    fixed_cases: List[str]
    broken_cases: List[str]
    false_positive_cases: List[str]


def _safe_report_path(output_dir: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"ablation_study_{timestamp}.md"


def _components_in_rules(rules: Iterable[Dict[str, Any]]) -> set[str]:
    return set(component_counts(rules))


def _only_components(raw_rules: List[Dict[str, Any]], components: set[str]) -> List[Dict[str, Any]]:
    available = _components_in_rules(raw_rules)
    selected = components & available
    if not selected:
        return []
    return filter_rules(raw_rules, enable_components=selected)


def _without_components(raw_rules: List[Dict[str, Any]], components: set[str]) -> List[Dict[str, Any]]:
    available = _components_in_rules(raw_rules)
    disabled = components & available
    if not disabled:
        return filter_rules(raw_rules)
    return filter_rules(raw_rules, disable_components=disabled)


def _dynamic_components(components: set[str]) -> set[str]:
    return {component for component in components if component.startswith(GENERATED_DYNAMIC_PREFIXES)}


def build_study_variants(method: str, raw_rules: List[Dict[str, Any]]) -> List[StudyVariant]:
    components = _components_in_rules(raw_rules)
    variants: List[StudyVariant] = [
        StudyVariant(method=method, name="baseline", category="baseline", rules=filter_rules(raw_rules))
    ]

    for item in build_ablation_variants(raw_rules, plan="leave_one_component_out")[1:]:
        variants.append(
            StudyVariant(
                method=method,
                name=item.name,
                category="leave_one_component_out",
                rules=item.rules,
                enabled_components=item.enabled_components,
                disabled_components=item.disabled_components,
            )
        )
    for item in build_ablation_variants(raw_rules, plan="only_one_component")[1:]:
        variants.append(
            StudyVariant(
                method=method,
                name=item.name,
                category="only_one_component",
                rules=item.rules,
                enabled_components=item.enabled_components,
                disabled_components=item.disabled_components,
            )
        )

    targeted: List[StudyVariant] = []
    static_components = {component for component in components if component.startswith("static_")}
    dynamic_components = _dynamic_components(components)
    targeted_specs = [
        ("static_only", "targeted_subset", _only_components(raw_rules, static_components), tuple(sorted(static_components)), ()),
        ("dynamic_only", "targeted_subset", _only_components(raw_rules, dynamic_components), tuple(sorted(dynamic_components)), ()),
        (
            "without_final_answer",
            "targeted_subset",
            _without_components(raw_rules, FINAL_ANSWER_COMPONENTS),
            (),
            tuple(sorted(FINAL_ANSWER_COMPONENTS & components)),
        ),
        (
            "without_dynamic_fields",
            "targeted_subset",
            _without_components(raw_rules, DYNAMIC_FIELD_COMPONENTS),
            (),
            tuple(sorted(DYNAMIC_FIELD_COMPONENTS & components)),
        ),
        (
            "without_generated_dynamic",
            "targeted_subset",
            _without_components(raw_rules, dynamic_components),
            (),
            tuple(sorted(dynamic_components)),
        ),
        ("numeric_only", "targeted_subset", _only_components(raw_rules, NUMERIC_COMPONENTS), tuple(sorted(NUMERIC_COMPONENTS & components)), ()),
        ("field_only", "targeted_subset", _only_components(raw_rules, FIELD_COMPONENTS), tuple(sorted(FIELD_COMPONENTS & components)), ()),
    ]
    for name, category, rules, enabled, disabled in targeted_specs:
        if not rules and name not in {"dynamic_only", "numeric_only", "field_only"}:
            continue
        if name != "baseline":
            targeted.append(
                StudyVariant(
                    method=method,
                    name=name,
                    category=category,
                    rules=rules,
                    enabled_components=enabled,
                    disabled_components=disabled,
                )
            )

    variants.extend(_dedupe_variants(targeted))
    return _dedupe_variants(variants)


def _variant_signature(variant: StudyVariant) -> tuple[str, tuple[str, ...]]:
    return variant.name, tuple(str(rule.get("id", "")) for rule in variant.rules)


def _dedupe_variants(variants: Sequence[StudyVariant]) -> List[StudyVariant]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: List[StudyVariant] = []
    for variant in variants:
        signature = _variant_signature(variant)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(variant)
    return result


def _row_correct(row: Dict[str, Any]) -> bool | None:
    label = row.get("label")
    if label not in {"goodcase", "badcase"}:
        return None
    return row.get("predicted_label") == label


def _case_names(rows: Iterable[Dict[str, Any]], *, false_positive: bool = False) -> List[str]:
    names: List[str] = []
    for row in rows:
        if false_positive:
            if row.get("label") == "goodcase" and row.get("predicted_label") == "badcase":
                names.append(str(row.get("name")))
    return names


def evaluate_variant(
    variant: StudyVariant,
    *,
    records: List[Any],
    final_answer_config: Any,
    args: argparse.Namespace,
    baseline_rows: List[Dict[str, Any]] | None = None,
    baseline_metrics: Dict[str, Any] | None = None,
) -> StudyResult:
    rows = predict_with_method(records, variant.rules, final_answer_config, args)
    metrics = confusion_and_scores(rows)
    baseline_rows = baseline_rows or rows
    baseline_metrics = baseline_metrics or metrics
    baseline_by_name = {row["name"]: row for row in baseline_rows}
    changed = 0
    fixed_cases: List[str] = []
    broken_cases: List[str] = []
    for row in rows:
        base = baseline_by_name.get(row["name"], {})
        if row.get("predicted_label") != base.get("predicted_label"):
            changed += 1
        base_correct = _row_correct(base)
        current_correct = _row_correct(row)
        if base_correct is False and current_correct is True:
            fixed_cases.append(str(row.get("name")))
        elif base_correct is True and current_correct is False:
            broken_cases.append(str(row.get("name")))
    return StudyResult(
        variant=variant,
        results=rows,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        changed=changed,
        fixed_cases=fixed_cases,
        broken_cases=broken_cases,
        false_positive_cases=_case_names(rows, false_positive=True),
    )


def _rank_key(result: StudyResult) -> tuple[float, float, float, float, int]:
    metrics = result.metrics
    return (
        float(metrics.get("precision", 0.0)),
        float(metrics.get("recall", 0.0)),
        float(metrics.get("f1", 0.0)),
        float(metrics.get("accuracy", 0.0)),
        -int(metrics.get("fp", 0)),
    )


def _delta(result: StudyResult, key: str) -> float:
    return round(float(result.metrics.get(key, 0.0)) - float(result.baseline_metrics.get(key, 0.0)), 4)


def _format_components(values: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in values) if values else ""


def _format_names(values: Sequence[str], limit: int) -> str:
    if not values:
        return ""
    shown = [f"`{name}`" for name in values[:limit]]
    if len(values) > limit:
        shown.append(f"... +{len(values) - limit}")
    return ", ".join(shown)


def _source_breakdown(rows: List[Dict[str, Any]]) -> List[tuple[str, Dict[str, Any]]]:
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        by_source.setdefault(source, []).append(row)
    if len(by_source) <= 1:
        return []
    return [(source, confusion_and_scores(source_rows)) for source, source_rows in sorted(by_source.items())]


def _recommendations(results: List[StudyResult], recall_drop_tolerance: float) -> List[StudyResult]:
    recommended: List[StudyResult] = []
    for result in results:
        if result.variant.name == "baseline":
            continue
        baseline = result.baseline_metrics
        metrics = result.metrics
        if metrics.get("count", 0) == 0:
            continue
        precision_ok = float(metrics["precision"]) >= float(baseline["precision"])
        recall_ok = float(metrics["recall"]) >= float(baseline["recall"]) - recall_drop_tolerance
        if precision_ok and recall_ok:
            recommended.append(result)
    return sorted(recommended, key=_rank_key, reverse=True)


def write_summary_report(
    output_path: str | Path,
    *,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    results: List[StudyResult],
    baseline_by_method: Dict[str, StudyResult],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=_rank_key, reverse=True)
    recommended = _recommendations(results, args.recall_drop_tolerance)

    lines: List[str] = [
        "# TraceSorter 消融实验总报告",
        "",
        "## 实验配置",
        "",
        f"- Trace path: `{args.trace_path}`",
        f"- Metadata: `{args.metadata}`" if args.metadata else "- Metadata: none",
        f"- Train source: `{bundle.train_source}`",
        f"- Train samples: {len(bundle.train_records)}",
        f"- Test source: `{bundle.test_source}`",
        f"- Test samples: {len(bundle.test_records)}",
        f"- Methods: `{args.methods}`",
        f"- 主指标: `precision(badcase)`",
        f"- 次级指标: `recall(badcase)`",
        f"- Recall drop tolerance: {args.recall_drop_tolerance}",
        "",
        "## 迭代实验步骤",
        "",
        "1. Baseline：运行完整规则集合，建立 precision/recall/fp/fn 对照。",
        "2. Leave-one-component-out：逐个禁用组件，观察 precision 是否上升、recall 是否明显下降。",
        "3. Only-one-component：逐个只保留单一组件，判断组件单独预测能力。",
        "4. Targeted subsets：运行 static-only、dynamic-only、without-final-answer、without-dynamic-fields 等少量关键组合。",
        "5. 推荐筛选：优先选择 precision 不低于 baseline，且 recall 下降不超过容忍范围的方案。",
        "",
    ]

    lines.extend(["## Baseline 指标", ""])
    lines.append("| method | rules | precision | recall | f1 | accuracy | tp | fp | tn | fn |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method, result in baseline_by_method.items():
        metrics = result.metrics
        lines.append(
            f"| `{method}` | {len(result.variant.rules)} | {metrics['precision']} | {metrics['recall']} | "
            f"{metrics['f1']} | {metrics['accuracy']} | {metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} |"
        )
    lines.append("")

    lines.extend(["## Precision 优先总榜", ""])
    lines.append(
        "| rank | method | variant | category | rules | precision | recall | f1 | accuracy | fp | fn | delta precision | delta recall | changed |"
    )
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, result in enumerate(ranked[: args.top_k], start=1):
        metrics = result.metrics
        lines.append(
            f"| {index} | `{result.variant.method}` | `{result.variant.name}` | {result.variant.category} | "
            f"{len(result.variant.rules)} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | "
            f"{metrics['accuracy']} | {metrics['fp']} | {metrics['fn']} | {_delta(result, 'precision')} | "
            f"{_delta(result, 'recall')} | {result.changed} |"
        )
    lines.append("")

    lines.extend(["## 推荐候选", ""])
    if recommended:
        lines.append("| method | variant | category | precision | recall | f1 | fp | fn | changed | reason |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for result in recommended[: args.top_k]:
            metrics = result.metrics
            reason = "precision 不低于 baseline，recall 未超过容忍下降范围"
            lines.append(
                f"| `{result.variant.method}` | `{result.variant.name}` | {result.variant.category} | "
                f"{metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['fp']} | "
                f"{metrics['fn']} | {result.changed} | {reason} |"
            )
    else:
        lines.append("没有找到满足默认推荐条件的候选。建议优先查看 precision 总榜和 false positive case。")
    lines.append("")

    lines.extend(["## 全量实验结果", ""])
    lines.append(
        "| method | variant | category | enabled components | disabled components | rules | precision | recall | f1 | accuracy | tp | fp | tn | fn | fixed | broken |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in sorted(results, key=lambda item: (item.variant.method, item.variant.category, item.variant.name)):
        metrics = result.metrics
        lines.append(
            f"| `{result.variant.method}` | `{result.variant.name}` | {result.variant.category} | "
            f"{_format_components(result.variant.enabled_components)} | {_format_components(result.variant.disabled_components)} | "
            f"{len(result.variant.rules)} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | "
            f"{metrics['accuracy']} | {metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} | "
            f"{len(result.fixed_cases)} | {len(result.broken_cases)} |"
        )
    lines.append("")

    lines.extend(["## Baseline 组件贡献", ""])
    for method, baseline in baseline_by_method.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| component | rules | bad rules | good rules | weight sum |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in summarize_rule_components(baseline.variant.rules):
            lines.append(
                f"| `{row['component']}` | {row['rules']} | {row['bad_rules']} | "
                f"{row['good_rules']} | {round(row['weight_sum'], 4)} |"
            )
        lines.append("")
        lines.append("| component | cases hit | hits | bad score | good score | correct cases | incorrect cases |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in summarize_result_components(baseline.results):
            lines.append(
                f"| `{row['component']}` | {row['cases']} | {row['hits']} | "
                f"{round(row['bad_score'], 4)} | {round(row['good_score'], 4)} | "
                f"{row['correct_cases']} | {row['incorrect_cases']} |"
            )
        lines.append("")

    lines.extend(["## Case 变化摘要", ""])
    interesting = [
        result
        for result in ranked
        if result.variant.name != "baseline" and (result.fixed_cases or result.broken_cases or result.false_positive_cases)
    ]
    if not interesting:
        lines.append("没有发现相对 baseline 的 case 变化，或测试样本没有标签。")
    else:
        lines.append("| method | variant | fixed cases | broken cases | false positive cases |")
        lines.append("|---|---|---|---|---|")
        for result in interesting[: args.top_k]:
            lines.append(
                f"| `{result.variant.method}` | `{result.variant.name}` | "
                f"{_format_names(result.fixed_cases, args.max_case_names)} | "
                f"{_format_names(result.broken_cases, args.max_case_names)} | "
                f"{_format_names(result.false_positive_cases, args.max_case_names)} |"
            )
    lines.append("")

    source_rows_written = False
    source_lines: List[str] = ["## Source 分组检查", ""]
    for result in recommended[: min(args.top_k, 5)]:
        breakdown = _source_breakdown(result.results)
        if not breakdown:
            continue
        source_rows_written = True
        source_lines.append(f"### {result.variant.method} / {result.variant.name}")
        source_lines.append("")
        source_lines.append("| source | count | precision | recall | f1 | accuracy | fp | fn |")
        source_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for source, metrics in breakdown:
            source_lines.append(
                f"| `{source}` | {metrics['count']} | {metrics['precision']} | {metrics['recall']} | "
                f"{metrics['f1']} | {metrics['accuracy']} | {metrics['fp']} | {metrics['fn']} |"
            )
        source_lines.append("")
    if source_rows_written:
        lines.extend(source_lines)

    lines.extend(
        [
            "## 使用建议",
            "",
            "- 如果某个组件在 leave-one-out 中被禁用后 precision 上升，优先检查它贡献的 false positive。",
            "- 如果某个组件 only-one-component precision 高但 recall 低，可以保留为高置信辅助组件。",
            "- 如果 dynamic field 相关组件只在单一 source 上有效，不建议直接进入默认方案。",
            "- 如果 final-answer 相关组件 precision 下降明显，应考虑只在用户显式指定 final-answer 字段时启用强证据。",
            "- 最终采用前，建议至少再跑一次跨 source 或新的 eval split 验证。",
            "",
        ]
    )

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_study(args: argparse.Namespace) -> Path:
    bundle = prepare_datasets(args)
    methods = parse_methods(args.methods, bundle)
    if not methods:
        raise ValueError("No methods selected.")

    final_answer_config = discover_default_final_answer_config(
        bundle.train_records or bundle.all_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )

    all_results: List[StudyResult] = []
    baseline_by_method: Dict[str, StudyResult] = {}
    for method in methods:
        train_method(method, bundle, final_answer_config, args)
        raw_rules = load_rules(rule_paths_for_method(method))
        variants = build_study_variants(method, raw_rules)
        baseline_variant = variants[0]
        baseline = evaluate_variant(
            baseline_variant,
            records=bundle.test_records,
            final_answer_config=final_answer_config,
            args=args,
        )
        baseline_by_method[method] = baseline
        all_results.append(baseline)
        for variant in variants[1:]:
            all_results.append(
                evaluate_variant(
                    variant,
                    records=bundle.test_records,
                    final_answer_config=final_answer_config,
                    args=args,
                    baseline_rows=baseline.results,
                    baseline_metrics=baseline.metrics,
                )
            )

    report_path = Path(args.report_output) if args.report_output else _safe_report_path(args.output_dir)
    return write_summary_report(
        report_path,
        args=args,
        bundle=bundle,
        results=all_results,
        baseline_by_method=baseline_by_method,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a precision-first component ablation study and aggregate useful results into one Markdown report."
    )
    parser.add_argument("trace_path", help="Test/all trace JSON file or directory.")
    parser.add_argument("--metadata", help="Metadata CSV with name,label,source,split. Labels are required for precision/recall.")
    parser.add_argument("--train-trace-path", help="Optional separate training trace JSON file or directory.")
    parser.add_argument("--train-metadata", help="Optional training metadata CSV. Defaults to --metadata.")
    parser.add_argument("--train-split", help="Split value used as training data.")
    parser.add_argument("--eval-split", help="Split value used as evaluation data.")
    parser.add_argument(
        "--methods",
        default="non_llm_labeled",
        help="Comma-separated methods to study. Supports run_experiments.py method names and aliases.",
    )
    parser.add_argument("--bad-threshold", type=float, default=0.60)
    parser.add_argument("--good-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="./results/ablation", help="Directory for the aggregate report.")
    parser.add_argument("--report-output", help="Explicit aggregate Markdown report path.")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum rows shown in ranked/recommendation sections.")
    parser.add_argument("--max-case-names", type=int, default=8, help="Maximum case names shown per case-change cell.")
    parser.add_argument(
        "--recall-drop-tolerance",
        type=float,
        default=0.05,
        help="Allowed recall drop when recommending a precision-first variant.",
    )
    parser.add_argument("--final-answer-config", help="Optional final answer detection config.")
    parser.add_argument(
        "--final-answer-item",
        action="append",
        help="Business-specific final answer key:value pattern. Use * as a wildcard. Can be repeated.",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG),
        help="YAML config for LLM provider, output paths, use_existing_rules, and prompt budget.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = apply_llm_config(build_parser().parse_args(argv))
    report_path = run_study(args)
    print(f"Wrote ablation study report: {report_path}")


if __name__ == "__main__":
    # You may write parameters here when running this file from an IDE.
    # Example:
    # SCRIPT_ARGS = [r".\traces", "--metadata", r".\metadata.csv", "--train-split", "train", "--eval-split", "test"]
    SCRIPT_ARGS = None
    main(SCRIPT_ARGS)

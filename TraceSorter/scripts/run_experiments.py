from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from features import discover_default_final_answer_config, extract_features, load_final_answer_config
from reporting import default_report_path, write_report
from rule_engine import classify_features, load_rules
from rule_generation import generate_labeled_rules, generate_unlabeled_rules
from trace_io import TraceRecord, load_records, records_with_labels, split_records
from metrics import confusion_and_scores


SCRIPT_DIR = Path(__file__).resolve().parent
GENERAL_RULES = SCRIPT_DIR / "rules" / "static" / "general_rules.json"
UNLABELED_RULES = SCRIPT_DIR / "rules" / "dynamic" / "unlabeled_rules.json"
LABELED_RULES = SCRIPT_DIR / "rules" / "dynamic" / "labeled_rules.json"
LLM_RULES = SCRIPT_DIR / "rules" / "dynamic" / "llm_rules.json"


def _has_both_labels(records: List[TraceRecord]) -> bool:
    labels = {record.label for record in records}
    return {"goodcase", "badcase"}.issubset(labels)


def _fit_records(records: List[TraceRecord]) -> List[TraceRecord]:
    train = split_records(records, "train")
    return train or records


def _eval_records(records: List[TraceRecord]) -> List[TraceRecord]:
    test = split_records(records, "test")
    return test or records


def maybe_generate_rules(args: argparse.Namespace, records: List[TraceRecord]) -> List[str]:
    mode = args.generate_dynamic_rules
    generated: List[str] = []
    fit_records = _fit_records(records)
    labeled_fit = records_with_labels(fit_records)
    final_answer_config = discover_default_final_answer_config(
        fit_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_keys),
    )

    do_unlabeled = mode in {"unlabeled", "both"} or (
        mode == "auto" and args.rule_layer in {"unlabeled", "all"} and len(fit_records) >= 3
    )
    do_labeled = mode in {"labeled", "both"} or (
        mode == "auto" and args.rule_layer in {"labeled", "all"} and _has_both_labels(labeled_fit)
    )

    if do_unlabeled:
        generate_unlabeled_rules(fit_records, UNLABELED_RULES, final_answer_config)
        generated.append(str(UNLABELED_RULES))
    if do_labeled:
        generate_labeled_rules(labeled_fit, LABELED_RULES, final_answer_config)
        generated.append(str(LABELED_RULES))
    return generated


def rule_paths_for_layer(layer: str) -> List[Path]:
    paths = [GENERAL_RULES]
    if layer in {"unlabeled", "all"}:
        paths.append(UNLABELED_RULES)
    if layer in {"labeled", "all"}:
        paths.append(LABELED_RULES)
    if layer in {"llm", "all"}:
        paths.append(LLM_RULES)
    return paths


def run_experiment(args: argparse.Namespace) -> Path:
    if args.metadata is None and (
        args.rule_layer == "labeled" or args.generate_dynamic_rules in {"labeled", "both"}
    ):
        raise ValueError("metadata is required for labeled/supervised rule generation or evaluation")
    records = load_records(args.trace_path, args.metadata)
    maybe_generate_rules(args, records)
    rules = load_rules(rule_paths_for_layer(args.rule_layer))
    eval_records = _eval_records(records)
    final_answer_config = discover_default_final_answer_config(
        records,
        load_final_answer_config(args.final_answer_config, args.final_answer_keys),
    )

    results: List[Dict[str, Any]] = []
    for record in eval_records:
        features = extract_features(record, final_answer_config)
        prediction = classify_features(
            features,
            rules,
            bad_threshold=args.bad_threshold,
            good_threshold=args.good_threshold,
            aggregation=args.aggregation,
        )
        results.append(
            {
                "name": record.name,
                "label": record.label,
                "source": record.source,
                "split": record.split,
                "has_final_answer": features["has_final_answer"],
                "final_answer_source": features["final_answer_source"],
                "final_answer_evidence_enabled": features["final_answer_evidence_enabled"],
                "final_answer_evidence_strength": features["final_answer_evidence_strength"],
                "final_answer_evidence_source": features["final_answer_evidence_source"],
                "final_answer_adopted_fields": features["final_answer_adopted_fields"],
                **prediction,
            }
        )

    method = f"rule_{args.rule_layer}_{args.aggregation}"
    output_path = Path(args.output) if args.output else default_report_path(args.output_dir, method)
    return write_report(
        output_path,
        method=method,
        trace_path=args.trace_path,
        metadata_path=args.metadata,
        rules=rules,
        results=results,
        max_rows=args.max_rows,
    )


def _predict_records(
    records: List[TraceRecord],
    rules: List[Dict[str, Any]],
    args: argparse.Namespace,
    aggregation: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    final_answer_config = discover_default_final_answer_config(
        records,
        load_final_answer_config(args.final_answer_config, args.final_answer_keys),
    )
    for record in records:
        features = extract_features(record, final_answer_config)
        prediction = classify_features(
            features,
            rules,
            bad_threshold=args.bad_threshold,
            good_threshold=args.good_threshold,
            aggregation=aggregation,
        )
        results.append(
            {
                "name": record.name,
                "label": record.label,
                "source": record.source,
                "split": record.split,
                "has_final_answer": features["has_final_answer"],
                "final_answer_source": features["final_answer_source"],
                "final_answer_evidence_enabled": features["final_answer_evidence_enabled"],
                "final_answer_evidence_strength": features["final_answer_evidence_strength"],
                "final_answer_evidence_source": features["final_answer_evidence_source"],
                "final_answer_adopted_fields": features["final_answer_adopted_fields"],
                **prediction,
            }
        )
    return results


def run_general_method_comparison(args: argparse.Namespace) -> Path:
    records = load_records(args.trace_path, args.metadata)
    eval_records = _eval_records(records)
    rules = load_rules([GENERAL_RULES])
    weighted = _predict_records(eval_records, rules, args, "weighted")
    capped = _predict_records(eval_records, rules, args, "group_capped")
    metrics = {
        "weighted": confusion_and_scores(weighted),
        "group_capped": confusion_and_scores(capped),
    }
    method = "rule_general_comparison"
    output_path = Path(args.output) if args.output else default_report_path(args.output_dir, method)
    lines: List[str] = [
        "# Trace Sorter 通用规则方法对比",
        "",
        f"- Trace path: `{args.trace_path}`",
        f"- Metadata: `{args.metadata}`" if args.metadata else "- Metadata: none",
        f"- Samples evaluated: {len(eval_records)}",
        f"- Rules loaded: {len(rules)}",
        f"- Bad threshold: {args.bad_threshold}",
        f"- Good threshold: {args.good_threshold}",
        "",
        "## 方法说明",
        "",
        "- `weighted`：命中的 badcase/goodcase 规则分别直接加权求和。",
        "- `group_capped`：先按规则组累加，再将每个组的贡献限制在 `group_cap` 内，降低相关规则重复计分的影响。",
        "",
    ]
    if metrics["weighted"]["count"]:
        lines.extend(
            [
                "## 指标对比",
                "",
                "| method | accuracy | precision(badcase) | recall(badcase) | f1(badcase) | tp | fp | tn | fn |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("weighted", "group_capped"):
            row = metrics[name]
            lines.append(
                f"| {name} | {row['accuracy']} | {row['precision']} | {row['recall']} | "
                f"{row['f1']} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} |"
            )
        lines.append("")
    else:
        lines.extend(["## 指标对比", "", "metadata 中没有可用标签，因此只输出预测差异。", ""])

    lines.extend(
        [
            "## 预测差异",
            "",
            "| name | label | weighted | capped | weighted_score | capped_score | final_answer_policy | final_answer_source | weighted_reason | capped_reason |",
            "|---|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    capped_by_name = {row["name"]: row for row in capped}
    shown = 0
    for row in weighted:
        other = capped_by_name[row["name"]]
        if row["predicted_label"] == other["predicted_label"] and args.only_differences:
            continue
        weighted_reason = str(row["reason"]).replace("|", "\\|")
        capped_reason = str(other["reason"]).replace("|", "\\|")
        final_policy = (
            f"{row['final_answer_evidence_source']}/"
            f"{row['final_answer_evidence_strength']}:"
            f"{row['final_answer_adopted_fields'] or 'none'}"
        )
        lines.append(
            f"| `{row['name']}` | {row.get('label') or ''} | {row['predicted_label']} | "
            f"{other['predicted_label']} | {row['bad_score']} | {other['bad_score']} | "
            f"{final_policy} | {row['final_answer_source'] or 'none'} | "
            f"{weighted_reason} | {capped_reason} |"
        )
        shown += 1
        if shown >= args.max_rows:
            break
    if shown == 0:
        lines.append("| none | | | | | | | | 两种方法预测完全一致，或未显示相同预测 | |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trace sorting experiments and write a Markdown report named by method and time."
    )
    parser.add_argument("trace_path", help="Trace JSON file or directory containing JSON trace files.")
    parser.add_argument(
        "--metadata",
        help="Optional metadata CSV with columns name,label,source,split. Required only for labeled/supervised methods.",
    )
    parser.add_argument(
        "--rule-layer",
        choices=["general", "unlabeled", "labeled", "llm", "all"],
        default="general",
        help="Rule layer to evaluate.",
    )
    parser.add_argument(
        "--generate-dynamic-rules",
        choices=["none", "unlabeled", "labeled", "both", "auto"],
        default="none",
        help="Generate dynamic rules before evaluation.",
    )
    parser.add_argument("--bad-threshold", type=float, default=0.60)
    parser.add_argument("--good-threshold", type=float, default=0.50)
    parser.add_argument(
        "--aggregation",
        choices=["weighted", "group_capped"],
        default="weighted",
        help="How to combine matched rule weights for a normal experiment.",
    )
    parser.add_argument(
        "--compare-general-methods",
        action="store_true",
        help="Compare weighted general rules with grouped+capped general rules in one Markdown report.",
    )
    parser.add_argument(
        "--only-differences",
        action="store_true",
        help="When comparing general methods, show only samples whose predictions differ.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for timestamped Markdown reports.")
    parser.add_argument("--output", help="Explicit Markdown output path. Overrides method+time naming.")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--final-answer-config",
        help="Optional JSON config for business-specific final answer detection.",
    )
    parser.add_argument(
        "--final-answer-keys",
        help="Comma-separated business-specific final answer keys added to top-level and nested detection.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report_path = run_general_method_comparison(args) if args.compare_general_methods else run_experiment(args)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    # You may write parameters here when running this file from an IDE.
    # Example:
    # SCRIPT_ARGS = [r".\traces", "--compare-general-methods", "--output-dir", r".\results"]
    SCRIPT_ARGS = None
    main(SCRIPT_ARGS)

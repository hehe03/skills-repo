from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from features import discover_default_final_answer_config, extract_features, load_final_answer_config
from metrics import confusion_and_scores
from reporting import default_report_path, write_report
from rule_engine import classify_features, load_rules
from rule_generation import generate_labeled_rules, generate_unlabeled_rules
from trace_io import TraceRecord, load_records, records_with_labels, split_records


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


def _eval_records(records: List[TraceRecord], eval_split: str | None = None) -> List[TraceRecord]:
    if eval_split:
        selected = split_records(records, eval_split)
        if not selected:
            raise ValueError(f"metadata split column has no samples for eval split: {eval_split}")
        return selected
    return records


def maybe_generate_rules(
    args: argparse.Namespace,
    records: List[TraceRecord],
    methods: List[str] | None = None,
) -> List[str]:
    mode = args.generate_dynamic_rules
    target_methods = methods or [args.rule_layer]
    generated: List[str] = []
    fit_records = _fit_records(records)
    labeled_fit = records_with_labels(fit_records)
    final_answer_config = discover_default_final_answer_config(
        fit_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )

    do_unlabeled = mode in {"unlabeled", "both"} or (
        mode == "auto" and any(method in {"unlabeled", "all"} for method in target_methods) and len(fit_records) >= 3
    )
    do_labeled = mode in {"labeled", "both"} or (
        mode == "auto" and any(method in {"labeled", "all"} for method in target_methods) and _has_both_labels(labeled_fit)
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


def parse_methods(value: str | None) -> List[str]:
    if not value:
        return []
    methods: List[str] = []
    for item in value.split(","):
        method = item.strip().lower()
        if not method:
            continue
        if method == "all":
            for candidate in ("general", "unlabeled", "labeled", "llm"):
                if candidate not in methods:
                    methods.append(candidate)
            continue
        if method not in {"general", "unlabeled", "labeled", "llm"}:
            raise ValueError(f"unsupported method: {method}")
        if method not in methods:
            methods.append(method)
    return methods


def validate_method_requirements(args: argparse.Namespace, methods: List[str]) -> None:
    if args.metadata is None and (
        "labeled" in methods or args.generate_dynamic_rules in {"labeled", "both"}
    ):
        raise ValueError("metadata is required for labeled/supervised rule generation or evaluation")


def predict_with_method(
    records: List[TraceRecord],
    rules: List[Dict[str, Any]],
    final_answer_config: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for record in records:
        features = extract_features(record, final_answer_config)
        prediction = classify_features(
            features,
            rules,
            bad_threshold=args.bad_threshold,
            good_threshold=args.good_threshold,
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


def run_experiment(args: argparse.Namespace) -> Path:
    validate_method_requirements(args, [args.rule_layer])
    records = load_records(args.trace_path, args.metadata)
    maybe_generate_rules(args, records, [args.rule_layer])
    rules = load_rules(rule_paths_for_layer(args.rule_layer))
    eval_records = _eval_records(records, args.eval_split)
    final_answer_config = discover_default_final_answer_config(
        records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )

    results = predict_with_method(eval_records, rules, final_answer_config, args)

    method = f"rule_{args.rule_layer}"
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


def run_methods_comparison(args: argparse.Namespace, methods: List[str]) -> Path:
    validate_method_requirements(args, methods)
    records = load_records(args.trace_path, args.metadata)
    maybe_generate_rules(args, records, methods)
    eval_records = _eval_records(records, args.eval_split)
    final_answer_config = discover_default_final_answer_config(
        records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )
    results_by_method: Dict[str, List[Dict[str, Any]]] = {}
    rules_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method in methods:
        rules = load_rules(rule_paths_for_layer(method))
        rules_by_method[method] = rules
        results_by_method[method] = predict_with_method(eval_records, rules, final_answer_config, args)

    output_path = Path(args.output) if args.output else default_report_path(args.output_dir, "methods_comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Trace Sorter Methods Comparison",
        "",
        f"- Trace path: `{args.trace_path}`",
        f"- Metadata: `{args.metadata}`" if args.metadata else "- Metadata: none",
        f"- Eval split: `{args.eval_split}`" if args.eval_split else "- Eval split: all samples",
        f"- Methods: {', '.join(methods)}",
        f"- Samples evaluated: {len(eval_records)}",
        "",
    ]
    first_results = next(iter(results_by_method.values()), [])
    policies: Dict[str, int] = {}
    for row in first_results:
        policy = (
            f"{row.get('final_answer_evidence_source', 'none')}/"
            f"{row.get('final_answer_evidence_strength', 'none')}:"
            f"{row.get('final_answer_adopted_fields') or 'none'}"
        )
        policies[policy] = policies.get(policy, 0) + 1
    lines.extend(["## Method Notes", "", "Final answer policy:", "| policy | samples |", "|---|---:|"])
    for policy, count in sorted(policies.items()):
        lines.append(f"| `{policy}` | {count} |")
    lines.append("")

    if any(row.get("label") in {"goodcase", "badcase"} for row in first_results):
        lines.extend(
            [
                "## Metrics",
                "",
                "| method | rules | accuracy | precision(badcase) | recall(badcase) | f1(badcase) | tp | fp | tn | fn |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            metrics = confusion_and_scores(results_by_method[method])
            lines.append(
                f"| {method} | {len(rules_by_method[method])} | {metrics['accuracy']} | "
                f"{metrics['precision']} | {metrics['recall']} | {metrics['f1']} | "
                f"{metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} |"
            )
        lines.append("")

    lines.extend(["## Predictions", ""])
    header = "| name | label | " + " | ".join(methods) + " |"
    separator = "|---|---|" + "|".join("---" for _ in methods) + "|"
    lines.append(header)
    lines.append(separator)
    by_method_name = {
        method: {row["name"]: row for row in rows}
        for method, rows in results_by_method.items()
    }
    for record in eval_records[: args.max_rows]:
        row_cells = [f"`{record.name}`", record.label or ""]
        for method in methods:
            prediction = by_method_name[method].get(record.name, {})
            row_cells.append(
                f"{prediction.get('predicted_label', '')} "
                f"(bad={prediction.get('bad_score', '')}, good={prediction.get('good_score', '')})"
            )
        lines.append("| " + " | ".join(row_cells) + " |")
    if len(eval_records) > args.max_rows:
        lines.append(f"| ... | ... | {' | '.join('...' for _ in methods)} |")
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
        help="Single rule layer to evaluate when --methods is not set.",
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated methods to compare in one report, e.g. general,unlabeled,llm or all.",
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
        "--eval-split",
        help="Optional metadata split value to evaluate. If omitted, all samples are evaluated.",
    )
    parser.add_argument("--output-dir", default="./results", help="Directory used for auto-named Markdown reports.")
    parser.add_argument("--output", help="Explicit Markdown output file path. Overrides --output-dir auto naming.")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--final-answer-config",
        help="Optional JSON config for business-specific final answer detection.",
    )
    parser.add_argument(
        "--final-answer-item",
        action="append",
        help="Business-specific final answer key:value pattern. Use * as a wildcard. Can be repeated.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    report_path = run_methods_comparison(args, methods) if methods else run_experiment(args)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    # You may write parameters here when running this file from an IDE.
    # Example:
    # SCRIPT_ARGS = [r".\traces", "--rule-layer", "general", "--output-dir", r".\results"]
    SCRIPT_ARGS = None
    main(SCRIPT_ARGS)

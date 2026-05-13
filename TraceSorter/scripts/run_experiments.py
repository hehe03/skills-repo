from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from features import extract_features
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


def _eval_records(records: List[TraceRecord]) -> List[TraceRecord]:
    test = split_records(records, "test")
    return test or records


def maybe_generate_rules(args: argparse.Namespace, records: List[TraceRecord]) -> List[str]:
    mode = args.generate_dynamic_rules
    generated: List[str] = []
    fit_records = _fit_records(records)
    labeled_fit = records_with_labels(fit_records)

    do_unlabeled = mode in {"unlabeled", "both"} or (
        mode == "auto" and args.rule_layer in {"unlabeled", "all"} and len(fit_records) >= 3
    )
    do_labeled = mode in {"labeled", "both"} or (
        mode == "auto" and args.rule_layer in {"labeled", "all"} and _has_both_labels(labeled_fit)
    )

    if do_unlabeled:
        generate_unlabeled_rules(fit_records, UNLABELED_RULES)
        generated.append(str(UNLABELED_RULES))
    if do_labeled:
        generate_labeled_rules(labeled_fit, LABELED_RULES)
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
    records = load_records(args.trace_path, args.metadata)
    maybe_generate_rules(args, records)
    rules = load_rules(rule_paths_for_layer(args.rule_layer))
    eval_records = _eval_records(records)

    results: List[Dict[str, Any]] = []
    for record in eval_records:
        features = extract_features(record)
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
                **prediction,
            }
        )

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trace sorting experiments and write a Markdown report named by method and time."
    )
    parser.add_argument("trace_path", help="Trace JSON file or directory containing JSON trace files.")
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata CSV with columns name,label,source,split. Labels may be empty for unlabeled runs.",
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
    parser.add_argument("--output-dir", default=".", help="Directory for timestamped Markdown reports.")
    parser.add_argument("--output", help="Explicit Markdown output path. Overrides method+time naming.")
    parser.add_argument("--max-rows", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_path = run_experiment(args)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

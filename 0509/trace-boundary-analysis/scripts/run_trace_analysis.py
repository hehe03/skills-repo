import argparse
import json
from pathlib import Path

import classify_rule
import classify_supervised
import classify_unsupervised
import classify_unsupervised_hybrid
from common import (
    BAD_LABEL,
    GOOD_LABEL,
    build_items,
    discover_trace_files,
    load_metadata,
    print_metrics,
    print_predictions,
    serialize_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained Agent trace boundary analysis entrypoint.")
    parser.add_argument("input_path", help="Single trace JSON file or a directory containing trace JSON files.")
    parser.add_argument("--metadata", help="Optional metadata CSV. Recommended columns: name,label,source,split.")
    parser.add_argument(
        "--strategy",
        choices=["auto", "rule", "unsupervised", "unsupervised_hybrid", "supervised"],
        default="auto",
        help="Classification strategy. Defaults to auto.",
    )
    parser.add_argument("--split", choices=["train", "test"], help="Only analyze one metadata split.")
    parser.add_argument(
        "--rule-layer",
        choices=["general", "trace_format", "domain_prior"],
        default="domain_prior",
        help="Rule knowledge layer. Layers are cumulative: general < trace_format < domain_prior.",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=None,
        help="Rule repeated/looped-task threshold. Defaults to rules.md.",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Unsupervised score threshold.")
    parser.add_argument("--bad-risk-threshold", type=float, default=0.45, help="Unsupervised bad-risk threshold.")
    parser.add_argument("--bad-risk-weight", type=float, default=0.60, help="Unsupervised bad-risk penalty weight.")
    parser.add_argument("--centrality-weight", type=float, default=0.20, help="Unsupervised batch centrality weight.")
    parser.add_argument("--hybrid-threshold", type=float, default=0.52, help="Hybrid good_score threshold.")
    parser.add_argument("--hybrid-bad-risk-threshold", type=float, default=0.62, help="Hybrid bad_score threshold.")
    parser.add_argument(
        "--good-margin",
        type=float,
        default=0.08,
        help="Minimum bad_score margin over good_score when hybrid predicts badcase.",
    )
    parser.add_argument("--supervised-threshold", type=float, default=0.85, help="Supervised goodcase threshold.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--error-analysis-output", help="Optional Markdown error-analysis output path.")
    return parser.parse_args(argv)


def resolve_strategy(strategy: str, items) -> str:
    if strategy != "auto":
        return strategy
    train_labels = {
        item.meta.label
        for item in items
        if item.meta.split == "train" and item.meta.label in {GOOD_LABEL, BAD_LABEL}
    }
    if train_labels == {GOOD_LABEL, BAD_LABEL}:
        return "supervised"
    if len(items) >= 3:
        return "unsupervised"
    return "rule"


def run_predictions(args: argparse.Namespace, items):
    strategy = resolve_strategy(args.strategy, items)
    if strategy == "rule":
        predictions = classify_rule.classify(items, args.repeat_threshold, args.rule_layer)
    elif strategy == "unsupervised":
        predictions = classify_unsupervised.classify(
            items,
            threshold=args.threshold,
            bad_risk_threshold=args.bad_risk_threshold,
            bad_risk_weight=args.bad_risk_weight,
            centrality_weight=args.centrality_weight,
        )
    elif strategy == "unsupervised_hybrid":
        predictions = classify_unsupervised_hybrid.classify(
            items,
            threshold=args.hybrid_threshold,
            bad_risk_threshold=args.hybrid_bad_risk_threshold,
            good_margin=args.good_margin,
            centrality_weight=args.centrality_weight,
        )
    elif strategy == "supervised":
        predictions = classify_supervised.classify(items, args.supervised_threshold)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return strategy, predictions


def write_error_analysis_markdown(predictions, strategy: str, output_path: Path) -> None:
    from datetime import datetime
    from common import badcase_confusion, badcase_metrics

    labeled = [
        prediction
        for prediction in predictions
        if prediction.actual_label in {GOOD_LABEL, BAD_LABEL}
    ]
    matrix = badcase_confusion(labeled)
    metrics = badcase_metrics(matrix)

    def rows_for(actual_label: str, predicted_label: str):
        return [
            prediction
            for prediction in labeled
            if prediction.actual_label == actual_label
            and prediction.predicted_label == predicted_label
        ]

    def append_section(lines: list[str], title: str, section_predictions) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| sample | source | split | predicted | actual | score | bad_risk | reason |",
                "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        if not section_predictions:
            lines.append("|  |  |  |  |  |  |  |  |")
        for prediction in section_predictions:
            detail = prediction.detail
            reason = str(detail.get("reason", "")).replace("|", "\\|").replace("\n", " ")
            score = detail.get("score", "")
            bad_risk = detail.get("bad_risk", detail.get("bad_score", ""))
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    prediction.name,
                    prediction.source,
                    prediction.split,
                    prediction.predicted_label,
                    prediction.actual_label,
                    score,
                    bad_risk,
                    reason,
                )
            )
        lines.append("")

    lines = [
        f"# {strategy} error analysis",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- precision: {metrics['precision']:.4f}",
        f"- recall: {metrics['recall']:.4f}",
        f"- f1: {metrics['f1']:.4f}",
        f"- tp: {matrix['tp']}",
        f"- fp: {matrix['fp']}",
        f"- fn: {matrix['fn']}",
        f"- tn: {matrix['tn']}",
        "",
    ]
    append_section(lines, "False positives", rows_for(GOOD_LABEL, BAD_LABEL))
    append_section(lines, "False negatives", rows_for(BAD_LABEL, GOOD_LABEL))
    append_section(lines, "True positives", rows_for(BAD_LABEL, BAD_LABEL))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = load_metadata(Path(args.metadata)) if args.metadata else None
    trace_files = discover_trace_files(Path(args.input_path))
    items = build_items(trace_files, metadata, args.split)
    if not items:
        print("No trace samples selected.")
        return 0

    strategy, predictions = run_predictions(args, items)
    print_predictions(predictions, strategy)
    print_metrics(predictions)

    if args.output:
        output = {
            "strategy": strategy,
            "predictions": serialize_predictions(predictions, strategy),
        }
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.error_analysis_output:
        write_error_analysis_markdown(predictions, strategy, Path(args.error_analysis_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

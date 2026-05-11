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
    badcase_confusion,
    badcase_metrics,
    build_items,
    discover_trace_files,
    load_metadata,
    print_metrics,
    print_predictions,
    serialize_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自包含 Agent trace 定界分析入口。")
    parser.add_argument("input_path", help="单个 trace JSON 文件或 trace JSON 文件夹。")
    parser.add_argument("--metadata", help="可选 metadata CSV，推荐列：name,label,source,split。")
    parser.add_argument(
        "--strategy",
        choices=["auto", "rule", "unsupervised", "unsupervised_hybrid", "supervised"],
        default="auto",
        help="分类方法，默认 auto。",
    )
    parser.add_argument("--split", choices=["train", "test"], help="只分析指定 split。")
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=None,
        help="规则法重复/循环阈值；不传时读取 rules.md。",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="无监督 score 阈值。")
    parser.add_argument("--bad-risk-threshold", type=float, default=0.55, help="无监督 bad-risk 阈值。")
    parser.add_argument("--bad-risk-weight", type=float, default=0.45, help="无监督 bad-risk 惩罚权重。")
    parser.add_argument("--centrality-weight", type=float, default=0.10, help="无监督批内中心性权重。")
    parser.add_argument("--hybrid-threshold", type=float, default=0.52, help="hybrid 方法 good_score 阈值。")
    parser.add_argument("--hybrid-bad-risk-threshold", type=float, default=0.62, help="hybrid 方法 bad_score 阈值。")
    parser.add_argument("--good-margin", type=float, default=0.08, help="hybrid 方法判 badcase 时要求 bad_score 超过 good_score 的最小边距。")
    parser.add_argument(
        "--sweep-hybrid",
        action="store_true",
        help="对 hybrid 方法做阈值扫描，输出多组 precision/recall。",
    )
    parser.add_argument(
        "--sweep-hybrid-thresholds",
        default="0.45,0.50,0.55,0.60",
        help="hybrid good_score 阈值扫描列表，逗号分隔。",
    )
    parser.add_argument(
        "--sweep-hybrid-bad-risk-thresholds",
        default="0.45,0.50,0.55,0.60,0.65",
        help="hybrid bad_score 阈值扫描列表，逗号分隔。",
    )
    parser.add_argument(
        "--sweep-good-margins",
        default="0.00,0.05,0.10,0.15",
        help="hybrid good_margin 扫描列表，逗号分隔。",
    )
    parser.add_argument("--supervised-threshold", type=float, default=0.85, help="监督法 goodcase 阈值。")
    parser.add_argument("--output", help="可选 JSON 输出路径。")
    return parser.parse_args(argv)


def parse_float_list(value: str) -> list[float]:
    numbers: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped:
            numbers.append(float(stripped))
    if not numbers:
        raise ValueError("阈值列表不能为空。")
    return numbers


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


def run_hybrid_sweep(args, items) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for threshold in parse_float_list(args.sweep_hybrid_thresholds):
        for bad_threshold in parse_float_list(args.sweep_hybrid_bad_risk_thresholds):
            for margin in parse_float_list(args.sweep_good_margins):
                predictions = classify_unsupervised_hybrid.classify(
                    items,
                    threshold=threshold,
                    bad_risk_threshold=bad_threshold,
                    good_margin=margin,
                    centrality_weight=args.centrality_weight,
                )
                matrix = badcase_confusion(predictions)
                metrics = badcase_metrics(matrix)
                predicted_bad = matrix["tp"] + matrix["fp"]
                rows.append(
                    {
                        "hybrid_threshold": threshold,
                        "hybrid_bad_risk_threshold": bad_threshold,
                        "good_margin": margin,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "tp": matrix["tp"],
                        "fp": matrix["fp"],
                        "fn": matrix["fn"],
                        "tn": matrix["tn"],
                        "predicted_bad": predicted_bad,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["precision"],
            row["recall"],
            row["f1"],
            -row["fp"],
        ),
        reverse=True,
    )
    return rows


def print_hybrid_sweep(rows: list[dict[str, float | int]]) -> None:
    print(
        "rank\thybrid_threshold\thybrid_bad_risk_threshold\tgood_margin\tprecision\trecall\tf1\ttp\tfp\tfn\ttn\tpredicted_bad"
    )
    for index, row in enumerate(rows, start=1):
        print(
            "{rank}\t{threshold:.2f}\t{bad_threshold:.2f}\t{margin:.2f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{tp}\t{fp}\t{fn}\t{tn}\t{predicted_bad}".format(
                rank=index,
                threshold=row["hybrid_threshold"],
                bad_threshold=row["hybrid_bad_risk_threshold"],
                margin=row["good_margin"],
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
                tn=row["tn"],
                predicted_bad=row["predicted_bad"],
            )
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = load_metadata(Path(args.metadata)) if args.metadata else None
    trace_files = discover_trace_files(Path(args.input_path))
    items = build_items(trace_files, metadata, args.split)
    if not items:
        print("No trace samples selected.")
        return 0

    if args.sweep_hybrid:
        rows = run_hybrid_sweep(args, items)
        print_hybrid_sweep(rows)
        if args.output:
            output = {
                "strategy": "unsupervised_hybrid_sweep",
                "rows": rows,
            }
            Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    strategy = resolve_strategy(args.strategy, items)
    if strategy == "rule":
        predictions = classify_rule.classify(items, args.repeat_threshold)
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

    print_predictions(predictions, strategy)
    print_metrics(predictions)

    if args.output:
        output = {
            "strategy": strategy,
            "predictions": serialize_predictions(predictions, strategy),
        }
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

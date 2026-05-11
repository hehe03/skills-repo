import argparse
import sys
import json
from datetime import datetime
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
    parser.add_argument("--sweep-rule", action="store_true", help="对 rule 方法做 repeat_threshold 扫描。")
    parser.add_argument("--sweep-unsupervised", action="store_true", help="对 unsupervised 方法做参数扫描。")
    parser.add_argument("--sweep-supervised", action="store_true", help="对 supervised 方法做 threshold 扫描。")
    parser.add_argument("--sweep-ensemble", action="store_true", help="对 rule + unsupervised 组合方法做参数扫描。")
    parser.add_argument("--sweep-all", action="store_true", help="依次扫描 rule、unsupervised、unsupervised_hybrid、supervised。")
    parser.add_argument(
        "--sweep-repeat-thresholds",
        default="2,3,4",
        help="rule repeat_threshold 扫描列表，逗号分隔。",
    )
    parser.add_argument(
        "--sweep-thresholds",
        default="0.45,0.50,0.55,0.60,0.65",
        help="unsupervised score 阈值扫描列表，逗号分隔。",
    )
    parser.add_argument(
        "--sweep-bad-risk-thresholds",
        default="0.45,0.50,0.55,0.60,0.65,0.70",
        help="unsupervised bad-risk 阈值扫描列表，逗号分隔。",
    )
    parser.add_argument(
        "--sweep-bad-risk-weights",
        default="0.30,0.45,0.60",
        help="unsupervised bad-risk 权重扫描列表，逗号分隔。",
    )
    parser.add_argument(
        "--sweep-centrality-weights",
        default="0.00,0.10,0.20",
        help="unsupervised/hybrid centrality 权重扫描列表，逗号分隔。",
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
    parser.add_argument(
        "--sweep-supervised-thresholds",
        default="0.50,0.65,0.75,0.85,0.90",
        help="supervised threshold 扫描列表，逗号分隔。",
    )
    parser.add_argument("--output", help="可选 JSON 输出路径。")
    parser.add_argument("--sweep-output-dir", default=".", help="sweep Markdown 结果输出目录，默认当前目录。")
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


def parse_int_list(value: str) -> list[int]:
    numbers: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped:
            numbers.append(int(stripped))
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


def has_supervised_training(items) -> bool:
    train_labels = {
        item.meta.label
        for item in items
        if item.meta.split == "train" and item.meta.label in {GOOD_LABEL, BAD_LABEL}
    }
    return train_labels == {GOOD_LABEL, BAD_LABEL}


def build_metric_row(strategy: str, params: dict[str, float | int], predictions) -> dict[str, float | int | str]:
    matrix = badcase_confusion(predictions)
    metrics = badcase_metrics(matrix)
    row: dict[str, float | int | str] = {
        "strategy": strategy,
        **params,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": matrix["tp"],
        "fp": matrix["fp"],
        "fn": matrix["fn"],
        "tn": matrix["tn"],
        "predicted_bad": matrix["tp"] + matrix["fp"],
    }
    return row


def sort_sweep_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
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


def run_rule_sweep(args, items) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for repeat_threshold in parse_int_list(args.sweep_repeat_thresholds):
        predictions = classify_rule.classify(items, repeat_threshold)
        rows.append(build_metric_row("rule", {"repeat_threshold": repeat_threshold}, predictions))
    return sort_sweep_rows(rows)


def run_unsupervised_sweep(args, items) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for threshold in parse_float_list(args.sweep_thresholds):
        for bad_threshold in parse_float_list(args.sweep_bad_risk_thresholds):
            for bad_weight in parse_float_list(args.sweep_bad_risk_weights):
                for centrality_weight in parse_float_list(args.sweep_centrality_weights):
                    predictions = classify_unsupervised.classify(
                        items,
                        threshold=threshold,
                        bad_risk_threshold=bad_threshold,
                        bad_risk_weight=bad_weight,
                        centrality_weight=centrality_weight,
                    )
                    rows.append(
                        build_metric_row(
                            "unsupervised",
                            {
                                "threshold": threshold,
                                "bad_risk_threshold": bad_threshold,
                                "bad_risk_weight": bad_weight,
                                "centrality_weight": centrality_weight,
                            },
                            predictions,
                        )
                    )
    return sort_sweep_rows(rows)


def run_hybrid_sweep(args, items) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for threshold in parse_float_list(args.sweep_hybrid_thresholds):
        for bad_threshold in parse_float_list(args.sweep_hybrid_bad_risk_thresholds):
            for margin in parse_float_list(args.sweep_good_margins):
                for centrality_weight in parse_float_list(args.sweep_centrality_weights):
                    predictions = classify_unsupervised_hybrid.classify(
                        items,
                        threshold=threshold,
                        bad_risk_threshold=bad_threshold,
                        good_margin=margin,
                        centrality_weight=centrality_weight,
                    )
                    rows.append(
                        build_metric_row(
                            "unsupervised_hybrid",
                            {
                                "hybrid_threshold": threshold,
                                "hybrid_bad_risk_threshold": bad_threshold,
                                "good_margin": margin,
                                "centrality_weight": centrality_weight,
                            },
                            predictions,
                        )
                    )
    return sort_sweep_rows(rows)


def run_supervised_sweep(args, items) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for threshold in parse_float_list(args.sweep_supervised_thresholds):
        predictions = classify_supervised.classify(items, threshold)
        rows.append(build_metric_row("supervised", {"supervised_threshold": threshold}, predictions))
    return sort_sweep_rows(rows)


def combine_predictions(rule_predictions, other_predictions, mode: str):
    combined = []
    other_by_name = {prediction.name: prediction for prediction in other_predictions}
    for rule_prediction in rule_predictions:
        other_prediction = other_by_name[rule_prediction.name]
        rule_bad = rule_prediction.predicted_label == BAD_LABEL
        other_bad = other_prediction.predicted_label == BAD_LABEL
        if mode == "or":
            predicted_label = BAD_LABEL if rule_bad or other_bad else GOOD_LABEL
        elif mode == "and":
            predicted_label = BAD_LABEL if rule_bad and other_bad else GOOD_LABEL
        else:
            raise ValueError(f"Unsupported ensemble mode: {mode}")
        reason = f"ensemble_{mode}: rule={rule_prediction.predicted_label}; unsupervised={other_prediction.predicted_label}"
        combined.append(
            type(rule_prediction)(
                name=rule_prediction.name,
                source=rule_prediction.source,
                split=rule_prediction.split,
                actual_label=rule_prediction.actual_label,
                predicted_label=predicted_label,
                detail={"reason": reason},
            )
        )
    return combined


def run_ensemble_sweep(args, items) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for repeat_threshold in parse_int_list(args.sweep_repeat_thresholds):
        rule_predictions = classify_rule.classify(items, repeat_threshold)
        for threshold in parse_float_list(args.sweep_thresholds):
            for bad_threshold in parse_float_list(args.sweep_bad_risk_thresholds):
                for bad_weight in parse_float_list(args.sweep_bad_risk_weights):
                    for centrality_weight in parse_float_list(args.sweep_centrality_weights):
                        unsup_predictions = classify_unsupervised.classify(
                            items,
                            threshold=threshold,
                            bad_risk_threshold=bad_threshold,
                            bad_risk_weight=bad_weight,
                            centrality_weight=centrality_weight,
                        )
                        for mode in ["or", "and"]:
                            predictions = combine_predictions(rule_predictions, unsup_predictions, mode)
                            rows.append(
                                build_metric_row(
                                    f"ensemble_rule_unsupervised_{mode}",
                                    {
                                        "repeat_threshold": repeat_threshold,
                                        "threshold": threshold,
                                        "bad_risk_threshold": bad_threshold,
                                        "bad_risk_weight": bad_weight,
                                        "centrality_weight": centrality_weight,
                                    },
                                    predictions,
                                )
                            )
    return sort_sweep_rows(rows)


def print_sweep(rows: list[dict[str, float | int | str]]) -> None:
    columns = build_sweep_columns(rows)
    print("\t".join(columns))
    for index, row in enumerate(rows, start=1):
        print("\t".join(format_sweep_row(row, columns, index)))


def build_sweep_columns(rows: list[dict[str, float | int | str]]) -> list[str]:
    param_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key
            not in {
                "strategy",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "tn",
                "predicted_bad",
            }
        }
    )
    return [
        "rank",
        "strategy",
        *param_keys,
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "predicted_bad",
    ]


def format_sweep_row(row: dict[str, float | int | str], columns: list[str], rank: int) -> list[str]:
    values: list[str] = []
    for column in columns:
        if column == "rank":
            values.append(str(rank))
            continue
        value = row.get(column, "")
        if isinstance(value, float):
            values.append(f"{value:.4f}")
        else:
            values.append(str(value))
    return values


def sweep_method_name(args) -> str:
    names: list[str] = []
    if args.sweep_all:
        return "all_sweep"
    if args.sweep_rule:
        names.append("rule")
    if args.sweep_unsupervised:
        names.append("unsupervised")
    if args.sweep_hybrid:
        names.append("unsupervised_hybrid")
    if args.sweep_supervised:
        names.append("supervised")
    if args.sweep_ensemble:
        names.append("ensemble")
    return "_".join(names) + "_sweep" if names else "sweep"


def write_sweep_markdown(
    rows: list[dict[str, float | int | str]],
    args,
    method_name: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.sweep_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method_name}_{timestamp}.md"
    columns = build_sweep_columns(rows)

    lines = [
        f"# {method_name} 参数扫描结果",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入路径：`{args.input_path}`",
        f"- metadata：`{args.metadata or ''}`",
        f"- split：`{args.split or 'all'}`",
        f"- 命令：`{' '.join(sys.argv)}`",
        "",
        "## 结果",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append("| " + " | ".join(format_sweep_row(row, columns, index)) + " |")

    lines.extend(
        [
            "",
            "## 选择建议",
            "",
            "- 优先看 precision 较高且 predicted_bad 不过低的行。",
            "- 如果 precision 很高但 recall 接近 0，通常说明参数过保守。",
            "- 如果 recall 高但 FP 明显增多，下一轮应提高 bad-risk 阈值或 margin。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = load_metadata(Path(args.metadata)) if args.metadata else None
    trace_files = discover_trace_files(Path(args.input_path))
    items = build_items(trace_files, metadata, args.split)
    if not items:
        print("No trace samples selected.")
        return 0

    if (
        args.sweep_all
        or args.sweep_rule
        or args.sweep_unsupervised
        or args.sweep_hybrid
        or args.sweep_supervised
        or args.sweep_ensemble
    ):
        rows: list[dict[str, float | int | str]] = []
        if args.sweep_all or args.sweep_rule:
            rows.extend(run_rule_sweep(args, items))
        if args.sweep_all or args.sweep_unsupervised:
            rows.extend(run_unsupervised_sweep(args, items))
        if args.sweep_all or args.sweep_hybrid:
            rows.extend(run_hybrid_sweep(args, items))
        if (args.sweep_all or args.sweep_supervised) and has_supervised_training(items):
            rows.extend(run_supervised_sweep(args, items))
        if args.sweep_all or args.sweep_ensemble:
            rows.extend(run_ensemble_sweep(args, items))
        rows = sort_sweep_rows(rows)
        print_sweep(rows)
        method_name = sweep_method_name(args)
        markdown_path = write_sweep_markdown(rows, args, method_name)
        print(f"\nSweep markdown written: {markdown_path}")
        if args.output:
            output = {
                "strategy": "sweep",
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

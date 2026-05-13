import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import classify_rule
import classify_supervised
import classify_unsupervised
import classify_unsupervised_hybrid
from common import (
    BAD_LABEL,
    GOOD_LABEL,
    Prediction,
    badcase_confusion,
    badcase_metrics,
    build_items,
    discover_trace_files,
    load_metadata,
)


DEFAULTS: dict[str, Any] = {
    "strategy": "auto",
    "split": None,
    "rule_layer": "domain_prior",
    "repeat_threshold": None,
    "threshold": 0.55,
    "bad_risk_threshold": 0.45,
    "bad_risk_weight": 0.60,
    "centrality_weight": 0.20,
    "hybrid_threshold": 0.52,
    "hybrid_bad_risk_threshold": 0.62,
    "good_margin": 0.08,
    "supervised_threshold": 0.85,
    "sweep_hybrid": False,
    "sweep_rule": False,
    "sweep_unsupervised": False,
    "sweep_supervised": False,
    "sweep_ensemble": False,
    "sweep_all": False,
    "sweep_rule_layers": "general,trace_format,domain_prior",
    "sweep_repeat_thresholds": "2,3,4",
    "sweep_thresholds": "0.45,0.50,0.55,0.60,0.65",
    "sweep_bad_risk_thresholds": "0.45,0.50,0.55,0.60,0.65,0.70",
    "sweep_bad_risk_weights": "0.30,0.45,0.60",
    "sweep_centrality_weights": "0.00,0.10,0.20",
    "sweep_hybrid_thresholds": "0.45,0.50,0.55,0.60",
    "sweep_hybrid_bad_risk_thresholds": "0.45,0.50,0.55,0.60,0.65",
    "sweep_good_margins": "0.00,0.05,0.10,0.15",
    "sweep_supervised_thresholds": "0.50,0.65,0.75,0.85,0.90",
    "rule_config": None,
    "output": None,
    "sweep_output_dir": ".",
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [parse_scalar(part) for part in value[1:-1].split(",") if part.strip()]
    if "," in value and not (value.startswith("'") or value.startswith('"')):
        return [parse_scalar(part) for part in value.split(",") if part.strip()]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        if indent == 0 and not value.strip():
            current_section = key.strip()
            data[current_section] = {}
            continue
        target = data
        if indent > 0 and current_section and isinstance(data.get(current_section), dict):
            target = data[current_section]
        target[key.strip()] = parse_scalar(value.strip())
    return data


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8-sig"))
    try:
        import yaml  # type: ignore
    except ImportError:
        return load_simple_yaml(config_path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    return loaded or {}


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(config)
    sweep = flattened.pop("sweep", None)
    if isinstance(sweep, dict):
        aliases = {
            "all": "sweep_all",
            "rule": "sweep_rule",
            "unsupervised": "sweep_unsupervised",
            "hybrid": "sweep_hybrid",
            "unsupervised_hybrid": "sweep_hybrid",
            "supervised": "sweep_supervised",
            "ensemble": "sweep_ensemble",
            "rule_layers": "sweep_rule_layers",
            "repeat_thresholds": "sweep_repeat_thresholds",
            "thresholds": "sweep_thresholds",
            "bad_risk_thresholds": "sweep_bad_risk_thresholds",
            "bad_risk_weights": "sweep_bad_risk_weights",
            "centrality_weights": "sweep_centrality_weights",
            "hybrid_thresholds": "sweep_hybrid_thresholds",
            "hybrid_bad_risk_thresholds": "sweep_hybrid_bad_risk_thresholds",
            "good_margins": "sweep_good_margins",
            "supervised_thresholds": "sweep_supervised_thresholds",
            "output_dir": "sweep_output_dir",
        }
        for key, value in sweep.items():
            flattened[aliases.get(key, key)] = value
    return flattened


def preparse_config(argv: list[str] | None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    args, _ = parser.parse_known_args(argv)
    return args.config


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trace-boundary comparison experiments and parameter sweeps.")
    parser.set_defaults(**defaults)
    parser.add_argument("input_path", nargs="?", help="Single trace JSON file or a directory containing trace JSON files.")
    parser.add_argument("--config", help="Optional YAML/JSON config file. CLI flags override config values.")
    parser.add_argument("--metadata", help="Optional metadata CSV. Recommended columns: name,label,source,split.")
    parser.add_argument("--split", choices=["train", "test"], help="Only analyze one metadata split.")
    parser.add_argument(
        "--strategy",
        choices=["auto", "rule", "unsupervised", "unsupervised_hybrid", "supervised"],
        help="Single-run strategy used when no sweep flag is enabled.",
    )
    parser.add_argument("--rule-layer", choices=["general", "trace_format", "domain_prior"])
    parser.add_argument("--repeat-threshold", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--bad-risk-threshold", type=float)
    parser.add_argument("--bad-risk-weight", type=float)
    parser.add_argument("--centrality-weight", type=float)
    parser.add_argument("--hybrid-threshold", type=float)
    parser.add_argument("--hybrid-bad-risk-threshold", type=float)
    parser.add_argument("--good-margin", type=float)
    parser.add_argument("--supervised-threshold", type=float)
    parser.add_argument("--sweep-hybrid", action="store_true")
    parser.add_argument("--sweep-rule", action="store_true")
    parser.add_argument("--sweep-unsupervised", action="store_true")
    parser.add_argument("--sweep-supervised", action="store_true")
    parser.add_argument("--sweep-ensemble", action="store_true")
    parser.add_argument("--sweep-all", action="store_true")
    parser.add_argument("--sweep-rule-layers")
    parser.add_argument("--sweep-repeat-thresholds")
    parser.add_argument("--sweep-thresholds")
    parser.add_argument("--sweep-bad-risk-thresholds")
    parser.add_argument("--sweep-bad-risk-weights")
    parser.add_argument("--sweep-centrality-weights")
    parser.add_argument("--sweep-hybrid-thresholds")
    parser.add_argument("--sweep-hybrid-bad-risk-thresholds")
    parser.add_argument("--sweep-good-margins")
    parser.add_argument("--sweep-supervised-thresholds")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--sweep-output-dir")
    return parser


def parse_args(argv: list[str] | None = None, config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = flatten_config(config or {})
    defaults = {**DEFAULTS, **config}
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    if args.input_path is None:
        args.input_path = defaults.get("input_path")
    if not args.input_path:
        parser.error("input_path is required either as a positional argument or in --config.")
    return args


def parse_float_list(value: str | list[Any]) -> list[float]:
    values = value if isinstance(value, list) else str(value).split(",")
    numbers = [float(item) for item in values if str(item).strip()]
    if not numbers:
        raise ValueError("Threshold list must not be empty.")
    return numbers


def parse_int_list(value: str | list[Any]) -> list[int]:
    values = value if isinstance(value, list) else str(value).split(",")
    numbers = [int(item) for item in values if str(item).strip()]
    if not numbers:
        raise ValueError("Threshold list must not be empty.")
    return numbers


def parse_str_list(value: str | list[Any]) -> list[str]:
    values = value if isinstance(value, list) else str(value).split(",")
    names = [str(item).strip() for item in values if str(item).strip()]
    if not names:
        raise ValueError("String list must not be empty.")
    return names


def has_supervised_training(items) -> bool:
    train_labels = {
        item.meta.label
        for item in items
        if item.meta.split == "train" and item.meta.label in {GOOD_LABEL, BAD_LABEL}
    }
    return train_labels == {GOOD_LABEL, BAD_LABEL}


def build_metric_row(strategy: str, params: dict[str, float | int | str], predictions) -> dict[str, float | int | str]:
    matrix = badcase_confusion(predictions)
    metrics = badcase_metrics(matrix)
    return {
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
    rule_config = getattr(args, "rule_config", None)
    for rule_layer in parse_str_list(args.sweep_rule_layers):
        for repeat_threshold in parse_int_list(args.sweep_repeat_thresholds):
            predictions = classify_rule.classify(items, repeat_threshold, rule_layer, rule_config)
            rows.append(
                build_metric_row(
                    "rule",
                    {"rule_layer": rule_layer, "repeat_threshold": repeat_threshold},
                    predictions,
                )
            )
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
            Prediction(
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
    rule_config = getattr(args, "rule_config", None)
    for rule_layer in parse_str_list(args.sweep_rule_layers):
        for repeat_threshold in parse_int_list(args.sweep_repeat_thresholds):
            rule_predictions = classify_rule.classify(items, repeat_threshold, rule_layer, rule_config)
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
                                            "rule_layer": rule_layer,
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


def print_sweep(rows: list[dict[str, float | int | str]]) -> None:
    columns = build_sweep_columns(rows)
    print("\t".join(columns))
    for index, row in enumerate(rows, start=1):
        print("\t".join(format_sweep_row(row, columns, index)))


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
    return "_".join(names) + "_sweep" if names else "single_run"


def write_sweep_markdown(rows: list[dict[str, float | int | str]], args, method_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.sweep_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method_name}_{timestamp}.md"
    columns = build_sweep_columns(rows)

    lines = [
        f"# {method_name} parameter sweep results",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- input_path: `{args.input_path}`",
        f"- metadata: `{args.metadata or ''}`",
        f"- split: `{args.split or 'all'}`",
        f"- command: `{' '.join(sys.argv)}`",
        "",
        "## Results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append("| " + " | ".join(format_sweep_row(row, columns, index)) + " |")

    lines.extend(
        [
            "",
            "## Selection Notes",
            "",
            "- Prefer rows with high precision and enough predicted_bad coverage.",
            "- Very high precision with near-zero recall usually means the parameters are too conservative.",
            "- High recall with many false positives usually means bad-risk thresholds or margins should be raised.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_single_experiment(args, items) -> list[dict[str, float | int | str]]:
    from run_trace_analysis import run_predictions

    strategy, predictions = run_predictions(args, items)
    return [build_metric_row(strategy, {}, predictions)]


def main(argv: list[str] | None = None, config: dict[str, Any] | None = None) -> int:
    args = parse_args(argv, config)
    metadata = load_metadata(Path(args.metadata)) if args.metadata else None
    trace_files = discover_trace_files(Path(args.input_path))
    items = build_items(trace_files, metadata, args.split)
    if not items:
        print("No trace samples selected.")
        return 0

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
    if not rows:
        rows.extend(run_single_experiment(args, items))

    rows = sort_sweep_rows(rows)
    print_sweep(rows)
    method_name = sweep_method_name(args)
    markdown_path = write_sweep_markdown(rows, args, method_name)
    print(f"\nExperiment markdown written: {markdown_path}")

    if args.output:
        output = {"strategy": method_name, "rows": rows}
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    cli_argv = sys.argv[1:]
    runtime_config = load_config(preparse_config(cli_argv))
    raise SystemExit(main(cli_argv, runtime_config))

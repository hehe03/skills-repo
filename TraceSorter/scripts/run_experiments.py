from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from features import discover_default_final_answer_config, extract_features, load_final_answer_config
from llm_rule_prompt import build_prompt_from_records, call_llm, parse_llm_response, write_llm_rule_report
from metrics import confusion_and_scores
from reporting import default_report_path, write_report
from rule_engine import classify_features, load_rules
from rule_generation import generate_labeled_rules, generate_unlabeled_rules
from trace_io import TraceRecord, load_records, records_with_labels, split_records


SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_RULE_DIR = SCRIPT_DIR / "rules" / "static"
DYNAMIC_RULE_DIR = SCRIPT_DIR / "rules" / "dynamic"
NON_LLM_RULE_DIR = DYNAMIC_RULE_DIR / "non_llm"
LLM_RULE_DIR = DYNAMIC_RULE_DIR / "llm"

GENERAL_RULES = STATIC_RULE_DIR / "general_rules.json"
NON_LLM_UNLABELED_RULES = NON_LLM_RULE_DIR / "unlabeled_rules.json"
NON_LLM_LABELED_RULES = NON_LLM_RULE_DIR / "labeled_rules.json"
LLM_NO_TRAIN_RULES = LLM_RULE_DIR / "no_train_rules.json"
LLM_UNLABELED_RULES = LLM_RULE_DIR / "unlabeled_rules.json"
LLM_LABELED_RULES = LLM_RULE_DIR / "labeled_rules.json"

LEGACY_UNLABELED_RULES = DYNAMIC_RULE_DIR / "unlabeled_rules.json"
LEGACY_LABELED_RULES = DYNAMIC_RULE_DIR / "labeled_rules.json"
LEGACY_LLM_RULES = DYNAMIC_RULE_DIR / "llm_rules.json"

METHODS = {
    "non_llm_no_train",
    "non_llm_unlabeled",
    "non_llm_labeled",
    "llm_no_train",
    "llm_unlabeled",
    "llm_labeled",
}

LEGACY_ALIASES = {
    "general": "non_llm_no_train",
    "unlabeled": "non_llm_unlabeled",
    "labeled": "non_llm_labeled",
}


@dataclass
class DatasetBundle:
    train_records: List[TraceRecord]
    test_records: List[TraceRecord]
    all_records: List[TraceRecord]
    train_source: str
    test_source: str


def _has_both_labels(records: List[TraceRecord]) -> bool:
    labels = {record.label for record in records}
    return {"goodcase", "badcase"}.issubset(labels)


def _load_optional_records(path: str | None, metadata: str | None) -> List[TraceRecord]:
    if not path:
        return []
    return load_records(path, metadata)


def _select_by_split(records: List[TraceRecord], split: str, *, role: str) -> List[TraceRecord]:
    selected = split_records(records, split)
    if not selected:
        raise ValueError(f"metadata split column has no samples for {role} split: {split}")
    return selected


def prepare_datasets(args: argparse.Namespace) -> DatasetBundle:
    all_records = load_records(args.trace_path, args.metadata)

    if args.eval_split:
        test_records = _select_by_split(all_records, args.eval_split, role="eval")
        test_source = f"trace_path split={args.eval_split}"
    else:
        test_records = all_records
        test_source = "trace_path all samples"

    explicit_train = bool(args.train_trace_path)
    train_records = _load_optional_records(args.train_trace_path, args.train_metadata or args.metadata)
    train_source = "none"
    if explicit_train:
        train_source = f"train_trace_path={args.train_trace_path}"
        if args.train_split:
            train_records = _select_by_split(train_records, args.train_split, role="train")
            train_source += f", split={args.train_split}"
    elif args.train_split:
        train_records = _select_by_split(all_records, args.train_split, role="train")
        train_source = f"trace_path split={args.train_split}"

    return DatasetBundle(
        train_records=train_records,
        test_records=test_records,
        all_records=all_records,
        train_source=train_source,
        test_source=test_source,
    )


def method_family(method: str) -> str:
    return "llm" if method.startswith("llm_") else "non_llm"


def method_training_scenario(method: str) -> str:
    if method.endswith("_no_train"):
        return "no_train"
    if method.endswith("_unlabeled"):
        return "unlabeled"
    if method.endswith("_labeled"):
        return "labeled"
    raise ValueError(f"unsupported method: {method}")


def _llm_auto_method(bundle: DatasetBundle) -> str:
    if _has_both_labels(records_with_labels(bundle.train_records)):
        return "llm_labeled"
    if bundle.train_records:
        return "llm_unlabeled"
    return "llm_no_train"


def parse_methods(value: str | None, bundle: DatasetBundle | None = None) -> List[str]:
    if not value:
        return []
    methods: List[str] = []
    for item in value.split(","):
        raw = item.strip().lower()
        if not raw:
            continue
        if raw == "all":
            for candidate in (
                "non_llm_no_train",
                "non_llm_unlabeled",
                "non_llm_labeled",
                "llm_no_train",
                "llm_unlabeled",
                "llm_labeled",
            ):
                if candidate not in methods:
                    methods.append(candidate)
            continue
        if raw == "llm":
            if bundle is None:
                raise ValueError("llm alias requires loaded train/test data")
            raw = _llm_auto_method(bundle)
        method = LEGACY_ALIASES.get(raw, raw)
        if method not in METHODS:
            raise ValueError(f"unsupported method: {raw}")
        if method not in methods:
            methods.append(method)
    return methods


def _parse_llm_extra(items: List[str]) -> Dict[str, Any]:
    extra_args: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--llm-extra must use key=value format: {item}")
        key, value = item.split("=", 1)
        extra_args[key.strip()] = value.strip()
    return extra_args


def _write_rules_payload(path: Path, payload: Dict[str, Any]) -> int:
    rules = payload.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("rule payload must contain a JSON array field named 'rules'")
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_payload = {
        "rules": rules,
        "final_answer_config": payload.get("final_answer_config", {}),
        "proposed_features": payload.get("proposed_features", []),
        "note": "Generated by scripts/run_experiments.py.",
    }
    path.write_text(json.dumps(saved_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rules)


def _copy_payload(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _validate_train_data(method: str, train_records: List[TraceRecord]) -> None:
    scenario = method_training_scenario(method)
    if scenario == "no_train":
        return
    if not train_records:
        raise ValueError(
            f"{method} requires training traces. Use --train-trace-path or --train-split."
        )
    if scenario == "labeled" and not _has_both_labels(records_with_labels(train_records)):
        raise ValueError(
            f"{method} requires labeled train traces containing both goodcase and badcase."
        )


def rules_path_for_method(method: str) -> Path:
    mapping = {
        "non_llm_no_train": GENERAL_RULES,
        "non_llm_unlabeled": NON_LLM_UNLABELED_RULES,
        "non_llm_labeled": NON_LLM_LABELED_RULES,
        "llm_no_train": LLM_NO_TRAIN_RULES,
        "llm_unlabeled": LLM_UNLABELED_RULES,
        "llm_labeled": LLM_LABELED_RULES,
    }
    return mapping[method]


def rule_paths_for_method(method: str) -> List[Path]:
    if method == "non_llm_no_train":
        return [GENERAL_RULES]
    return [GENERAL_RULES, rules_path_for_method(method)]


def generate_non_llm_rules(
    method: str,
    train_records: List[TraceRecord],
    final_answer_config: Any,
) -> str | None:
    if method == "non_llm_no_train":
        return None
    _validate_train_data(method, train_records)
    if method == "non_llm_unlabeled":
        rules = generate_unlabeled_rules(train_records, NON_LLM_UNLABELED_RULES, final_answer_config)
        _copy_payload(NON_LLM_UNLABELED_RULES, LEGACY_UNLABELED_RULES)
    elif method == "non_llm_labeled":
        rules = generate_labeled_rules(records_with_labels(train_records), NON_LLM_LABELED_RULES, final_answer_config)
        _copy_payload(NON_LLM_LABELED_RULES, LEGACY_LABELED_RULES)
    else:
        raise ValueError(f"not a non-LLM method: {method}")
    print(f"Generated {method} rules: {rules_path_for_method(method)} ({len(rules)} rules)")
    return str(rules_path_for_method(method))


def generate_llm_rules(
    method: str,
    train_records: List[TraceRecord],
    final_answer_config: Any,
    args: argparse.Namespace,
) -> str:
    _validate_train_data(method, train_records)
    scenario = method_training_scenario(method)
    prompt_records = train_records if scenario != "no_train" else []
    prompt = build_prompt_from_records(
        prompt_records,
        final_answer_config=final_answer_config,
        training_scenario=scenario,
        max_samples=args.llm_max_samples,
        max_prompt_chars=args.llm_max_prompt_chars,
        max_trace_chars=args.llm_max_trace_chars,
        max_dynamic_fields=args.llm_max_dynamic_fields,
    )
    prompt_path = (
        Path(args.llm_prompt_output)
        if args.llm_prompt_output
        else Path(args.output_dir) / f"{method}_prompt.md"
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    response = call_llm(
        prompt,
        provider=args.llm_provider,
        model=args.llm_model,
        temperature=args.llm_temperature,
        extra_args=_parse_llm_extra(args.llm_extra),
    )
    if not response:
        raise RuntimeError(
            f"{method} was selected, so call_llm() was triggered but returned empty output. "
            "Please implement scripts/llm_rule_prompt.py::call_llm()."
        )

    llm_output_path = (
        Path(args.llm_output)
        if args.llm_output
        else Path(args.output_dir) / f"{method}_response.json"
    )
    llm_output_path.parent.mkdir(parents=True, exist_ok=True)
    llm_output_path.write_text(response, encoding="utf-8")

    payload = parse_llm_response(response)
    output_rules = rules_path_for_method(method)
    rule_count = _write_rules_payload(output_rules, payload)
    if method == "llm_labeled":
        _copy_payload(output_rules, LEGACY_LLM_RULES)

    report_path = (
        Path(args.llm_report_output)
        if args.llm_report_output
        else Path(args.output_dir) / f"{method}_llm_rule_repoert.md"
    )
    write_llm_rule_report(
        report_path,
        llm_output_path=str(llm_output_path),
        rules_path=output_rules,
        prompt_path=str(prompt_path),
    )
    print(f"Generated {method} rules: {output_rules} ({rule_count} rules)")
    print(f"Wrote LLM prompt: {prompt_path}")
    print(f"Wrote LLM response: {llm_output_path}")
    print(f"Wrote LLM rule report: {report_path}")
    return str(output_rules)


def train_method(
    method: str,
    bundle: DatasetBundle,
    final_answer_config: Any,
    args: argparse.Namespace,
) -> List[str]:
    family = method_family(method)
    if family == "non_llm":
        generated = generate_non_llm_rules(method, bundle.train_records, final_answer_config)
        return [generated] if generated else []
    return [generate_llm_rules(method, bundle.train_records, final_answer_config, args)]


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


def _method_notes(
    method: str,
    bundle: DatasetBundle,
    generated_rule_files: List[str],
) -> List[str]:
    scenario = method_training_scenario(method)
    notes = [
        f"Method family: `{method_family(method)}`",
        f"Training scenario: `{scenario}`",
        f"Train source: `{bundle.train_source}`",
        f"Train samples: {len(bundle.train_records) if scenario != 'no_train' else 0}",
        f"Test source: `{bundle.test_source}`",
        f"Test samples: {len(bundle.test_records)}",
    ]
    notes.extend(f"Generated dynamic rule file: `{path}`" for path in generated_rule_files)
    return notes


def run_single_method(args: argparse.Namespace, method: str, bundle: DatasetBundle) -> Path:
    final_answer_config = discover_default_final_answer_config(
        bundle.train_records or bundle.all_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )
    generated_rule_files = train_method(method, bundle, final_answer_config, args)
    rules = load_rules(rule_paths_for_method(method))
    results = predict_with_method(bundle.test_records, rules, final_answer_config, args)

    output_path = Path(args.output) if args.output else default_report_path(args.output_dir, method)
    return write_report(
        output_path,
        method=method,
        trace_path=args.trace_path,
        metadata_path=args.metadata,
        rules=rules,
        results=results,
        max_rows=args.max_rows,
        notes=_method_notes(method, bundle, generated_rule_files),
    )


def run_methods_comparison(args: argparse.Namespace, methods: List[str], bundle: DatasetBundle) -> Path:
    final_answer_config = discover_default_final_answer_config(
        bundle.train_records or bundle.all_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )
    generated_by_method: Dict[str, List[str]] = {}
    results_by_method: Dict[str, List[Dict[str, Any]]] = {}
    rules_by_method: Dict[str, List[Dict[str, Any]]] = {}

    for method in methods:
        generated_by_method[method] = train_method(method, bundle, final_answer_config, args)
        rules = load_rules(rule_paths_for_method(method))
        rules_by_method[method] = rules
        results_by_method[method] = predict_with_method(bundle.test_records, rules, final_answer_config, args)

    output_path = Path(args.output) if args.output else default_report_path(args.output_dir, "methods_comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Trace Sorter Methods Comparison",
        "",
        f"- Trace path: `{args.trace_path}`",
        f"- Metadata: `{args.metadata}`" if args.metadata else "- Metadata: none",
        f"- Train source: `{bundle.train_source}`",
        f"- Train samples: {len(bundle.train_records)}",
        f"- Test source: `{bundle.test_source}`",
        f"- Test samples: {len(bundle.test_records)}",
        f"- Methods: {', '.join(methods)}",
        "",
        "## Method Structure",
        "",
        "| method | family | training scenario | rules loaded | generated rule files |",
        "|---|---|---|---:|---|",
    ]
    for method in methods:
        generated = ", ".join(f"`{path}`" for path in generated_by_method[method]) or ""
        lines.append(
            f"| {method} | {method_family(method)} | {method_training_scenario(method)} | "
            f"{len(rules_by_method[method])} | {generated} |"
        )
    lines.append("")

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
    for record in bundle.test_records[: args.max_rows]:
        row_cells = [f"`{record.name}`", record.label or ""]
        for method in methods:
            prediction = by_method_name[method].get(record.name, {})
            row_cells.append(
                f"{prediction.get('predicted_label', '')} "
                f"(bad={prediction.get('bad_score', '')}, good={prediction.get('good_score', '')})"
            )
        lines.append("| " + " | ".join(row_cells) + " |")
    if len(bundle.test_records) > args.max_rows:
        lines.append(f"| ... | ... | {' | '.join('...' for _ in methods)} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train rule methods and test Agent trace goodcase/badcase classification."
    )
    parser.add_argument("trace_path", help="Test trace JSON file or directory, unless --eval-split selects a subset.")
    parser.add_argument(
        "--metadata",
        help="Optional test/all metadata CSV with columns name,label,source,split.",
    )
    parser.add_argument(
        "--train-trace-path",
        help="Optional separate training trace JSON file or directory.",
    )
    parser.add_argument(
        "--train-metadata",
        help="Optional training metadata CSV. Defaults to --metadata when omitted.",
    )
    parser.add_argument(
        "--train-split",
        help="Split value used as training data. Common value: train. If omitted and no --train-trace-path is given, no training data is used.",
    )
    parser.add_argument(
        "--eval-split",
        help="Split value used as test/evaluation data. If omitted, all samples from trace_path are tested.",
    )
    parser.add_argument(
        "--method",
        "--rule-layer",
        dest="method",
        default="non_llm_no_train",
        help="Single method to run when --methods is not set. Legacy aliases: general, unlabeled, labeled, llm.",
    )
    parser.add_argument(
        "--methods",
        help=(
            "Comma-separated methods to compare. Valid methods: non_llm_no_train, "
            "non_llm_unlabeled, non_llm_labeled, llm_no_train, llm_unlabeled, llm_labeled. "
            "Legacy aliases general, unlabeled, labeled, llm and all are supported."
        ),
    )
    parser.add_argument(
        "--generate-dynamic-rules",
        choices=["none", "unlabeled", "labeled", "both", "auto"],
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--bad-threshold", type=float, default=0.60)
    parser.add_argument("--good-threshold", type=float, default=0.50)
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
    parser.add_argument("--llm-provider", help="Provider name passed to call_llm() when an LLM method is selected.")
    parser.add_argument("--llm-model", help="Model name passed to call_llm() when an LLM method is selected.")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Temperature passed to call_llm().")
    parser.add_argument(
        "--llm-extra",
        action="append",
        default=[],
        help="Extra key=value argument passed to call_llm(). Can be repeated.",
    )
    parser.add_argument("--llm-output", help="Path for raw LLM response JSON.")
    parser.add_argument("--llm-prompt-output", help="Path for generated LLM prompt.")
    parser.add_argument("--llm-report-output", help="Path for Chinese LLM rule report.")
    parser.add_argument("--llm-max-samples", type=int, default=30, help="Maximum representative train samples included in the LLM prompt.")
    parser.add_argument("--llm-max-prompt-chars", type=int, default=60000, help="Maximum LLM prompt characters before truncation.")
    parser.add_argument("--llm-max-trace-chars", type=int, default=2000, help="Maximum raw trace excerpt characters per selected sample.")
    parser.add_argument("--llm-max-dynamic-fields", type=int, default=80, help="Maximum dynamic field paths included per selected sample.")
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle = prepare_datasets(args)
    methods = parse_methods(args.methods, bundle) if args.methods else parse_methods(args.method, bundle)
    if not methods:
        raise ValueError("No method selected.")
    report_path = (
        run_methods_comparison(args, methods, bundle)
        if len(methods) > 1
        else run_single_method(args, methods[0], bundle)
    )
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    # You may write parameters here when running this file from an IDE.
    # Example:
    # SCRIPT_ARGS = [r".\traces", "--method", "non_llm_no_train", "--output-dir", r".\results"]
    SCRIPT_ARGS = None
    main(SCRIPT_ARGS)

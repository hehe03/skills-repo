from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from experiment_components.ablation_plans import AblationVariant, build_ablation_variants
from experiment_components.contribution import summarize_result_components, summarize_rule_components
from experiment_components.rule_filters import filter_rules, parse_component_csv
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

METHODS = {
    "non_llm_no_train",
    "non_llm_unlabeled",
    "non_llm_labeled",
    "llm_no_train",
    "llm_unlabeled",
    "llm_labeled",
}

METHOD_ALIASES = {
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
        method = METHOD_ALIASES.get(raw, raw)
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
    elif method == "non_llm_labeled":
        rules = generate_labeled_rules(records_with_labels(train_records), NON_LLM_LABELED_RULES, final_answer_config)
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
    for rule in payload.get("rules") or []:
        if isinstance(rule, dict):
            rule.setdefault("layer", "llm")
            rule.setdefault("component", "llm_rules")
            rule.setdefault("source_method", method)
    output_rules = rules_path_for_method(method)
    rule_count = _write_rules_payload(output_rules, payload)

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
    if args.llm_use_existing_rules:
        _validate_train_data(method, bundle.train_records)
        print(f"Using existing LLM rules without calling call_llm(): {rules_path_for_method(method)}")
        return []
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


def _component_filter_note(args: argparse.Namespace) -> List[str]:
    notes: List[str] = []
    if args.enable_components:
        notes.append(f"Enabled components: `{args.enable_components}`")
    if args.disable_components:
        notes.append(f"Disabled components: `{args.disable_components}`")
    if args.ablation_plan != "none":
        notes.append(f"Ablation plan: `{args.ablation_plan}`")
    return notes


def _filtered_rules(raw_rules: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return filter_rules(
        raw_rules,
        enable_components=parse_component_csv(args.enable_components),
        disable_components=parse_component_csv(args.disable_components),
    )


def _method_notes(
    method: str,
    bundle: DatasetBundle,
    generated_rule_files: List[str],
    args: argparse.Namespace | None = None,
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
    if args:
        notes.extend(_component_filter_note(args))
    return notes


def _format_component_tuple(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else ""


def write_ablation_report(
    path: str | Path,
    *,
    method: str,
    trace_path: str,
    metadata_path: str | None,
    bundle: DatasetBundle,
    raw_rule_count: int,
    variants: List[AblationVariant],
    results_by_variant: Dict[str, List[Dict[str, Any]]],
    max_rows: int,
    notes: List[str],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_results = results_by_variant.get("baseline", [])
    baseline_by_name = {row["name"]: row for row in baseline_results}
    lines: List[str] = [
        f"# Trace Sorter Component Ablation: {method}",
        "",
        f"- Trace path: `{trace_path}`",
        f"- Metadata: `{metadata_path}`" if metadata_path else "- Metadata: none",
        f"- Train source: `{bundle.train_source}`",
        f"- Train samples: {len(bundle.train_records)}",
        f"- Test source: `{bundle.test_source}`",
        f"- Test samples: {len(bundle.test_records)}",
        f"- Raw rules before component filtering: {raw_rule_count}",
        "",
    ]
    if notes:
        lines.extend(["## Run Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    lines.extend(
        [
            "## Ablation Metrics",
            "",
            "| variant | rules | disabled components | enabled components | accuracy | precision(badcase) | recall(badcase) | f1(badcase) | tp | fp | tn | fn | changed vs baseline |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in variants:
        rows = results_by_variant[variant.name]
        metrics = confusion_and_scores(rows)
        changed = 0
        if variant.name != "baseline":
            for row in rows:
                base = baseline_by_name.get(row["name"], {})
                if row.get("predicted_label") != base.get("predicted_label"):
                    changed += 1
        lines.append(
            f"| `{variant.name}` | {len(variant.rules)} | "
            f"{_format_component_tuple(variant.disabled_components)} | "
            f"{_format_component_tuple(variant.enabled_components)} | "
            f"{metrics['accuracy']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} | {changed} |"
        )
    lines.append("")

    lines.extend(["## Baseline Component Rule Summary", ""])
    lines.append("| component | rules | bad rules | good rules | weight sum |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summarize_rule_components(variants[0].rules):
        lines.append(
            f"| `{row['component']}` | {row['rules']} | {row['bad_rules']} | "
            f"{row['good_rules']} | {round(row['weight_sum'], 4)} |"
        )
    lines.append("")

    lines.extend(["## Baseline Component Hit Summary", ""])
    lines.append("| component | cases hit | hits | bad score | good score | correct cases | incorrect cases |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summarize_result_components(baseline_results):
        lines.append(
            f"| `{row['component']}` | {row['cases']} | {row['hits']} | "
            f"{round(row['bad_score'], 4)} | {round(row['good_score'], 4)} | "
            f"{row['correct_cases']} | {row['incorrect_cases']} |"
        )
    lines.append("")

    lines.extend(["## Prediction Changes", ""])
    lines.append("| name | label | variant | baseline | variant prediction |")
    lines.append("|---|---|---|---|---|")
    change_rows = 0
    for variant in variants:
        if variant.name == "baseline":
            continue
        for row in results_by_variant[variant.name]:
            base = baseline_by_name.get(row["name"], {})
            if row.get("predicted_label") == base.get("predicted_label"):
                continue
            lines.append(
                f"| `{row['name']}` | {row.get('label') or ''} | `{variant.name}` | "
                f"{base.get('predicted_label', '')} | {row.get('predicted_label', '')} |"
            )
            change_rows += 1
            if change_rows >= max_rows:
                break
        if change_rows >= max_rows:
            break
    if change_rows == 0:
        lines.append("| none | | | | |")
    lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_single_method(args: argparse.Namespace, method: str, bundle: DatasetBundle) -> Path:
    final_answer_config = discover_default_final_answer_config(
        bundle.train_records or bundle.all_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )
    generated_rule_files = train_method(method, bundle, final_answer_config, args)
    raw_rules = load_rules(rule_paths_for_method(method))
    if args.ablation_plan != "none":
        variants = build_ablation_variants(
            raw_rules,
            plan=args.ablation_plan,
            enable_components=parse_component_csv(args.enable_components),
            disable_components=parse_component_csv(args.disable_components),
        )
        results_by_variant = {
            variant.name: predict_with_method(bundle.test_records, variant.rules, final_answer_config, args)
            for variant in variants
        }
        output_path = Path(args.output) if args.output else default_report_path(args.output_dir, f"{method}_{args.ablation_plan}")
        return write_ablation_report(
            output_path,
            method=method,
            trace_path=args.trace_path,
            metadata_path=args.metadata,
            bundle=bundle,
            raw_rule_count=len(raw_rules),
            variants=variants,
            results_by_variant=results_by_variant,
            max_rows=args.max_rows,
            notes=_method_notes(method, bundle, generated_rule_files, args),
        )

    rules = _filtered_rules(raw_rules, args)
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
        notes=_method_notes(method, bundle, generated_rule_files, args),
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
        rules = _filtered_rules(load_rules(rule_paths_for_method(method)), args)
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
    ]
    lines.extend(f"- {note}" for note in _component_filter_note(args))
    lines.extend(
        [
            "",
            "## Method Structure",
            "",
            "| method | family | training scenario | rules loaded | generated rule files |",
            "|---|---|---|---:|---|",
        ]
    )
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


def run_methods_ablation_comparison(args: argparse.Namespace, methods: List[str], bundle: DatasetBundle) -> Path:
    final_answer_config = discover_default_final_answer_config(
        bundle.train_records or bundle.all_records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )
    output_path = Path(args.output) if args.output else default_report_path(args.output_dir, "methods_component_ablation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Trace Sorter Methods Component Ablation",
        "",
        f"- Trace path: `{args.trace_path}`",
        f"- Metadata: `{args.metadata}`" if args.metadata else "- Metadata: none",
        f"- Train source: `{bundle.train_source}`",
        f"- Train samples: {len(bundle.train_records)}",
        f"- Test source: `{bundle.test_source}`",
        f"- Test samples: {len(bundle.test_records)}",
        f"- Methods: {', '.join(methods)}",
        f"- Ablation plan: `{args.ablation_plan}`",
    ]
    lines.extend(f"- {note}" for note in _component_filter_note(args) if not note.startswith("Ablation plan:"))
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| method | variant | rules | disabled components | enabled components | accuracy | precision(badcase) | recall(badcase) | f1(badcase) | tp | fp | tn | fn | changed vs baseline |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        train_method(method, bundle, final_answer_config, args)
        raw_rules = load_rules(rule_paths_for_method(method))
        variants = build_ablation_variants(
            raw_rules,
            plan=args.ablation_plan,
            enable_components=parse_component_csv(args.enable_components),
            disable_components=parse_component_csv(args.disable_components),
        )
        results_by_variant = {
            variant.name: predict_with_method(bundle.test_records, variant.rules, final_answer_config, args)
            for variant in variants
        }
        baseline_by_name = {row["name"]: row for row in results_by_variant.get("baseline", [])}
        for variant in variants:
            rows = results_by_variant[variant.name]
            metrics = confusion_and_scores(rows)
            changed = 0
            if variant.name != "baseline":
                for row in rows:
                    base = baseline_by_name.get(row["name"], {})
                    if row.get("predicted_label") != base.get("predicted_label"):
                        changed += 1
            lines.append(
                f"| `{method}` | `{variant.name}` | {len(variant.rules)} | "
                f"{_format_component_tuple(variant.disabled_components)} | "
                f"{_format_component_tuple(variant.enabled_components)} | "
                f"{metrics['accuracy']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | "
                f"{metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} | {changed} |"
            )
    lines.append("")
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
    parser.add_argument(
        "--enable-components",
        help="Comma-separated rule components to keep. When omitted, all components are enabled before --disable-components is applied.",
    )
    parser.add_argument(
        "--disable-components",
        help="Comma-separated rule components to remove from loaded rules, for component ablation or manual filtering.",
    )
    parser.add_argument(
        "--ablation-plan",
        choices=["none", "leave_one_component_out", "only_one_component"],
        default="none",
        help="Run component ablation variants after training. Default keeps the normal single run behavior.",
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
    parser.add_argument(
        "--llm-use-existing-rules",
        action="store_true",
        help="For Agent-driven workflows: load existing LLM rule files and do not call call_llm().",
    )
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
    if len(methods) > 1 and args.ablation_plan != "none":
        report_path = run_methods_ablation_comparison(args, methods, bundle)
    elif len(methods) > 1:
        report_path = run_methods_comparison(args, methods, bundle)
    else:
        report_path = run_single_method(args, methods[0], bundle)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    # You may write parameters here when running this file from an IDE.
    # Example:
    # SCRIPT_ARGS = [r".\traces", "--method", "non_llm_no_train", "--output-dir", r".\results"]
    SCRIPT_ARGS = None
    main(SCRIPT_ARGS)

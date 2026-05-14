from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from features import discover_default_final_answer_config, extract_features, load_final_answer_config
from trace_io import TraceRecord, load_records, records_with_labels, split_records


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LLM_RULES = SCRIPT_DIR / "rules" / "dynamic" / "llm" / "labeled_rules.json"

FIXED_FEATURE_NAMES = [
    "parse_error",
    "is_empty_trace",
    "has_steps",
    "step_count",
    "unique_action_count",
    "unique_action_ratio",
    "repeated_action_count",
    "max_consecutive_same_action",
    "error_count",
    "has_error_text",
    "empty_result_count",
    "empty_result_ratio",
    "nonempty_result_ratio",
    "has_final_answer",
    "final_answer_evidence_enabled",
    "final_answer_evidence_strength",
    "final_answer_evidence_source",
    "final_answer_adopted_fields",
    "final_answer_chars",
    "final_answer_source",
    "text_chars",
]

FEATURE_DESCRIPTIONS = {
    "fixed_features": {
        "parse_error": "JSON parse failed.",
        "is_empty_trace": "Trace object is empty.",
        "has_steps": "Trace contains observable steps, messages, events, or actions.",
        "step_count": "Number of observable steps/actions.",
        "unique_action_ratio": "Unique action names divided by step_count.",
        "repeated_action_count": "Total repeated action count.",
        "max_consecutive_same_action": "Longest run of the same consecutive action.",
        "error_count": "Number of error/failure/timeout terms found in text.",
        "empty_result_ratio": "Empty result count divided by step_count.",
        "has_final_answer": "Whether a final answer/final response is visible.",
        "text_chars": "Total visible trace text characters.",
    },
    "dynamic_field_features": {
        "trace_field_paths": "Newline-separated normalized leaf field paths observed in the trace.",
        "field_exists:<path>": "True when the normalized trace field path exists.",
        "field_count:<path>": "Number of leaf values found at this path.",
        "field_text:<path>": "Joined text values found at this path.",
        "field_nonempty_ratio:<path>": "Non-empty value ratio for this path.",
        "field_number_mean:<path>": "Mean numeric value for this path when all values are numeric.",
        "field_number_min:<path>": "Minimum numeric value for this path when all values are numeric.",
        "field_number_max:<path>": "Maximum numeric value for this path when all values are numeric.",
        "field_bool_true_ratio:<path>": "True ratio for this path when all values are boolean.",
    },
}


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_llm_response(text: str) -> Dict[str, Any]:
    return json.loads(_extract_json_text(text))


def load_rules_payload(path: str | Path) -> Dict[str, Any]:
    rule_path = Path(path)
    if not rule_path.exists():
        return {"rules": []}
    with rule_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"rules": data}
    return data if isinstance(data, dict) else {"rules": []}


def call_llm(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    extra_args: Dict[str, Any] | None = None,
) -> str:
    """Implement this hook to call your LLM provider and return raw JSON text."""
    return ""


def _conditions_to_text(rule: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("all", "any"):
        conditions = rule.get(key) or []
        if not conditions:
            continue
        rendered = [
            f"{condition.get('feature')} {condition.get('op', '==')} {condition.get('value')}"
            for condition in conditions
        ]
        parts.append(f"{key}: " + "; ".join(rendered))
    return " / ".join(parts) if parts else "无条件"


def write_llm_rule_report(
    output_path: str | Path,
    *,
    llm_output_path: str | None,
    rules_path: str | Path,
    prompt_path: str | None,
) -> Path:
    if llm_output_path:
        llm_payload = parse_llm_response(Path(llm_output_path).read_text(encoding="utf-8-sig"))
    else:
        llm_payload = {}
    accepted_payload = load_rules_payload(rules_path)
    rules = llm_payload.get("rules") or accepted_payload.get("rules") or []
    final_answer_config = llm_payload.get("final_answer_config") or accepted_payload.get("final_answer_config") or {}
    proposed_features = llm_payload.get("proposed_features") or accepted_payload.get("proposed_features") or []

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# LLM 规则发现报告",
        "",
        f"- Prompt 文件: `{prompt_path}`" if prompt_path else "- Prompt 文件: 未写入文件",
        f"- LLM 原始输出: `{llm_output_path}`" if llm_output_path else "- LLM 原始输出: 未提供",
        f"- 已采纳规则文件: `{rules_path}`",
        f"- 发现/采纳规则数: {len(rules)}",
        "",
        "## Final Answer 字段发现",
        "",
    ]
    if final_answer_config:
        lines.append("| item | value |")
        lines.append("|---|---|")
        for key in (
            "evidence_source",
            "top_level_keys",
            "nested_keys",
            "final_answer_items",
            "assistant_roles",
            "assistant_content_keys",
            "min_chars",
            "rationale",
        ):
            if key in final_answer_config:
                value = final_answer_config[key]
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                lines.append(f"| {key} | {value} |")
    else:
        lines.append("LLM 未提供 final-answer 字段配置，或当前规则文件中没有该配置。")
    lines.append("")

    lines.extend(["## LLM 发现的规则", ""])
    if rules:
        lines.append("| id | label | weight | description | conditions |")
        lines.append("|---|---|---:|---|---|")
        for rule in rules:
            description = str(rule.get("description", "")).replace("|", "\\|")
            conditions = _conditions_to_text(rule).replace("|", "\\|")
            lines.append(
                f"| `{rule.get('id', '')}` | {rule.get('label', '')} | "
                f"{rule.get('weight', '')} | {description} | {conditions} |"
            )
    else:
        lines.append("没有发现可报告的 LLM 规则。")
    lines.append("")

    lines.extend(["## LLM 建议新增的特征", ""])
    if proposed_features:
        lines.append("| feature | description |")
        lines.append("|---|---|")
        for feature in proposed_features:
            if isinstance(feature, dict):
                description = str(feature.get("description", "")).replace("|", "\\|")
                lines.append(f"| `{feature.get('name', feature.get('feature', ''))}` | {description} |")
            else:
                lines.append(f"| `{feature}` | |")
    else:
        lines.append("没有新增特征建议。")
    lines.append("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _json_excerpt(value: Any, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _median(values: List[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _numeric_stats(values: List[Any]) -> Dict[str, float] | None:
    numbers: List[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            pass
    if not numbers:
        return None
    return {
        "min": round(min(numbers), 4),
        "median": _median(numbers),
        "max": round(max(numbers), 4),
    }


def _row_signature(row: Dict[str, Any]) -> tuple[Any, ...]:
    features = row["features"]
    return (
        features.get("has_error_text"),
        features.get("has_final_answer"),
        min(int(features.get("step_count", 0)), 20),
        int(features.get("text_chars", 0)) // 500,
        hash(features.get("trace_field_paths", "")) % 17,
    )


def _select_diverse_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    selected: List[Dict[str, Any]] = []
    seen_signatures: set[tuple[Any, ...]] = set()

    sorted_rows = sorted(rows, key=lambda row: int(row["features"].get("text_chars", 0)), reverse=True)
    priority_rows = [
        *sorted(rows, key=lambda row: int(row["features"].get("error_count", 0)), reverse=True)[:2],
        *sorted_rows[:2],
        *sorted(rows, key=lambda row: int(row["features"].get("text_chars", 0)))[:2],
        *rows,
    ]
    for row in priority_rows:
        signature = _row_signature(row)
        if signature in seen_signatures and len(selected) >= max(1, limit // 2):
            continue
        if row not in selected:
            selected.append(row)
            seen_signatures.add(signature)
        if len(selected) >= limit:
            return selected
    return selected[:limit]


def _select_prompt_rows(
    rows: List[Dict[str, Any]],
    training_scenario: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    if training_scenario == "labeled":
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row.get("label") or "unlabeled"].append(row)
        labels = [label for label in ("goodcase", "badcase") if grouped.get(label)]
        if not labels:
            return _select_diverse_rows(rows, max_samples)
        per_label = max(1, max_samples // len(labels))
        selected: List[Dict[str, Any]] = []
        for label in labels:
            selected.extend(_select_diverse_rows(grouped[label], per_label))
        if len(selected) < max_samples:
            selected.extend(row for row in _select_diverse_rows(rows, max_samples) if row not in selected)
        return selected[:max_samples]
    return _select_diverse_rows(rows, max_samples)


def _extract_field_paths(features: Dict[str, Any]) -> List[str]:
    raw = str(features.get("trace_field_paths", "")).strip()
    return [line for line in raw.splitlines() if line.strip()]


def _summarize_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = Counter(row.get("label") or "unlabeled" for row in rows)
    sources = Counter(row.get("source") or "" for row in rows)
    splits = Counter(row.get("split") or "" for row in rows)
    text_chars = [float(row["features"].get("text_chars", 0)) for row in rows]
    step_counts = [float(row["features"].get("step_count", 0)) for row in rows]
    field_counts = Counter()
    for row in rows:
        field_counts.update(_extract_field_paths(row["features"]))
    return {
        "sample_count": len(rows),
        "labels": dict(labels),
        "sources_top": dict(sources.most_common(10)),
        "splits": dict(splits),
        "text_chars": _numeric_stats(text_chars),
        "step_count": _numeric_stats(step_counts),
        "top_field_paths": dict(field_counts.most_common(80)),
    }


def _summarize_label_contrasts(rows: List[Dict[str, Any]], max_items: int = 40) -> List[Dict[str, Any]]:
    grouped = {
        "goodcase": [row for row in rows if row.get("label") == "goodcase"],
        "badcase": [row for row in rows if row.get("label") == "badcase"],
    }
    if not grouped["goodcase"] or not grouped["badcase"]:
        return []

    paths = sorted(
        set(path for row in rows for path in _extract_field_paths(row["features"]))
    )
    contrasts: List[Dict[str, Any]] = []
    for path in paths:
        good_present = sum(1 for row in grouped["goodcase"] if row["features"].get(f"field_exists:{path}") is True)
        bad_present = sum(1 for row in grouped["badcase"] if row["features"].get(f"field_exists:{path}") is True)
        good_rate = good_present / len(grouped["goodcase"])
        bad_rate = bad_present / len(grouped["badcase"])
        diff = bad_rate - good_rate
        if abs(diff) >= 0.25:
            contrasts.append(
                {
                    "field": path,
                    "signal": "presence",
                    "good_rate": round(good_rate, 3),
                    "bad_rate": round(bad_rate, 3),
                    "direction": "badcase" if diff > 0 else "goodcase",
                }
            )
    return sorted(contrasts, key=lambda item: abs(item["bad_rate"] - item["good_rate"]), reverse=True)[:max_items]


def _summarize_feature_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for feature in FIXED_FEATURE_NAMES:
        values = [row["features"].get(feature) for row in rows if feature in row["features"]]
        if not values:
            continue
        numeric = _numeric_stats(values)
        if numeric:
            stats[feature] = numeric
        else:
            stats[feature] = dict(Counter(str(value) for value in values).most_common(8))
    return stats


def _compact_features(features: Dict[str, Any], max_dynamic_fields: int) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        key: features[key] for key in FIXED_FEATURE_NAMES if key in features
    }
    paths = _extract_field_paths(features)[:max_dynamic_fields]
    compact["trace_field_paths"] = paths
    for path in paths:
        for prefix in (
            "field_exists",
            "field_count",
            "field_nonempty_ratio",
            "field_number_mean",
            "field_number_min",
            "field_number_max",
            "field_bool_true_ratio",
        ):
            key = f"{prefix}:{path}"
            if key in features:
                compact[key] = features[key]
        text_key = f"field_text:{path}"
        if text_key in features:
            text = str(features[text_key])
            compact[text_key] = text[:300] + ("...<truncated>" if len(text) > 300 else "")
    return compact


def _build_rows(records: List[TraceRecord], final_answer_config: Any) -> List[Dict[str, Any]]:
    return [
        {
            "name": record.name,
            "label": record.label,
            "source": record.source,
            "split": record.split,
            "trace": record.trace,
            "features": extract_features(record, final_answer_config),
        }
        for record in records
    ]


def _prompt_payload(
    records: List[TraceRecord],
    *,
    final_answer_config: Any,
    training_scenario: str,
    max_samples: int,
    max_trace_chars: int,
    max_dynamic_fields: int,
) -> Dict[str, Any]:
    rows = _build_rows(records, final_answer_config)
    selected = _select_prompt_rows(rows, training_scenario, max_samples)
    samples = []
    for row in selected:
        samples.append(
            {
                "name": row["name"],
                "label": row["label"],
                "source": row["source"],
                "split": row["split"],
                "features": _compact_features(row["features"], max_dynamic_fields),
                "trace_excerpt": _json_excerpt(row["trace"], max_trace_chars),
            }
        )
    return {
        "dataset_summary": _summarize_dataset(rows),
        "fixed_feature_stats": _summarize_feature_stats(rows),
        "label_contrasts": _summarize_label_contrasts(rows) if training_scenario == "labeled" else [],
        "selected_samples": samples,
        "selection_policy": {
            "max_samples": max_samples,
            "max_trace_chars_per_sample": max_trace_chars,
            "max_dynamic_fields_per_sample": max_dynamic_fields,
            "labeled_policy": "include both goodcase and badcase when available",
            "diversity_policy": "prefer varied length, error counts, final-answer presence, and field-path signatures",
        },
    }


def build_prompt_from_records(
    records: List[TraceRecord],
    *,
    final_answer_config: Any,
    training_scenario: str,
    max_samples: int = 30,
    max_prompt_chars: int = 60000,
    max_trace_chars: int = 2000,
    max_dynamic_fields: int = 80,
) -> str:
    if training_scenario not in {"no_train", "unlabeled", "labeled"}:
        raise ValueError(f"unsupported training scenario: {training_scenario}")

    schema = {
        "rules": [
            {
                "id": "<string: unique_rule_id>",
                "layer": "llm",
                "label": "<badcase|goodcase>",
                "weight": "<number: 0.0-1.0>",
                "description": "<string: why this rule helps classification>",
                "all": [{"feature": "<feature_name>", "op": "<operator>", "value": "<literal_value>"}],
                "any": [{"feature": "<feature_name>", "op": "<operator>", "value": "<literal_value>"}],
            }
        ],
        "final_answer_config": {
            "top_level_keys": ["<field_name_if_discovered>"],
            "nested_keys": ["<nested_field_name_if_discovered>"],
            "final_answer_items": ["<field_name_if_discovered>:*"],
            "assistant_roles": ["assistant"],
            "assistant_content_keys": ["content"],
            "min_chars": 1,
            "evidence_source": "llm",
            "rationale": "<string: why these fields represent final answers>",
        },
        "proposed_features": [
            {"name": "<new_feature_name>", "description": "<string: only if an important feature is missing>"}
        ],
    }
    scenario_guidance = {
        "no_train": [
            "No training traces are provided.",
            "Generate conservative generic rules from feature definitions and general Agent trace failure modes only.",
            "Do not invent business-field rules or dataset-specific thresholds.",
        ],
        "unlabeled": [
            "Training traces are unlabeled.",
            "Use dataset summary, field frequencies, feature statistics, and diverse samples.",
            "Generate anomaly-style rules. Use lower weights for cohort-relative risk rules.",
        ],
        "labeled": [
            "Training traces are labeled as goodcase or badcase.",
            "Use label contrasts and selected examples from both labels.",
            "Prefer rules that separate badcase from goodcase without memorizing file names, source names, or split names.",
        ],
    }
    payload = (
        {}
        if training_scenario == "no_train"
        else _prompt_payload(
            records,
            final_answer_config=final_answer_config,
            training_scenario=training_scenario,
            max_samples=max_samples,
            max_trace_chars=max_trace_chars,
            max_dynamic_fields=max_dynamic_fields,
        )
    )
    lines = [
        "# LLM Rule Extraction Prompt",
        "",
        "You are generating explainable JSON rules for Agent trace classification.",
        "Return only valid JSON matching the schema. Do not include prose outside JSON.",
        "",
        f"Training scenario: {training_scenario}",
        *[f"- {item}" for item in scenario_guidance[training_scenario]],
        "",
        "Important input design:",
        "- The prompt may omit full traces when the train set is large.",
        "- Values wrapped in <angle_brackets> inside the schema are placeholders; do not copy them into the final JSON.",
        "- Use dataset_summary and label_contrasts before relying on individual sample excerpts.",
        "- In labeled mode, selected_samples is balanced to include goodcase and badcase when available.",
        "- Prefer dynamic field rules such as field_exists:<path>, field_text:<path>, or field_number_mean:<path> when a concrete trace field is informative.",
        "- If a useful signal cannot be expressed with fixed or dynamic field features, put it in proposed_features; proposed_features are not executable in this run.",
        "",
        "Allowed operators:",
        "`==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `regex`, `truthy`, `falsey`",
        "",
        "Feature vocabulary:",
        json.dumps(FEATURE_DESCRIPTIONS, ensure_ascii=False, indent=2),
        "",
        "Required JSON schema:",
        json.dumps(schema, ensure_ascii=False, indent=2),
        "",
        "Compressed training context:",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "",
        "Return JSON now.",
    ]
    prompt = "\n".join(lines)
    if len(prompt) <= max_prompt_chars:
        return prompt
    truncated_payload = dict(payload)
    truncated_payload["selected_samples"] = truncated_payload.get("selected_samples", [])[: max(1, max_samples // 2)]
    truncated_payload["truncation_note"] = f"Prompt exceeded {max_prompt_chars} chars; selected_samples were reduced."
    lines[-3] = json.dumps(truncated_payload, ensure_ascii=False, indent=2)
    prompt = "\n".join(lines)
    return prompt[:max_prompt_chars] + "\n...<prompt truncated>"


def _select_records(args: argparse.Namespace) -> List[TraceRecord]:
    records = load_records(args.trace_path, args.metadata)
    if args.split:
        records = split_records(records, args.split)
        if not records:
            raise ValueError(f"metadata split column has no samples for split: {args.split}")
    return records


def build_prompt(args: argparse.Namespace) -> str:
    records = _select_records(args)
    if args.training_scenario == "labeled":
        records = records_with_labels(records)
    final_answer_config = discover_default_final_answer_config(
        records,
        load_final_answer_config(args.final_answer_config, args.final_answer_item),
    )
    return build_prompt_from_records(
        records,
        final_answer_config=final_answer_config,
        training_scenario=args.training_scenario,
        max_samples=args.max_samples,
        max_prompt_chars=args.max_prompt_chars,
        max_trace_chars=args.max_trace_chars,
        max_dynamic_fields=args.max_dynamic_fields,
    )


def _parse_llm_extra(items: List[str]) -> Dict[str, Any]:
    extra_args: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--llm-extra must use key=value format: {item}")
        key, value = item.split("=", 1)
        extra_args[key.strip()] = value.strip()
    return extra_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or call an LLM prompt for trace sorting rule extraction."
    )
    parser.add_argument("trace_path", help="Training trace JSON file or directory.")
    parser.add_argument("--metadata", help="Optional metadata CSV with name,label,source,split columns.")
    parser.add_argument("--split", help="Optional split value used as training data.")
    parser.add_argument(
        "--training-scenario",
        choices=["no_train", "unlabeled", "labeled"],
        default="unlabeled",
        help="Type of training input shown to the LLM.",
    )
    parser.add_argument("--max-samples", type=int, default=30, help="Maximum representative samples included in the prompt.")
    parser.add_argument("--max-prompt-chars", type=int, default=60000, help="Maximum prompt characters before truncation.")
    parser.add_argument("--max-trace-chars", type=int, default=2000, help="Maximum raw trace excerpt characters per selected sample.")
    parser.add_argument("--max-dynamic-fields", type=int, default=80, help="Maximum dynamic field paths included per selected sample.")
    parser.add_argument("--output", help="Optional prompt Markdown output path.")
    parser.add_argument("--call-llm", action="store_true", help="Call call_llm() after generating the prompt.")
    parser.add_argument("--llm-provider", help="Provider name passed to call_llm().")
    parser.add_argument("--llm-model", help="Model name passed to call_llm().")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Temperature passed to call_llm().")
    parser.add_argument(
        "--llm-extra",
        action="append",
        default=[],
        help="Extra key=value argument passed to call_llm(). Can be repeated.",
    )
    parser.add_argument("--llm-output", help="Optional JSON file for the LLM response.")
    parser.add_argument(
        "--rules-path",
        default=str(DEFAULT_LLM_RULES),
        help="LLM rules JSON to summarize when --llm-output is not provided.",
    )
    parser.add_argument(
        "--report-output",
        default="llm_rule_repoert.md",
        help="Chinese Markdown report describing rules found by the LLM.",
    )
    parser.add_argument("--no-report", action="store_true", help="Do not write the Chinese LLM rule report.")
    parser.add_argument("--final-answer-config", help="Optional JSON config for final answer detection.")
    parser.add_argument(
        "--final-answer-item",
        action="append",
        help="Business-specific final answer key:value pattern. Use * as a wildcard. Can be repeated.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prompt = build_prompt(args)
    if args.output:
        Path(args.output).write_text(prompt + "\n", encoding="utf-8")
        print(f"Wrote prompt: {args.output}")
    else:
        print(prompt)

    llm_output_path = args.llm_output
    if args.call_llm:
        response = call_llm(
            prompt,
            provider=args.llm_provider,
            model=args.llm_model,
            temperature=args.llm_temperature,
            extra_args=_parse_llm_extra(args.llm_extra),
        )
        if not response:
            raise RuntimeError("call_llm() returned empty output. Please implement scripts/llm_rule_prompt.py::call_llm.")
        llm_output_path = llm_output_path or "llm_response.json"
        Path(llm_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(llm_output_path).write_text(response, encoding="utf-8")
        print(f"Wrote LLM response: {llm_output_path}")

    if not args.no_report:
        report_path = write_llm_rule_report(
            args.report_output,
            llm_output_path=llm_output_path,
            rules_path=args.rules_path,
            prompt_path=args.output,
        )
        print(f"Wrote LLM rule report: {report_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from features import discover_default_final_answer_config, extract_features, load_final_answer_config
from trace_io import TraceRecord, load_records, records_with_labels, split_records


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LLM_RULES = SCRIPT_DIR / "rules" / "dynamic" / "llm" / "labeled_rules.json"

FEATURE_DESCRIPTIONS = {
    "parse_error": "JSON parse failed.",
    "is_empty_trace": "Trace object is empty.",
    "has_steps": "Trace contains observable steps, messages, events, or actions.",
    "step_count": "Number of observable steps/actions.",
    "unique_action_count": "Number of unique action names.",
    "unique_action_ratio": "Unique action names divided by step_count.",
    "repeated_action_count": "Total repeated action count.",
    "max_consecutive_same_action": "Longest run of the same consecutive action.",
    "error_count": "Number of error/failure/timeout terms found in text.",
    "has_error_text": "Whether error-like text appears.",
    "empty_result_count": "Number of steps with empty result-like fields.",
    "empty_result_ratio": "Empty result count divided by step_count.",
    "nonempty_result_ratio": "Non-empty result ratio.",
    "has_final_answer": "Whether a final answer/final response is visible.",
    "final_answer_evidence_enabled": "Whether final answer evidence is enabled for this run.",
    "final_answer_evidence_strength": "strong for user-configured fields, medium for default/LLM-discovered fields, none if disabled.",
    "final_answer_evidence_source": "How final answer fields were selected: user, default, llm, or none.",
    "final_answer_adopted_fields": "Comma-separated fields adopted as final answer evidence for this run.",
    "final_answer_chars": "Approximate final answer character count.",
    "final_answer_source": "Where final answer detection matched, such as top_level:final_answer or assistant:content.",
    "text_chars": "Total visible trace text characters.",
    "trace_field_paths": "Newline-separated normalized leaf field paths observed in the trace.",
    "field_exists:<path>": "Dynamic field feature. True when the normalized trace field path exists.",
    "field_count:<path>": "Dynamic field feature. Number of leaf values found at this path.",
    "field_text:<path>": "Dynamic field feature. Joined text values found at this path.",
    "field_nonempty_ratio:<path>": "Dynamic field feature. Non-empty value ratio for this path.",
    "field_number_mean:<path>": "Dynamic field feature. Mean numeric value for this path when all values are numeric.",
    "field_number_min:<path>": "Dynamic field feature. Minimum numeric value for this path when all values are numeric.",
    "field_number_max:<path>": "Dynamic field feature. Maximum numeric value for this path when all values are numeric.",
    "field_bool_true_ratio:<path>": "Dynamic field feature. True ratio for this path when all values are boolean.",
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
        rendered = []
        for condition in conditions:
            rendered.append(
                f"{condition.get('feature')} {condition.get('op', '==')} {condition.get('value')}"
            )
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


def _feature_rows(
    records: List[TraceRecord],
    final_answer_config: Any,
    max_samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records[:max_samples]:
        rows.append(
            {
                "name": record.name,
                "label": record.label,
                "source": record.source,
                "split": record.split,
                "features": extract_features(record, final_answer_config),
            }
        )
    return rows


def build_prompt_from_records(
    records: List[TraceRecord],
    *,
    final_answer_config: Any,
    training_scenario: str,
    max_samples: int = 30,
) -> str:
    if training_scenario not in {"no_train", "unlabeled", "labeled"}:
        raise ValueError(f"unsupported training scenario: {training_scenario}")

    schema = {
        "rules": [
            {
                "id": "short_snake_case_rule_id",
                "layer": "llm",
                "label": "badcase or goodcase",
                "weight": 0.25,
                "description": "Human-readable explanation.",
                "all": [{"feature": "feature_name", "op": ">=", "value": 1}],
                "any": [{"feature": "feature_name", "op": "==", "value": True}],
            }
        ],
        "final_answer_config": {
            "top_level_keys": ["business_result"],
            "nested_keys": ["business_result", "summary_text"],
            "final_answer_items": ["business_result:*", "status: *success*"],
            "assistant_roles": ["assistant"],
            "assistant_content_keys": ["content"],
            "min_chars": 1,
            "evidence_source": "llm",
            "rationale": "Why these fields appear to represent final answers.",
        },
        "proposed_features": [
            {"name": "new_feature_name", "description": "Only if an important feature is missing."}
        ],
    }

    scenario_guidance = {
        "no_train": [
            "No training traces are provided.",
            "Generate conservative generic rules from the feature definitions only.",
            "Do not invent dataset-specific thresholds.",
        ],
        "unlabeled": [
            "Training traces are unlabeled.",
            "Generate anomaly-style rules from feature distributions.",
            "Use dynamic field features when a concrete trace field appears informative.",
            "Use lower weights for cohort-relative risk rules.",
        ],
        "labeled": [
            "Training traces are labeled as goodcase or badcase.",
            "Prefer rules that separate labeled badcase from labeled goodcase.",
            "Actively inspect dynamic field features such as field_text:<path>, field_exists:<path>, and field_number_mean:<path>.",
            "Do not memorize file names, source names, or split names.",
        ],
    }

    rows = _feature_rows(records, final_answer_config, max_samples) if training_scenario != "no_train" else []
    return "\n".join(
        [
            "# LLM Rule Extraction Prompt",
            "",
            "You are generating explainable JSON rules for Agent trace classification.",
            "Return only valid JSON matching the schema below. Do not include prose outside JSON.",
            "",
            f"Training scenario: {training_scenario}",
            *[f"- {item}" for item in scenario_guidance[training_scenario]],
            "",
            "Goal:",
            "- Classify traces as goodcase or badcase.",
            "- Prefer conservative, interpretable rules.",
            "- Use fixed feature names and any dynamic field feature names present in the training feature rows.",
            "- Dynamic field rules are immediately executable if their feature names follow the field_*:<path> convention.",
            "- If a useful signal cannot be expressed with existing fixed or dynamic field features, describe it in `proposed_features`; such proposed features are not executable until implemented.",
            "- Also judge whether these traces expose a business final-answer field. If yes, include `final_answer_config` with `evidence_source: \"llm\"`.",
            "",
            "Allowed operators:",
            "`==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `regex`, `truthy`, `falsey`",
            "",
            "Allowed features:",
            json.dumps(FEATURE_DESCRIPTIONS, ensure_ascii=False, indent=2),
            "",
            "Required JSON schema:",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "",
            "Training feature rows:",
            json.dumps(rows, ensure_ascii=False, indent=2),
            "",
            "Return JSON now.",
        ]
    )


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
    parser.add_argument("--max-samples", type=int, default=30, help="Maximum feature rows included in the prompt.")
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
            raise RuntimeError("call_llm() returned empty output. Please implement scripts/llm_rule_prompt.py::call_llm().")
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

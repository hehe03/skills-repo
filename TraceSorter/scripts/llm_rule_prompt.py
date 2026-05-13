from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from features import discover_default_final_answer_config, extract_features, load_final_answer_config
from trace_io import load_records


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
    "final_answer_evidence_strength": "Evidence strength: strong for user-configured fields, medium for default/LLM-discovered fields, none if disabled.",
    "final_answer_evidence_source": "How final answer fields were selected: user, default, llm, or none.",
    "final_answer_adopted_fields": "Comma-separated fields adopted as final answer evidence for this run.",
    "final_answer_chars": "Approximate final answer character count.",
    "final_answer_source": "Where final answer detection matched, such as top_level:final_answer or assistant:content.",
    "text_chars": "Total visible trace text characters.",
}


def build_prompt(args: argparse.Namespace) -> str:
    records = load_records(args.trace_path, args.metadata)
    final_answer_config = discover_default_final_answer_config(
        records,
        load_final_answer_config(args.final_answer_config, args.final_answer_keys),
    )
    if args.split:
        records = [record for record in records if (record.split or "") == args.split]
    rows: List[Dict[str, Any]] = []
    for record in records[: args.max_samples]:
        features = extract_features(record, final_answer_config)
        rows.append(
            {
                "name": record.name,
                "label": record.label,
                "source": record.source,
                "split": record.split,
                "features": features,
            }
        )

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
        ]
    }
    allowed_features = sorted(FEATURE_DESCRIPTIONS)
    return "\n".join(
        [
            "# LLM Rule Extraction Prompt",
            "",
            "You are generating explainable rules for Agent trace classification.",
            "Return only valid JSON matching the schema below. Do not include prose outside JSON.",
            "",
            "Goal:",
            "- Classify traces as goodcase or badcase.",
            "- Prefer conservative, interpretable rules.",
            "- Use only the allowed feature names unless you explicitly propose a new feature in a separate `proposed_features` array.",
            "- Avoid rules that memorize file names, source names, or split names.",
            "- For labeled samples, prefer rules that separate badcase from goodcase.",
            "- For unlabeled samples, prefer anomaly-style risk rules and mark them with lower weights.",
            "- Also judge whether these traces expose a business final-answer field. If yes, include a `final_answer_config` object with `evidence_source: \"llm\"`, candidate top-level/nested keys, and a short rationale.",
            "",
            "Allowed operators:",
            "`==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `regex`, `truthy`, `falsey`",
            "",
            "Allowed features:",
            json.dumps(FEATURE_DESCRIPTIONS, ensure_ascii=False, indent=2),
            "",
            "Rule JSON schema:",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "",
            "Optional final-answer config schema:",
            json.dumps(
                {
                    "final_answer_config": {
                        "top_level_keys": ["business_result"],
                        "nested_keys": ["business_result", "summary_text"],
                        "assistant_roles": ["assistant"],
                        "assistant_content_keys": ["content"],
                        "min_chars": 1,
                        "evidence_source": "llm",
                        "rationale": "Why these fields appear to represent final answers.",
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Sample feature rows:",
            json.dumps(rows, ensure_ascii=False, indent=2),
            "",
            "Return JSON now.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a prompt that Codex/OpenCode or another LLM can use to propose trace sorting rules."
    )
    parser.add_argument("trace_path", help="Trace JSON file or directory containing JSON trace files.")
    parser.add_argument("--metadata", help="Optional metadata CSV with name,label,source,split columns.")
    parser.add_argument("--split", choices=["train", "test"], help="Optional split filter for examples.")
    parser.add_argument("--max-samples", type=int, default=30, help="Maximum feature rows included in the prompt.")
    parser.add_argument("--output", help="Optional prompt Markdown output path.")
    parser.add_argument(
        "--final-answer-config",
        help="Optional JSON config for business-specific final answer detection.",
    )
    parser.add_argument(
        "--final-answer-keys",
        help="Comma-separated business-specific final answer keys added to top-level and nested detection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompt = build_prompt(args)
    if args.output:
        Path(args.output).write_text(prompt + "\n", encoding="utf-8")
        print(f"Wrote prompt: {args.output}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()

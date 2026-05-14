from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from metrics import confusion_and_scores
from experiment_components.contribution import summarize_result_components, summarize_rule_components


def safe_method_name(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in method).strip("_")


def default_report_path(output_dir: str | Path, method: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{safe_method_name(method)}_{timestamp}.md"


def _rule_summary(rules: Iterable[Dict[str, Any]]) -> List[str]:
    lines = []
    for rule in rules:
        lines.append(
            f"| `{rule.get('id')}` | {rule.get('component', '')} | {rule.get('feature_group', '')} | "
            f"{rule.get('layer', '')} | {rule.get('label', '')} | "
            f"{rule.get('weight', '')} | {rule.get('description', '')} |"
        )
    return lines


def _component_rule_summary(rules: List[Dict[str, Any]]) -> List[str]:
    rows = summarize_rule_components(rules)
    lines = ["## Component Rule Summary", ""]
    lines.append("| component | rules | bad rules | good rules | weight sum | description |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if not rows:
        lines.append("| none | 0 | 0 | 0 | 0 | |")
    for row in rows:
        description = str(row.get("description", "")).replace("|", "\\|")
        lines.append(
            f"| `{row['component']}` | {row['rules']} | {row['bad_rules']} | "
            f"{row['good_rules']} | {round(row['weight_sum'], 4)} | {description} |"
        )
    lines.append("")
    return lines


def _component_hit_summary(results: List[Dict[str, Any]]) -> List[str]:
    rows = summarize_result_components(results)
    lines = ["## Component Hit Summary", ""]
    lines.append("| component | cases hit | hits | bad score | good score | correct cases | incorrect cases |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    if not rows:
        lines.append("| none | 0 | 0 | 0 | 0 | 0 | 0 |")
    for row in rows:
        lines.append(
            f"| `{row['component']}` | {row['cases']} | {row['hits']} | "
            f"{round(row['bad_score'], 4)} | {round(row['good_score'], 4)} | "
            f"{row['correct_cases']} | {row['incorrect_cases']} |"
        )
    lines.append("")
    return lines


def _final_answer_summary(results: List[Dict[str, Any]]) -> List[str]:
    if not results:
        return ["- Final answer policy: no samples", ""]
    policies: Dict[str, int] = {}
    for row in results:
        policy = (
            f"{row.get('final_answer_evidence_source', 'none')}/"
            f"{row.get('final_answer_evidence_strength', 'none')}:"
            f"{row.get('final_answer_adopted_fields') or 'none'}"
        )
        policies[policy] = policies.get(policy, 0) + 1
    lines = ["## Method Notes", ""]
    lines.append("Final answer policy:")
    lines.append("| policy | samples |")
    lines.append("|---|---:|")
    for policy, count in sorted(policies.items()):
        lines.append(f"| `{policy}` | {count} |")
    lines.append("")
    return lines


def write_report(
    path: str | Path,
    *,
    method: str,
    trace_path: str,
    metadata_path: str | None,
    rules: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    max_rows: int = 200,
    notes: List[str] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = confusion_and_scores(results)
    lines: List[str] = []
    lines.append(f"# Trace Sorter Experiment: {method}")
    lines.append("")
    lines.append(f"- Trace path: `{trace_path}`")
    lines.append(f"- Metadata: `{metadata_path}`" if metadata_path else "- Metadata: none")
    lines.append(f"- Samples evaluated: {len(results)}")
    lines.append(f"- Rules loaded: {len(rules)}")
    lines.append("")
    if notes:
        lines.append("## Run Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    lines.extend(_final_answer_summary(results))
    lines.extend(_component_rule_summary(rules))
    lines.extend(_component_hit_summary(results))
    if metrics["count"]:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        for key in ("accuracy", "precision", "recall", "f1"):
            lines.append(f"| {key} | {metrics[key]} |")
        lines.append("")
        lines.append("| confusion | count |")
        lines.append("|---|---:|")
        for key in ("tp", "fp", "tn", "fn"):
            lines.append(f"| {key} | {metrics[key]} |")
        lines.append("")
    lines.append("## Rules Used")
    lines.append("")
    lines.append("| id | component | feature group | layer | label | weight | description |")
    lines.append("|---|---|---|---|---|---:|---|")
    lines.extend(_rule_summary(rules) or ["| none | | | | |"])
    lines.append("")
    lines.append("## Predictions")
    lines.append("")
    lines.append("| name | label | predicted | bad_score | good_score | reason |")
    lines.append("|---|---|---|---:|---:|---|")
    for row in results[:max_rows]:
        reason = str(row.get("reason", "")).replace("|", "\\|")
        lines.append(
            f"| `{row.get('name')}` | {row.get('label') or ''} | {row.get('predicted_label')} | "
            f"{row.get('bad_score')} | {row.get('good_score')} | {reason} |"
        )
    if len(results) > max_rows:
        lines.append(f"| ... | ... | ... | ... | ... | truncated at {max_rows} rows |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output

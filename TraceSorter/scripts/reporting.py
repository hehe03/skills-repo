from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from metrics import confusion_and_scores


def safe_method_name(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in method).strip("_")


def default_report_path(output_dir: str | Path, method: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{safe_method_name(method)}_{timestamp}.md"


def _rule_summary(rules: Iterable[Dict[str, Any]]) -> List[str]:
    lines = []
    for rule in rules:
        lines.append(
            f"| `{rule.get('id')}` | {rule.get('layer', '')} | {rule.get('label', '')} | "
            f"{rule.get('weight', '')} | {rule.get('description', '')} |"
        )
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
    lines.append("| id | layer | label | weight | description |")
    lines.append("|---|---|---|---:|---|")
    lines.extend(_rule_summary(rules) or ["| none | | | | |"])
    lines.append("")
    lines.append("## Predictions")
    lines.append("")
    lines.append(
        "| name | label | predicted | bad_score | good_score | "
        "final_answer_policy | final_answer_source | reason |"
    )
    lines.append("|---|---|---|---:|---:|---|---|---|")
    for row in results[:max_rows]:
        reason = str(row.get("reason", "")).replace("|", "\\|")
        final_policy = (
            f"{row.get('final_answer_evidence_source', 'none')}/"
            f"{row.get('final_answer_evidence_strength', 'none')}:"
            f"{row.get('final_answer_adopted_fields') or 'none'}"
        )
        lines.append(
            f"| `{row.get('name')}` | {row.get('label') or ''} | {row.get('predicted_label')} | "
            f"{row.get('bad_score')} | {row.get('good_score')} | {final_policy} | "
            f"{row.get('final_answer_source') or 'none'} | {reason} |"
        )
    if len(results) > max_rows:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | truncated at {max_rows} rows |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output

#!/usr/bin/env python3
"""Summarize JSONL traces for skill-ops projects."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize skill trace JSONL files.")
    parser.add_argument("trace_path", help="Path to JSONL trace file")
    parser.add_argument("--output-path", help="Markdown output path")
    return parser.parse_args()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_path)
    output_path = Path(args.output_path) if args.output_path else trace_path.with_name(f"{trace_path.stem}_summary.md")

    stage_counter = Counter()
    label_counter = Counter()
    candidate_counter = Counter()
    key_counter = Counter()
    row_count = 0

    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row_count += 1
            obj = json.loads(line)
            for key in obj.keys():
                key_counter[key] += 1
            stage_counter[str(obj.get("stage", obj.get("stage1_detected", "unknown")))] += 1
            for label in as_list(obj.get("predicted_labels")) + as_list(obj.get("confirmed_labels")):
                label_counter[str(label)] += 1
            for label in as_list(obj.get("candidate_labels")):
                candidate_counter[str(label)] += 1

    lines = [
        f"# Trace Summary: {trace_path.name}",
        "",
        f"- rows: {row_count}",
        "",
        "## Top-Level Keys",
    ]
    for key, count in key_counter.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Stage Distribution"])
    for key, count in stage_counter.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Confirmed Labels"])
    if label_counter:
        for key, count in label_counter.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Candidate Labels"])
    if candidate_counter:
        for key, count in candidate_counter.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"trace: {trace_path}")
    print(f"summary: {output_path}")
    print(f"rows: {row_count}")


if __name__ == "__main__":
    main()

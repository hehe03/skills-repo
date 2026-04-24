#!/usr/bin/env python3
"""Generic label-vs-prediction audit for skill optimization projects."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(path, index=False)


def split_labels(value: Any, separator: str, null_label: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == null_label:
        return []
    return [item.strip() for item in text.split(separator) if item.strip()]


def join_labels(labels: list[str], separator: str, null_label: str) -> str:
    return separator.join(labels) if labels else null_label


def classify_gap(
    human_set: set[str],
    predicted_set: set[str],
    trace_supported: set[str],
) -> str:
    if human_set == predicted_set:
        return "一致"
    if not human_set and predicted_set:
        return "疑似人工漏标"
    if human_set and not predicted_set:
        return "疑似人工误标" if any(label not in trace_supported for label in human_set) else "规则漏判"
    unsupported_human = [label for label in (human_set - predicted_set) if label not in trace_supported]
    if unsupported_human:
        return "疑似人工误标"
    if predicted_set - human_set:
        return "疑似人工漏标"
    return "混合差异"


def explain_gap(
    human_set: set[str],
    predicted_set: set[str],
    trace_supported: set[str],
    separator: str,
    null_label: str,
) -> str:
    only_human = sorted(human_set - predicted_set)
    only_model = sorted(predicted_set - human_set)
    notes: list[str] = []
    if only_human:
        unsupported = [label for label in only_human if label not in trace_supported]
        if unsupported:
            notes.append(f"人工多标但缺少支持证据: {join_labels(unsupported, separator, null_label)}")
        else:
            notes.append(f"人工包含但预测未输出: {join_labels(only_human, separator, null_label)}")
    if only_model:
        notes.append(f"预测新增命中: {join_labels(only_model, separator, null_label)}")
    return "；".join(notes) if notes else "一致"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit skill outputs against noisy labels.")
    parser.add_argument("input_path", help="CSV or Excel file with labels and predictions")
    parser.add_argument("--human-col", required=True, help="Human label column")
    parser.add_argument("--pred-col", required=True, help="Prediction column")
    parser.add_argument("--id-col", help="Optional stable sample id column")
    parser.add_argument("--trace-col", help="Optional trace summary column")
    parser.add_argument(
        "--candidate-col",
        help="Optional candidate/low-confidence label column; used for context but not main match",
    )
    parser.add_argument(
        "--low-confidence-labels",
        default="",
        help="Comma-separated labels that should not affect the main agreement score",
    )
    parser.add_argument("--separator", default="、", help="Multilabel separator")
    parser.add_argument("--null-label", default="未匹配到异常分支", help="Empty-label placeholder")
    parser.add_argument(
        "--keep-cols",
        default="",
        help="Comma-separated extra source columns to keep near the front of the audit output",
    )
    parser.add_argument("--output-path", help="Audit workbook path")
    parser.add_argument("--report-path", help="Markdown summary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    df = read_table(input_path)

    low_conf = {item.strip() for item in args.low_confidence_labels.split(",") if item.strip()}
    keep_cols = [item.strip() for item in args.keep_cols.split(",") if item.strip()]

    output_path = Path(args.output_path) if args.output_path else input_path.with_name(f"{input_path.stem}_audit.xlsx")
    report_path = Path(args.report_path) if args.report_path else input_path.with_name(f"{input_path.stem}_audit.md")

    ids: list[Any] = []
    human_core_list: list[str] = []
    pred_list: list[str] = []
    gap_types: list[str] = []
    gap_notes: list[str] = []
    candidate_context: list[str] = []
    primary_match: list[bool] = []
    suspected_label_issue: list[bool] = []
    rule_counter = Counter()
    type_counter = Counter()

    for idx, row in df.iterrows():
        row_id = row[args.id_col] if args.id_col and args.id_col in df.columns else idx
        human_all = set(split_labels(row[args.human_col], args.separator, args.null_label))
        pred_all = set(split_labels(row[args.pred_col], args.separator, args.null_label))
        human_core = {label for label in human_all if label not in low_conf}
        pred_core = {label for label in pred_all if label not in low_conf}
        trace_supported = pred_all.copy()
        if args.candidate_col and args.candidate_col in df.columns:
            trace_supported |= set(split_labels(row[args.candidate_col], args.separator, args.null_label))

        gap_type = classify_gap(human_core, pred_core, trace_supported)
        gap_note = explain_gap(human_core, pred_core, trace_supported, args.separator, args.null_label)
        only_low_conf_gap = human_all != human_core and human_core == pred_core
        if only_low_conf_gap:
            gap_type = "低置信规则差异"
            low_conf_gap = sorted(human_all - human_core)
            gap_note = f"人工仅多出低置信规则: {join_labels(low_conf_gap, args.separator, args.null_label)}"

        matched = gap_type in {"一致", "低置信规则差异"}
        issue_flag = gap_type in {"疑似人工误标", "疑似人工漏标", "低置信规则差异"}

        ids.append(row_id)
        human_core_list.append(join_labels(sorted(human_core), args.separator, args.null_label))
        pred_list.append(join_labels(sorted(pred_core), args.separator, args.null_label))
        gap_types.append(gap_type)
        gap_notes.append(gap_note)
        candidate_context.append(row[args.candidate_col] if args.candidate_col and args.candidate_col in df.columns else "")
        primary_match.append(matched)
        suspected_label_issue.append(issue_flag)
        type_counter[gap_type] += 1

        for label in human_core:
            rule_counter[(label, "human")] += 1
        for label in pred_core:
            rule_counter[(label, "pred")] += 1

    insert_cols = {}
    insert_cols["sample_id"] = ids
    insert_cols["人工高置信标签"] = human_core_list
    insert_cols["预测高置信标签"] = pred_list
    if args.candidate_col and args.candidate_col in df.columns:
        insert_cols["候选标签上下文"] = candidate_context
    insert_cols["主口径一致"] = primary_match
    insert_cols["差异类型"] = gap_types
    insert_cols["差异说明"] = gap_notes
    insert_cols["疑似标签问题"] = suspected_label_issue
    if args.trace_col and args.trace_col in df.columns:
        insert_cols["trace摘要"] = df[args.trace_col]

    result = df.copy()
    for col_name, values in reversed(list(insert_cols.items())):
        if col_name in result.columns:
            result = result.drop(columns=[col_name])
        result.insert(0, col_name, values)

    if keep_cols:
        ordered = [col for col in insert_cols.keys()]
        ordered += [col for col in keep_cols if col in result.columns and col not in ordered]
        ordered += [col for col in result.columns if col not in ordered]
        result = result[ordered]

    write_table(result, output_path)

    total = len(result)
    matched = sum(primary_match)
    labels = sorted({label for label, _kind in rule_counter.keys()})
    lines = [
        f"# Audit Report: {input_path.name}",
        "",
        f"- total_rows: {total}",
        f"- primary_matches: {matched}",
        f"- primary_match_rate: {matched / total:.2%}",
        f"- low_confidence_labels: {', '.join(sorted(low_conf)) if low_conf else '(none)'}",
        "",
        "## Discrepancy Types",
    ]
    for gap_type, count in type_counter.most_common():
        lines.append(f"- {gap_type}: {count}")
    lines.extend(["", "## Label Counts"])
    for label in labels:
        lines.append(f"- {label}: human={rule_counter[(label, 'human')]} pred={rule_counter[(label, 'pred')]}")

    suspect_df = result[result["疑似标签问题"]]
    lines.extend(["", "## Suspected Label-Issue Rows"])
    if suspect_df.empty:
        lines.append("- none")
    else:
        sample_field = "sample_id"
        for _, row in suspect_df.iterrows():
            lines.append(
                f"- {sample_field}={row[sample_field]} | type={row['差异类型']} | human={row['人工高置信标签']} | pred={row['预测高置信标签']} | note={row['差异说明']}"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"report: {report_path}")
    print(f"primary_match_rate: {matched / total:.2%}")


if __name__ == "__main__":
    main()

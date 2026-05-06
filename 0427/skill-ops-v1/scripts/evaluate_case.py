#!/usr/bin/env python3
"""通用案例评估脚本。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd


def split_labels(value: object) -> Set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip().replace(',', '、')
    if not text or text == '- 未匹配到分支':
        return set()
    return {item.strip() for item in text.split('、') if item.strip()}


def label_signature(value: object) -> frozenset[str]:
    return frozenset(split_labels(value))


def load_trace_summary(trace_path: Path) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    with trace_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for rule_name, rule_trace in record.get('trace', {}).items():
                if rule_trace.get('matched'):
                    counter[f'命中:{rule_name}'] += 1
    return dict(counter)


def normalize_key_frame(df: pd.DataFrame, ignore_cols: Sequence[str]) -> pd.DataFrame:
    work = df.drop(columns=[col for col in ignore_cols if col in df.columns], errors='ignore').copy()
    for column in work.columns:
        work[column] = work[column].apply(
            lambda value: '__NaN__' if pd.isna(value) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
    return work


def build_report(
    df: pd.DataFrame,
    *,
    answer_col: str,
    prediction_col: str,
    trace_summary: Dict[str, int] | None,
) -> str:
    exact_match = int(
        sum(
            label_signature(answer_value) == label_signature(prediction_value)
            for answer_value, prediction_value in zip(df[answer_col], df[prediction_col])
        )
    )
    total = int(len(df))

    miss_counter: Counter[str] = Counter()
    extra_counter: Counter[str] = Counter()

    for _, row in df.iterrows():
        answer_labels = split_labels(row[answer_col])
        prediction_labels = split_labels(row[prediction_col])
        for label in sorted(answer_labels - prediction_labels):
            miss_counter[label] += 1
        for label in sorted(prediction_labels - answer_labels):
            extra_counter[label] += 1

    lines: List[str] = []
    lines.append('# 案例评估报告')
    lines.append('')
    lines.append('## 总览')
    lines.append('')
    lines.append(f'- 总样本数：{total}')
    lines.append(f'- 精确一致样本数：{exact_match}')
    lines.append(f'- 精确一致率：{exact_match / total:.2%}' if total else '- 精确一致率：N/A')
    lines.append(f'- 错例数：{total - exact_match}')
    lines.append('')

    lines.append('## 漏报分布')
    lines.append('')
    if miss_counter:
        for label, count in miss_counter.most_common():
            lines.append(f'- {label}：{count}')
    else:
        lines.append('- 无')
    lines.append('')

    lines.append('## 误报分布')
    lines.append('')
    if extra_counter:
        for label, count in extra_counter.most_common():
            lines.append(f'- {label}：{count}')
    else:
        lines.append('- 无')
    lines.append('')

    if trace_summary:
        lines.append('## Trace 命中概览')
        lines.append('')
        for key, count in sorted(trace_summary.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f'- {key}：{count}')
        lines.append('')

    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='评估规则型 skill 案例')
    parser.add_argument('excel_path', help='案例 Excel 路径')
    parser.add_argument('--answer-col', default='answer', help='人工标注列名')
    parser.add_argument('--prediction-col', default='L2分类结果', help='预测列名')
    parser.add_argument('--trace-path', help='可选 trace.jsonl 路径')
    parser.add_argument('--report-path', help='输出 Markdown 报告路径')
    parser.add_argument(
        '--ignore-cols',
        nargs='*',
        default=['references', 'answer', 'L2分类结果'],
        help='重复冲突分组时忽略的列',
    )
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    df = pd.read_excel(excel_path)

    trace_summary = load_trace_summary(Path(args.trace_path)) if args.trace_path else None
    report = build_report(
        df,
        answer_col=args.answer_col,
        prediction_col=args.prediction_col,
        trace_summary=trace_summary,
    )

    key_frame = normalize_key_frame(df, args.ignore_cols)
    grouped = key_frame.groupby(list(key_frame.columns), dropna=False).size()
    duplicate_groups = int((grouped > 1).sum())

    answer_sets = df[args.answer_col].apply(label_signature)
    prediction_sets = df[args.prediction_col].apply(label_signature)
    report += '\n## 重复样本概览\n\n'
    report += f"- 忽略列 {', '.join(args.ignore_cols)} 后的重复样本组数：{duplicate_groups}\n"
    report += f'- 按标签集合比较的一致样本数：{int((answer_sets == prediction_sets).sum())}\n'

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding='utf-8')
    else:
        print(report)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""给规则型案例中的疑似错标样本排序。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def split_labels(value: object) -> Set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip().replace(',', '、')
    if not text or text == '- 未匹配到分支':
        return set()
    return {item.strip() for item in text.split('、') if item.strip()}


def build_duplicate_conflicts(df: pd.DataFrame, ignore_cols: List[str], answer_col: str) -> Dict[int, int]:
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    work = df[feature_cols + [answer_col]].copy()
    for column in feature_cols:
        work[column] = work[column].apply(lambda value: '__NaN__' if pd.isna(value) else json.dumps(value, ensure_ascii=False, sort_keys=True))

    grouped = work.groupby(feature_cols, dropna=False)[answer_col].agg(lambda s: {str(v) for v in s})
    conflict_map: Dict[tuple, int] = {}
    for key, answers in grouped.items():
        if len(answers) > 1:
            conflict_map[key if isinstance(key, tuple) else (key,)] = len(answers)

    row_scores: Dict[int, int] = {}
    if not conflict_map:
        return row_scores

    for idx, row in work.iterrows():
        key = tuple(row[col] for col in feature_cols)
        if key in conflict_map:
            row_scores[idx] = conflict_map[key]
    return row_scores


def load_trace_target_matches(trace_path: Path, target_label: str) -> Dict[int, bool]:
    result: Dict[int, bool] = {}
    with trace_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rule_trace = record.get('trace', {}).get(target_label, {})
            result[int(record['row_index'])] = bool(rule_trace.get('matched'))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='给疑似错标样本排序')
    parser.add_argument('excel_path', help='案例 Excel 路径')
    parser.add_argument('--report-path', required=True, help='输出 Markdown 报告路径')
    parser.add_argument('--answer-col', default='answer')
    parser.add_argument('--prediction-col', default='L2分类结果')
    parser.add_argument('--target-label', default='替代交付异常')
    parser.add_argument('--trace-path')
    parser.add_argument('--ignore-cols', nargs='*', default=['references', 'answer', 'L2分类结果'])
    parser.add_argument('--pattern-col', help='布尔模式列名，可选')
    args = parser.parse_args()

    df = pd.read_excel(Path(args.excel_path))
    duplicate_conflicts = build_duplicate_conflicts(df, args.ignore_cols, args.answer_col)
    trace_target = load_trace_target_matches(Path(args.trace_path), args.target_label) if args.trace_path else {}

    rows: List[dict] = []
    for idx, row in df.iterrows():
        answer_labels = split_labels(row[args.answer_col])
        prediction_labels = split_labels(row[args.prediction_col])
        if answer_labels == prediction_labels:
            continue

        score = 0
        reasons: List[str] = []
        if idx in duplicate_conflicts:
            score += 4
            reasons.append(f'相同输入存在 {duplicate_conflicts[idx]} 种不同标注')
        if args.target_label in answer_labels and args.target_label not in prediction_labels:
            score += 3
            reasons.append(f'标注包含{args.target_label}但预测未命中')
        if args.target_label not in answer_labels and args.target_label in prediction_labels:
            score += 2
            reasons.append(f'预测包含{args.target_label}但标注未包含')
        if idx in trace_target and args.target_label in answer_labels and not trace_target[idx]:
            score += 3
            reasons.append(f'trace显示{args.target_label}规则未命中')
        if args.pattern_col and args.pattern_col in df.columns and bool(row[args.pattern_col]):
            score += 2
            reasons.append(f'满足高风险模式:{args.pattern_col}')

        rows.append({
            'row_index': idx,
            'score': score,
            'answer': row[args.answer_col],
            'prediction': row[args.prediction_col],
            'reasons': '；'.join(reasons) if reasons else '普通错例',
        })

    ranked = sorted(rows, key=lambda item: (-item['score'], item['row_index']))

    lines: List[str] = []
    lines.append('# 标注疑点排序报告')
    lines.append('')
    lines.append(f'- 目标标签：{args.target_label}')
    lines.append(f'- 错例总数：{len(ranked)}')
    lines.append('')
    lines.append('## Top 疑点样本')
    lines.append('')
    for item in ranked[:20]:
        lines.append(f"- row {item['row_index']} | score={item['score']} | answer={item['answer']} | prediction={item['prediction']} | {item['reasons']}")

    Path(args.report_path).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

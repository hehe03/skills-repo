#!/usr/bin/env python3
"""挖掘高频标签冲突模式。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


def split_labels(value: object) -> Set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip().replace(',', '、')
    if not text or text == '- 未匹配到分支':
        return set()
    return {item.strip() for item in text.split('、') if item.strip()}


def is_empty_list(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() == '[]'


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def eval_expr(expr: str, row: pd.Series) -> Any:
    env = {
        'col': lambda name: row.get(name),
        'num': lambda name: num(row.get(name)),
        'text': lambda name: '' if pd.isna(row.get(name)) else str(row.get(name)),
        'is_empty_list': is_empty_list,
        'isna': pd.isna,
    }
    return eval(expr, {'__builtins__': {}}, env)


def main() -> None:
    parser = argparse.ArgumentParser(description='挖掘高频标签冲突模式')
    parser.add_argument('excel_path', help='案例 Excel 路径')
    parser.add_argument('--report-path', required=True, help='输出 Markdown 报告路径')
    parser.add_argument('--answer-col', default='answer')
    parser.add_argument('--prediction-col', default='L2分类结果')
    parser.add_argument('--target-label', default='替代交付异常')
    parser.add_argument('--derive', action='append', default=[], help='派生列，格式为 名称=表达式')
    args = parser.parse_args()

    df = pd.read_excel(Path(args.excel_path))
    derived_cols: List[str] = []
    for item in args.derive:
        name, expr = item.split('=', 1)
        name = name.strip()
        expr = expr.strip()
        df[name] = df.apply(lambda row: eval_expr(expr, row), axis=1)
        derived_cols.append(name)

    answer_has = df[args.answer_col].apply(lambda value: args.target_label in split_labels(value))
    pred_has = df[args.prediction_col].apply(lambda value: args.target_label in split_labels(value))

    lines: List[str] = []
    lines.append('# 高频冲突模式报告')
    lines.append('')
    lines.append(f'- 目标标签：{args.target_label}')
    lines.append(f'- 派生模式列：{", ".join(derived_cols) if derived_cols else "无"}')
    lines.append('')

    if derived_cols:
        grouped = df.groupby(derived_cols, dropna=False).size().reset_index(name='support')
        grouped['answer_positive'] = grouped[derived_cols].apply(
            lambda r: int((answer_has & pd.Series([True] * len(df), index=df.index)).sum()), axis=1
        )

        pattern_rows: List[Dict[str, Any]] = []
        for _, group_row in grouped.iterrows():
            mask = pd.Series(True, index=df.index)
            for col_name in derived_cols:
                mask &= df[col_name].eq(group_row[col_name])
            support = int(mask.sum())
            answer_pos = int(answer_has[mask].sum())
            pred_pos = int(pred_has[mask].sum())
            conflict_gap = answer_pos - pred_pos
            pattern_rows.append({
                'pattern': ', '.join(f'{col}={group_row[col]}' for col in derived_cols),
                'support': support,
                'answer_pos': answer_pos,
                'pred_pos': pred_pos,
                'gap': conflict_gap,
            })

        pattern_rows.sort(key=lambda item: (-abs(item['gap']), -item['support'], item['pattern']))
        lines.append('## 冲突模式')
        lines.append('')
        for item in pattern_rows[:20]:
            lines.append(f"- {item['pattern']} | support={item['support']} | answer_pos={item['answer_pos']} | pred_pos={item['pred_pos']} | gap={item['gap']}")
    else:
        lines.append('未提供派生模式列。')

    Path(args.report_path).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

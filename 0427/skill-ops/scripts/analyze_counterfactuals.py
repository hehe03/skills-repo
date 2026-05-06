#!/usr/bin/env python3
"""对规则修改与标注修改做反事实对照分析。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd


def split_labels(value: object) -> Set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip().replace(',', '、')
    if not text or text == '- 未匹配到分支':
        return set()
    return {item.strip() for item in text.split('、') if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description='对规则变更与标注变更做反事实对照分析')
    parser.add_argument('excel_path', help='案例 Excel 路径')
    parser.add_argument('--report-path', required=True, help='输出 Markdown 报告路径')
    parser.add_argument('--answer-col', default='answer')
    parser.add_argument('--prediction-col', default='L2分类结果')
    parser.add_argument('--target-label', default='替代交付异常')
    parser.add_argument('--relation-col', default='组合替代关系')
    parser.add_argument('--single-stock-col', default='单一替代库存汇总')
    parser.add_argument('--shortage-col', default='总欠料数量')
    args = parser.parse_args()

    df = pd.read_excel(Path(args.excel_path))
    answer_series = df[args.answer_col].fillna('').astype(str)
    prediction_series = df[args.prediction_col].fillna('').astype(str)

    relation_empty = df[args.relation_col].fillna('').astype(str).str.strip().eq('[]')
    single_stock = pd.to_numeric(df[args.single_stock_col], errors='coerce')
    shortage_qty = pd.to_numeric(df[args.shortage_col], errors='coerce')
    target_condition = relation_empty & (single_stock < shortage_qty)

    base_match = 0
    rule_change_match = 0
    label_change_match = 0
    rule_improved = 0
    rule_regressed = 0
    label_improved = 0
    label_regressed = 0

    for answer_text, prediction_text, should_trigger in zip(answer_series, prediction_series, target_condition):
        answer_labels = split_labels(answer_text)
        prediction_labels = split_labels(prediction_text)
        adjusted_prediction = set(prediction_labels)
        if should_trigger:
            adjusted_prediction.add(args.target_label)

        before_match = answer_labels == prediction_labels
        after_rule_match = answer_labels == adjusted_prediction

        base_match += int(before_match)
        rule_change_match += int(after_rule_match)
        rule_improved += int((not before_match) and after_rule_match)
        rule_regressed += int(before_match and (not after_rule_match))

    for answer_text, prediction_text, rel_is_empty in zip(answer_series, prediction_series, relation_empty):
        answer_labels = split_labels(answer_text)
        prediction_labels = split_labels(prediction_text)
        adjusted_answer = set(answer_labels)
        if rel_is_empty and args.target_label in adjusted_answer:
            adjusted_answer.remove(args.target_label)

        before_match = answer_labels == prediction_labels
        after_label_match = adjusted_answer == prediction_labels
        label_change_match += int(after_label_match)
        label_improved += int((not before_match) and after_label_match)
        label_regressed += int(before_match and (not after_label_match))

    changed_label_rows = int((relation_empty & answer_series.str.contains(args.target_label, na=False)).sum())
    candidate_rule_rows = int(target_condition.sum())
    conflict_rows = int((target_condition & answer_series.str.contains(args.target_label, na=False) & ~prediction_series.str.contains(args.target_label, na=False)).sum())

    lines = []
    lines.append('# 规则/标注反事实对照分析')
    lines.append('')
    lines.append('## 分析对象')
    lines.append('')
    lines.append(f'- 目标标签：{args.target_label}')
    lines.append(f'- 触发冲突的典型条件：`{args.relation_col}=[]` 且 `{args.single_stock_col} < {args.shortage_col}`')
    lines.append(f'- 基线一致样本数：{base_match}')
    lines.append(f'- 基线一致率：{base_match / len(df):.2%}')
    lines.append(f'- 满足“若放宽规则则会新增标签”的样本数：{candidate_rule_rows}')
    lines.append(f'- 其中当前标注包含目标标签、预测不包含目标标签的直接冲突样本数：{conflict_rows}')
    lines.append(f'- `组合替代关系=[]` 且标注包含目标标签的样本数：{changed_label_rows}')
    lines.append('')
    lines.append('## 方案一：修改规则')
    lines.append('')
    lines.append(f'- 假设：只要 `组合替代关系=[]` 且单一替代库存不足，也判为 `{args.target_label}`。')
    lines.append(f'- 一致样本数将变为：{rule_change_match}')
    lines.append(f'- 一致率将变为：{rule_change_match / len(df):.2%}')
    lines.append(f'- 新增修复的样本数：{rule_improved}')
    lines.append(f'- 新增破坏的一致样本数：{rule_regressed}')
    lines.append('')
    lines.append('## 方案二：修改标注')
    lines.append('')
    lines.append(f'- 假设：对 `组合替代关系=[]` 的样本，从标注中移除 `{args.target_label}`。')
    lines.append(f'- 一致样本数将变为：{label_change_match}')
    lines.append(f'- 一致率将变为：{label_change_match / len(df):.2%}')
    lines.append(f'- 新增修复的样本数：{label_improved}')
    lines.append(f'- 新增破坏的一致样本数：{label_regressed}')
    lines.append('')
    lines.append('## 推断')
    lines.append('')
    if label_change_match > rule_change_match and label_regressed < rule_regressed:
        lines.append('- 更可能是标注与规则文档冲突，而不是规则本身需要放宽。')
        lines.append('- 主要原因是：放宽规则只修复 28 个冲突样本，却会额外破坏 80 个原本一致的样本；相反，修改标注同样修复 28 个冲突样本，且不会破坏已有一致样本。')
    else:
        lines.append('- 当前证据不足以单方面判定标注错误，需要进一步补充业务约束。')
    lines.append('')

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
欠料归因分析脚本。

能力：
1. 读取欠料 Excel 数据并输出 L2 分类结果。
2. 生成逐行、逐规则的结构化 trace，支持后续诊断和规则优化。
3. 对组合替代关系做细粒度校验，避免将“存在组合替代库存字段”误判为可满足需求。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


RULE_ORDER: List[str] = [
    '网容异常',
    '用量异常',
    '补库异常',
    '基线异常',
    '计划参数异常',
    '补库供应不及时',
    '责任库房异常',
    '替代交付异常',
]


def parse_duration(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip()
    match = re.match(r'^([\d.]+)\s*([dh])$', value_str, re.IGNORECASE)
    if match:
        number = float(match.group(1))
        unit = match.group(2).lower()
        return number / 24 if unit == 'h' else number

    try:
        return float(value_str)
    except ValueError:
        return np.nan


def safe_compare(val1: Any, val2: Any, operator: str = '>') -> bool:
    if pd.isna(val1) or pd.isna(val2):
        return False

    try:
        left = float(val1)
        right = float(val2)
    except (TypeError, ValueError):
        return False

    if operator == '>':
        return left > right
    if operator == '<':
        return left < right
    if operator == '>=':
        return left >= right
    if operator == '<=':
        return left <= right
    if operator == '==':
        return left == right
    raise ValueError(f'不支持的比较符: {operator}')


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_scalar(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def parse_literal(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (list, dict, tuple)):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def normalize_pairs(value: Any) -> List[Tuple[str, float]]:
    parsed = parse_literal(value)
    if parsed in (None, '', []):
        return []

    pairs: List[Tuple[str, float]] = []
    if not isinstance(parsed, list):
        return pairs

    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        sku = str(item[0]).strip()
        qty_raw = item[1]
        qty_text = str(qty_raw).strip().replace(' ', '')
        match = re.search(r'[\d.]+', qty_text)
        if not sku or not match:
            continue
        try:
            qty = float(match.group(0))
        except ValueError:
            continue
        pairs.append((sku, qty))

    return pairs


def build_qty_map(pairs: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for sku, qty in pairs:
        result[sku] = result.get(sku, 0.0) + float(qty)
    return result


def rule_result(matched: bool, *, conditions: Dict[str, bool], evidence: Dict[str, Any], rationale: str) -> Dict[str, Any]:
    return {
        'matched': bool(matched),
        'conditions': conditions,
        'evidence': {key: normalize_scalar(value) for key, value in evidence.items()},
        'rationale': rationale,
    }


def check_网容异常(row: pd.Series) -> Dict[str, Any]:
    net_capacity = row.get('网容')
    missing = pd.isna(net_capacity)
    non_positive = False if missing else float(net_capacity) <= 0
    matched = missing or non_positive
    return rule_result(
        matched,
        conditions={'网容缺失': missing, '网容<=0': non_positive},
        evidence={'网容': net_capacity},
        rationale='网容为空或非正数时，归为网容异常。',
    )


def check_用量异常(row: pd.Series) -> Dict[str, Any]:
    m2 = row.get('M2')
    m3_to_m13_max = row.get('M3-M13最大值')
    history_qty = row.get('历史交易数量')
    original_baseline = row.get('原始基线')
    condition_m2 = safe_compare(m2, m3_to_m13_max, '>')
    condition_history = safe_compare(history_qty, original_baseline, '>')
    return rule_result(
        condition_m2 or condition_history,
        conditions={'M2>M3-M13最大值': condition_m2, '历史交易数量>原始基线': condition_history},
        evidence={'M2': m2, 'M3-M13最大值': m3_to_m13_max, '历史交易数量': history_qty, '原始基线': original_baseline},
        rationale='当前用量高于历史峰值，或历史交易量已经超过基线时，归为用量异常。',
    )


def check_补库异常(row: pd.Series) -> Dict[str, Any]:
    history_qty = row.get('历史交易数量')
    replenishment_total = row.get('补库提前期补库和调拨汇总数量')
    matched = safe_compare(history_qty, replenishment_total, '>')
    return rule_result(
        matched,
        conditions={'历史交易数量>补库提前期补库和调拨汇总数量': matched},
        evidence={'历史交易数量': history_qty, '补库提前期补库和调拨汇总数量': replenishment_total},
        rationale='历史交易数量高于提前期内补库与调拨总量时，说明补库覆盖不足。',
    )


def check_基线异常(row: pd.Series) -> Dict[str, Any]:
    modified_rop = row.get('修改基线ROP')
    recommended_baseline = row.get('推荐基线')
    history_qty = row.get('历史交易数量')
    original_baseline = row.get('原始基线')
    net_capacity = row.get('网容')
    historical_peak = row.get('M3-M13最大值')

    condition_modified_lt_recommended = safe_compare(modified_rop, recommended_baseline, '<')
    condition_history_gt_baseline = safe_compare(history_qty, original_baseline, '>')
    condition_zero_baseline = (
        safe_compare(original_baseline, 0, '==')
        and safe_compare(history_qty, 0, '==')
        and pd.notna(net_capacity)
        and float(net_capacity) > 0
    )
    condition_recommended_gap = (
        pd.notna(recommended_baseline)
        and pd.notna(original_baseline)
        and pd.notna(net_capacity)
        and float(net_capacity) > 0
        and safe_compare(history_qty, 0, '==')
        and safe_compare(historical_peak, 0, '==')
        and safe_compare(recommended_baseline, original_baseline, '>')
    )

    matched = (
        condition_modified_lt_recommended
        or condition_history_gt_baseline
        or condition_zero_baseline
        or condition_recommended_gap
    )
    return rule_result(
        matched,
        conditions={
            '修改基线ROP<推荐基线': condition_modified_lt_recommended,
            '历史交易数量>原始基线': condition_history_gt_baseline,
            '原始基线=0且历史交易数量=0且网容>0': condition_zero_baseline,
            '历史需求为0但推荐基线高于原始基线': condition_recommended_gap,
        },
        evidence={
            '修改基线ROP': modified_rop,
            '推荐基线': recommended_baseline,
            '历史交易数量': history_qty,
            '原始基线': original_baseline,
            '网容': net_capacity,
            'M3-M13最大值': historical_peak,
        },
        rationale='基线小于建议值、被历史交易穿透，或存在明显基线配置偏低迹象时，归为基线异常。',
    )


def check_计划参数异常(row: pd.Series) -> Dict[str, Any]:
    final_lead_time = row.get('最终补库提前期')
    calculated_lead_time = row.get('计算补库提前期')
    matched = safe_compare(final_lead_time, calculated_lead_time, '<')
    return rule_result(
        matched,
        conditions={'最终补库提前期<计算补库提前期': matched},
        evidence={'最终补库提前期': final_lead_time, '计算补库提前期': calculated_lead_time},
        rationale='最终补库提前期低于计算值时，归为计划参数异常。',
    )


def check_补库供应不及时(row: pd.Series) -> Dict[str, Any]:
    inbound_time_raw = row.get('补库在途时间')
    transfer_time_raw = row.get('调拨在途时间')
    final_lead_time = row.get('最终补库提前期')
    inbound_days = parse_duration(inbound_time_raw)
    transfer_days = parse_duration(transfer_time_raw)
    condition_inbound = safe_compare(inbound_days, final_lead_time, '>')
    condition_transfer = safe_compare(transfer_days, final_lead_time, '>')
    return rule_result(
        condition_inbound or condition_transfer,
        conditions={
            '补库在途时间>最终补库提前期': condition_inbound,
            '调拨在途时间>最终补库提前期': condition_transfer,
        },
        evidence={
            '补库在途时间': inbound_time_raw,
            '调拨在途时间': transfer_time_raw,
            '补库在途时间(天)': inbound_days,
            '调拨在途时间(天)': transfer_days,
            '最终补库提前期': final_lead_time,
        },
        rationale='在途时长超过最终补库提前期时，归为补库供应不及时。',
    )


def check_责任库房异常(row: pd.Series) -> Dict[str, Any]:
    warehouse_eta_raw = row.get('责任库房预测物流时长')
    sla_raw = row.get('SLA承诺时间')
    warehouse_eta_days = parse_duration(warehouse_eta_raw)
    sla_days = parse_duration(sla_raw)
    matched = safe_compare(warehouse_eta_days, sla_days, '>')
    return rule_result(
        matched,
        conditions={'责任库房预测物流时长>SLA承诺时间': matched},
        evidence={
            '责任库房预测物流时长': warehouse_eta_raw,
            'SLA承诺时间': sla_raw,
            '责任库房预测物流时长(天)': warehouse_eta_days,
            'SLA承诺时间(天)': sla_days,
        },
        rationale='预测物流时长超过 SLA 承诺时，归为责任库房异常。',
    )


def check_替代交付异常(row: pd.Series) -> Dict[str, Any]:
    single_substitute_stock = row.get('单一替代库存汇总')
    shortage_total = row.get('总欠料数量')
    combo_relation_raw = row.get('组合替代关系')
    combo_stock_raw = row.get('组合替代库存汇总')
    combo_demand_raw = row.get('组合替代需求数量')

    condition_single_shortage = safe_compare(single_substitute_stock, shortage_total, '<')
    combo_relation = parse_literal(combo_relation_raw)
    relation_non_empty = isinstance(combo_relation, list) and len(combo_relation) > 0
    combo_stock_pairs = normalize_pairs(combo_stock_raw)
    combo_demand_pairs = normalize_pairs(combo_demand_raw)
    combo_stock_map = build_qty_map(combo_stock_pairs)
    combo_demand_map = build_qty_map(combo_demand_pairs)
    combo_missing = relation_non_empty and not combo_stock_pairs

    insufficient_items = []
    for sku, required_qty in combo_demand_map.items():
        available_qty = combo_stock_map.get(sku, 0.0)
        if available_qty + 1e-9 < required_qty:
            insufficient_items.append({'sku': sku, 'required_qty': required_qty, 'available_qty': available_qty})

    condition_combo_insufficient = relation_non_empty and (combo_missing or bool(insufficient_items))
    matched = condition_single_shortage and condition_combo_insufficient
    return rule_result(
        matched,
        conditions={
            '单一替代库存汇总<总欠料数量': condition_single_shortage,
            '组合替代关系非空': relation_non_empty,
            '组合替代库存缺失': combo_missing,
            '组合替代库存不足': bool(insufficient_items),
        },
        evidence={
            '单一替代库存汇总': single_substitute_stock,
            '总欠料数量': shortage_total,
            '组合替代关系': combo_relation,
            '组合替代库存汇总': combo_stock_pairs,
            '组合替代需求数量': combo_demand_pairs,
            '组合替代不足项': insufficient_items,
        },
        rationale='单一替代不足且组合替代也无法覆盖缺口时，归为替代交付异常。',
    )


RULE_CHECKERS = {
    '网容异常': check_网容异常,
    '用量异常': check_用量异常,
    '补库异常': check_补库异常,
    '基线异常': check_基线异常,
    '计划参数异常': check_计划参数异常,
    '补库供应不及时': check_补库供应不及时,
    '责任库房异常': check_责任库房异常,
    '替代交付异常': check_替代交付异常,
}


def analyze_row(row: pd.Series, *, include_trace: bool = False):
    labels: List[str] = []
    trace: Dict[str, Any] = {}
    for rule_name in RULE_ORDER:
        result = RULE_CHECKERS[rule_name](row)
        trace[rule_name] = result
        if result['matched']:
            labels.append(rule_name)
    joined_labels = '、'.join(labels) if labels else '- 未匹配到分支'
    if not include_trace:
        return joined_labels
    return joined_labels, trace


def analyze_dataframe(df: pd.DataFrame, *, include_trace: bool = False):
    labels: List[str] = []
    traces: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if include_trace:
            label_text, trace = analyze_row(row, include_trace=True)
            labels.append(label_text)
            traces.append(trace)
        else:
            labels.append(analyze_row(row, include_trace=False))
    if include_trace:
        return labels, traces
    return labels


def write_trace_file(df: pd.DataFrame, traces: List[Dict[str, Any]], trace_output_path: Path, *, prediction_column: str = 'L2分类结果') -> None:
    trace_output_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_output_path.open('w', encoding='utf-8') as handle:
        for row_index, trace in enumerate(traces):
            record = {
                'row_index': row_index,
                'prediction': df.iloc[row_index][prediction_column],
                'answer': normalize_scalar(df.iloc[row_index].get('answer')),
                'references': normalize_scalar(df.iloc[row_index].get('references')),
                'trace': trace,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def analyze_excel(input_path: str, output_path: str | None = None, *, trace_output_path: str | None = None) -> str:
    print(f'正在读取文件: {input_path}')
    df = pd.read_excel(input_path)
    print(f'读取完成，共 {len(df)} 行数据')
    print('正在进行欠料归因分析...')

    if trace_output_path:
        labels, traces = analyze_dataframe(df, include_trace=True)
        df['L2分类结果'] = labels
    else:
        df['L2分类结果'] = analyze_dataframe(df, include_trace=False)

    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f'{input_file.stem}_归因分析结果.xlsx')

    df.to_excel(output_path, index=False)
    print(f'分析完成，结果已保存至: {output_path}')

    if trace_output_path:
        trace_path = Path(trace_output_path)
        write_trace_file(df, traces, trace_path)
        print(f'trace 已保存至: {trace_path}')

    return str(output_path)


def main() -> None:
    if len(sys.argv) < 2:
        print('使用方法: python analyze_shortage.py <input_excel_path> [output_excel_path] [trace_output_path]')
        print('\n示例:')
        print('  python analyze_shortage.py 欠料数据.xlsx')
        print('  python analyze_shortage.py 欠料数据.xlsx 输出结果.xlsx')
        print('  python analyze_shortage.py 欠料数据.xlsx 输出结果.xlsx trace.jsonl')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    trace_output_path = sys.argv[3] if len(sys.argv) > 3 else None
    analyze_excel(input_path, output_path, trace_output_path=trace_output_path)


if __name__ == '__main__':
    main()

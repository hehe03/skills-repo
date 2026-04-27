#!/usr/bin/env python3
"""
欠料归因分析脚本

功能：
1. 读取欠料原始Excel数据
2. 根据业务规则对每一行数据进行L2分类
3. 输出带分类标签的新Excel文件

使用方法：
    python analyze_shortage.py <input_excel_path> [output_excel_path]
"""

import ast
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path


def parse_duration(value):
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip()
    match = re.match(r'^([\d.]+)\s*([dh])$', value_str, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        unit = match.group(2).lower()
        if unit == 'h':
            return num / 24
        return num

    try:
        return float(value_str)
    except ValueError:
        return np.nan


def safe_compare(val1, val2, operator='>'):
    if pd.isna(val1) or pd.isna(val2):
        return False

    try:
        val1 = float(val1)
        val2 = float(val2)
    except (ValueError, TypeError):
        return False

    if operator == '>':
        return val1 > val2
    elif operator == '<':
        return val1 < val2
    elif operator == '>=':
        return val1 >= val2
    elif operator == '<=':
        return val1 <= val2
    elif operator == '==':
        return val1 == val2

    return False


def parse_pairs(value):
    if pd.isna(value):
        return []
    try:
        obj = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(obj, list):
        return []

    pairs = []
    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), item[1]))
    return pairs


def parse_demand_amount(value):
    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value)
    if not value_str:
        return 0.0

    # 组合替代需求数量中存在 '1.01.0' 这类拼接格式，按出现次数计数。
    if '1.0' in value_str:
        return float(value_str.count('1.0'))

    try:
        return float(value_str)
    except ValueError:
        return 0.0


def combo_insufficient(stock_value, demand_value):
    stock_pairs = parse_pairs(stock_value)
    demand_pairs = parse_pairs(demand_value)
    if not demand_pairs:
        return False

    stock_map = {key: float(value) for key, value in stock_pairs}
    for key, demand_raw in demand_pairs:
        demand = parse_demand_amount(demand_raw)
        if stock_map.get(key, 0.0) < demand:
            return True
    return False


def check_网容异常(row):
    网容 = row.get('网容')
    if pd.isna(网容):
        return True
    return float(网容) <= 0


def check_用量异常(row):
    M2 = row.get('M2')
    M3_M13最大值 = row.get('M3-M13最大值')
    历史交易数量 = row.get('历史交易数量')
    原始基线 = row.get('原始基线')

    条件1 = safe_compare(M2, M3_M13最大值, '>')
    条件2 = safe_compare(历史交易数量, 原始基线, '>')

    return 条件1 or 条件2


def check_补库异常(row):
    历史交易数量 = row.get('历史交易数量')
    补库提前期补库和调拨汇总数量 = row.get('补库提前期补库和调拨汇总数量')
    return safe_compare(历史交易数量, 补库提前期补库和调拨汇总数量, '>')


def check_基线异常(row):
    修改基线ROP = row.get('修改基线ROP')
    推荐基线 = row.get('推荐基线')
    历史交易数量 = row.get('历史交易数量')
    原始基线 = row.get('原始基线')
    网容 = row.get('网容')

    条件1 = safe_compare(修改基线ROP, 推荐基线, '<')
    条件2 = safe_compare(历史交易数量, 原始基线, '>')

    条件3 = False
    if safe_compare(原始基线, 0, '==') and safe_compare(历史交易数量, 0, '=='):
        if pd.notna(网容) and float(网容) > 0:
            条件3 = True

    条件4 = False
    if pd.isna(修改基线ROP):
        if safe_compare(原始基线, 推荐基线, '<') and safe_compare(历史交易数量, 0, '=='):
            if pd.notna(网容) and float(网容) > 0:
                条件4 = True

    return 条件1 or 条件2 or 条件3 or 条件4


def check_计划参数异常(row):
    最终补库提前期 = row.get('最终补库提前期')
    计算补库提前期 = row.get('计算补库提前期')
    return safe_compare(最终补库提前期, 计算补库提前期, '<')


def check_补库供应不及时(row):
    补库在途时间 = parse_duration(row.get('补库在途时间'))
    调拨在途时间 = parse_duration(row.get('调拨在途时间'))
    最终补库提前期 = row.get('最终补库提前期')

    条件1 = safe_compare(补库在途时间, 最终补库提前期, '>')
    条件2 = safe_compare(调拨在途时间, 最终补库提前期, '>')
    return 条件1 or 条件2


def check_责任库房异常(row):
    责任库房预测物流时长 = parse_duration(row.get('责任库房预测物流时长'))
    SLA承诺时间 = parse_duration(row.get('SLA承诺时间'))
    return safe_compare(责任库房预测物流时长, SLA承诺时间, '>')


def check_替代交付异常(row, current_labels=None):
    单一替代库存汇总 = row.get('单一替代库存汇总')
    总欠料数量 = row.get('总欠料数量')
    组合替代关系 = row.get('组合替代关系')
    组合替代库存汇总 = row.get('组合替代库存汇总')
    组合替代需求数量 = row.get('组合替代需求数量')

    条件1 = safe_compare(单一替代库存汇总, 总欠料数量, '<')

    if pd.isna(组合替代关系) or str(组合替代关系).strip() == '[]':
        labels = current_labels or []
        条件2 = (
            条件1
            and safe_compare(row.get('M2'), 0, '==')
            and safe_compare(总欠料数量, 1, '==')
            and '补库供应不及时' not in labels
            and safe_compare(row.get('历史交易数量'), 0, '==')
            and pd.notna(row.get('最终补库提前期'))
            and float(row.get('最终补库提前期')) >= 60
        )
    else:
        if pd.isna(组合替代库存汇总) or str(组合替代库存汇总).strip() == '[]':
            条件2 = True
        else:
            条件2 = combo_insufficient(组合替代库存汇总, 组合替代需求数量)

    return 条件1 and 条件2


def analyze_row(row):
    labels = []

    if check_网容异常(row):
        labels.append('网容异常')
    if check_用量异常(row):
        labels.append('用量异常')
    if check_补库异常(row):
        labels.append('补库异常')
    if check_基线异常(row):
        labels.append('基线异常')
    if check_计划参数异常(row):
        labels.append('计划参数异常')
    if check_补库供应不及时(row):
        labels.append('补库供应不及时')
    if check_责任库房异常(row):
        labels.append('责任库房异常')
    if check_替代交付异常(row, labels):
        labels.append('替代交付异常')

    if not labels:
        return '- 未匹配到分支'

    return '、'.join(labels)


def analyze_excel(input_path, output_path=None):
    print(f"正在读取文件: {input_path}")
    df = pd.read_excel(input_path)
    print(f"读取完成，共 {len(df)} 行数据")

    print("正在进行欠料归因分析...")
    df['L2分类结果'] = df.apply(analyze_row, axis=1)

    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_归因分析结果.xlsx"

    df.to_excel(output_path, index=False)
    print(f"分析完成，结果已保存至: {output_path}")
    return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_shortage.py <input_excel_path> [output_excel_path]")
        print("\n示例:")
        print("  python analyze_shortage.py 欠料数据.xlsx")
        print("  python analyze_shortage.py 欠料数据.xlsx 输出结果.xlsx")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_excel(input_path, output_path)


if __name__ == '__main__':
    main()

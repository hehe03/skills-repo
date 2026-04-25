#!/usr/bin/env python3
"""
欠料归因分析脚本

功能：
1. 读取欠料原始Excel数据
2. 根据业务规则对每一行数据进行L2分类
3. 输出带分类标签的新Excel文件
4. 可选：输出JSONL格式的trace文件用于审计

使用方法：
    python analyze_shortage.py <input_excel_path> [output_excel_path] [--trace]

注意：
- 本脚本严格按照AI逻辑规则实现
- 如果验证数据发现准确率问题，请检查示例数据是否存在不一致
- 已知问题：部分规则与示例数据的answer列存在差异，可能是数据标注问题
"""

import sys
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def parse_duration(value):
    """
    解析时间字符串为数值（统一转换为天数）

    支持格式：
    - "62.87d" -> 62.87 (天)
    - "8.94h" -> 0.3725 (天，小时转天)
    - "0d" -> 0
    - NaN -> NaN
    """
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip()

    # 匹配数字+单位格式
    match = re.match(r'^([\d.]+)\s*([dh])$', value_str, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        unit = match.group(2).lower()
        if unit == 'h':
            return num / 24  # 小时转天
        return num  # 天

    # 尝试直接转换为数值
    try:
        return float(value_str)
    except ValueError:
        return np.nan


def safe_compare(val1, val2, operator='>'):
    """
    安全比较两个数值，处理NaN情况

    Args:
        val1: 第一个值
        val2: 第二个值
        operator: 比较操作符 ('>', '<', '>=', '<=', '==')

    Returns:
        bool: 比较结果，如果任一值为NaN则返回False
    """
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


def check_网容异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    网容异常判断
    条件1: 网容大于0
    匹配度: 若条件1不满足，匹配度为1；否则为0
    即: 网容 <= 0 或为空 -> 匹配
    """
    网容 = row.get('网容')
    hit = False
    conditions = {}
    evidence = {'网容': 网容 if not pd.isna(网容) else None}
    
    if pd.isna(网容):
        hit = True
        conditions['网容为空'] = True
    else:
        conditions['网容 <= 0'] = float(网容) <= 0
        hit = conditions['网容 <= 0']
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_用量异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    用量异常判断
    条件1: M2超过M3-M13最大值
    条件2: 历史交易数量超过原始基线
    匹配度: 若条件1或条件2完全满足，匹配度为1；否则为0
    """
    M2 = row.get('M2')
    M3_M13最大值 = row.get('M3-M13最大值')
    历史交易数量 = row.get('历史交易数量')
    原始基线 = row.get('原始基线')

    条件1 = safe_compare(M2, M3_M13最大值, '>')
    条件2 = safe_compare(历史交易数量, 原始基线, '>')
    hit = 条件1 or 条件2
    
    evidence = {
        'M2': M2 if not pd.isna(M2) else None,
        'M3-M13最大值': M3_M13最大值 if not pd.isna(M3_M13最大值) else None,
        '历史交易数量': 历史交易数量 if not pd.isna(历史交易数量) else None,
        '原始基线': 原始基线 if not pd.isna(原始基线) else None
    }
    
    conditions = {
        'M2 > M3-M13最大值': 条件1,
        '历史交易数量 > 原始基线': 条件2
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_补库异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    补库异常判断
    条件1: 历史交易数量超过补库提前期补库和调拨汇总数量
    匹配度: 若条件1完全满足，匹配度为1；否则为0
    """
    历史交易数量 = row.get('历史交易数量')
    补库提前期补库和调拨汇总数量 = row.get('补库提前期补库和调拨汇总数量')

    hit = safe_compare(历史交易数量, 补库提前期补库和调拨汇总数量, '>')
    
    evidence = {
        '历史交易数量': 历史交易数量 if not pd.isna(历史交易数量) else None,
        '补库提前期补库和调拨汇总数量': 补库提前期补库和调拨汇总数量 if not pd.isna(补库提前期补库和调拨汇总数量) else None
    }
    
    conditions = {
        '历史交易数量 > 补库提前期补库和调拨汇总数量': hit
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_基线异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    基线异常判断
    条件1: 修改基线ROP小于推荐基线
    条件2: 历史交易数量超过原始基线
    条件3: 原始基线=0 且 历史交易数量=0 且 网容>0（基线设置问题）
    匹配度: 若条件1或条件2或条件3完全满足，匹配度为1；否则为0
    """
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

    hit = 条件1 or 条件2 or 条件3
    
    evidence = {
        '修改基线ROP': 修改基线ROP if not pd.isna(修改基线ROP) else None,
        '推荐基线': 推荐基线 if not pd.isna(推荐基线) else None,
        '历史交易数量': 历史交易数量 if not pd.isna(历史交易数量) else None,
        '原始基线': 原始基线 if not pd.isna(原始基线) else None,
        '网容': 网容 if not pd.isna(网容) else None
    }
    
    conditions = {
        '修改基线ROP < 推荐基线': 条件1,
        '历史交易数量 > 原始基线': 条件2,
        '原始基线=0 且 历史交易数量=0 且 网容>0': 条件3
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_计划参数异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    计划参数异常判断
    条件1: 最终补库提前期小于计算补库提前期
    匹配度: 若条件1完全满足，匹配度为1；否则为0
    """
    最终补库提前期 = row.get('最终补库提前期')
    计算补库提前期 = row.get('计算补库提前期')

    hit = safe_compare(最终补库提前期, 计算补库提前期, '<')
    
    evidence = {
        '最终补库提前期': 最终补库提前期 if not pd.isna(最终补库提前期) else None,
        '计算补库提前期': 计算补库提前期 if not pd.isna(计算补库提前期) else None
    }
    
    conditions = {
        '最终补库提前期 < 计算补库提前期': hit
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_补库供应不及时(row) -> Tuple[bool, Dict[str, Any]]:
    """
    补库供应不及时判断
    条件1: 补库在途时间超过最终补库提前期
    条件2: 调拨在途时间超过最终补库提前期
    匹配度: 若条件1或条件2完全满足，匹配度为1；否则为0
    """
    补库在途时间_raw = row.get('补库在途时间')
    调拨在途时间_raw = row.get('调拨在途时间')
    最终补库提前期 = row.get('最终补库提前期')

    补库在途时间 = parse_duration(补库在途时间_raw)
    调拨在途时间 = parse_duration(调拨在途时间_raw)

    条件1 = safe_compare(补库在途时间, 最终补库提前期, '>')
    条件2 = safe_compare(调拨在途时间, 最终补库提前期, '>')
    
    hit = 条件1 or 条件2
    
    evidence = {
        '补库在途时间': 补库在途时间_raw if not pd.isna(补库在途时间_raw) else None,
        '调拨在途时间': 调拨在途时间_raw if not pd.isna(调拨在途时间_raw) else None,
        '最终补库提前期': 最终补库提前期 if not pd.isna(最终补库提前期) else None,
        '补库在途时间(天)': 补库在途时间 if not np.isnan(补库在途时间) else None,
        '调拨在途时间(天)': 调拨在途时间 if not np.isnan(调拨在途时间) else None
    }
    
    conditions = {
        '补库在途时间 > 最终补库提前期': 条件1,
        '调拨在途时间 > 最终补库提前期': 条件2
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_责任库房异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    责任库房异常判断
    条件1: 责任库房预测物流时长超过SLA承诺时间
    匹配度: 若条件1完全满足，匹配度为1；否则为0
    """
    责任库房预测物流时长_raw = row.get('责任库房预测物流时长')
    SLA承诺时间_raw = row.get('SLA承诺时间')

    责任库房预测物流时长 = parse_duration(责任库房预测物流时长_raw)
    SLA承诺时间 = parse_duration(SLA承诺时间_raw)

    hit = safe_compare(责任库房预测物流时长, SLA承诺时间, '>')
    
    evidence = {
        '责任库房预测物流时长': 责任库房预测物流时长_raw if not pd.isna(责任库房预测物流时长_raw) else None,
        'SLA承诺时间': SLA承诺时间_raw if not pd.isna(SLA承诺时间_raw) else None,
        '责任库房预测物流时长(天)': 责任库房预测物流时长 if not np.isnan(责任库房预测物流时长) else None,
        'SLA承诺时间(天)': SLA承诺时间 if not np.isnan(SLA承诺时间) else None
    }
    
    conditions = {
        '责任库房预测物流时长 > SLA承诺时间': hit
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence}


def check_替代交付异常(row) -> Tuple[bool, Dict[str, Any]]:
    """
    替代交付异常判断（低置信规则）
    条件1: 单一替代库存汇总小于总欠料数量
    条件2: 组合替代关系非空 且 (组合替代库存汇总为空/nan 或 无法满足需求)
    匹配度: 若条件1且条件2完全满足，匹配度为1；否则为0
    
    注意：
    - 当组合替代关系为空[]时，表示没有替代方案，不标记为替代交付异常
    - 此规则为低置信规则，判断逻辑复杂，建议作为候选标签输出
    """
    单一替代库存汇总 = row.get('单一替代库存汇总')
    总欠料数量 = row.get('总欠料数量')
    组合替代关系 = row.get('组合替代关系')
    组合替代库存汇总 = row.get('组合替代库存汇总')

    条件1 = safe_compare(单一替代库存汇总, 总欠料数量, '<')

    条件2 = False
    组合替代详情 = ''
    
    if pd.isna(组合替代关系) or str(组合替代关系).strip() == '[]':
        条件2 = False
        组合替代详情 = '无替代方案'
    else:
        if pd.isna(组合替代库存汇总) or str(组合替代库存汇总).strip() == '[]':
            条件2 = True
            组合替代详情 = '有替代方案但无库存数据'
        else:
            条件2 = False
            组合替代详情 = '有替代方案且有库存数据'

    hit = 条件1 and 条件2
    
    evidence = {
        '单一替代库存汇总': 单一替代库存汇总 if not pd.isna(单一替代库存汇总) else None,
        '总欠料数量': 总欠料数量 if not pd.isna(总欠料数量) else None,
        '组合替代关系': str(组合替代关系) if not pd.isna(组合替代关系) else None,
        '组合替代库存汇总': 组合替代库存汇总 if not pd.isna(组合替代库存汇总) else None
    }
    
    conditions = {
        '单一替代库存汇总 < 总欠料数量': 条件1,
        '组合替代关系非空且替代库存不足': 条件2
    }
    
    details = {
        '组合替代详情': 组合替代详情,
        'confidence': 'low'
    }
    
    return hit, {'conditions': conditions, 'evidence': evidence, 'details': details}


def analyze_row(row, return_trace=False) -> Any:
    """
    分析单行数据，返回L2标签列表和可选的trace数据
    
    Args:
        row: 数据行
        return_trace: 是否返回trace数据
        
    Returns:
        如果 return_trace=False: 返回标签字符串
        如果 return_trace=True: 返回 Tuple[标签字符串, trace字典]
    """
    labels = []
    candidate_labels = []
    rule_hits = []
    
    checks = [
        ('网容异常', check_网容异常, 'high'),
        ('用量异常', check_用量异常, 'high'),
        ('补库异常', check_补库异常, 'high'),
        ('基线异常', check_基线异常, 'high'),
        ('计划参数异常', check_计划参数异常, 'high'),
        ('补库供应不及时', check_补库供应不及时, 'high'),
        ('责任库房异常', check_责任库房异常, 'high'),
        ('替代交付异常', check_替代交付异常, 'low')
    ]
    
    for label_name, check_func, confidence in checks:
        hit, trace_info = check_func(row)
        rule_hit: Dict[str, Any] = {
            'label': label_name,
            'hit': hit,
            'conditions': trace_info['conditions'],
            'evidence': trace_info['evidence'],
            'confidence': confidence
        }
        if 'details' in trace_info:
            rule_hit['details'] = trace_info['details']
        
        rule_hits.append(rule_hit)
        
        if hit:
            if confidence == 'high':
                labels.append(label_name)
            else:
                candidate_labels.append(label_name)
    
    if not labels:
        label_str = '- 未匹配到分支'
    else:
        label_str = '、'.join(labels)
    
    if return_trace:
        trace: Dict[str, Any] = {
            'confirmed_labels': labels,
            'candidate_labels': candidate_labels,
            'rule_hits': rule_hits,
            'trace_digest': label_str
        }
        return label_str, trace
    
    return label_str


def analyze_excel(input_path, output_path=None, enable_trace=False):
    """
    分析Excel文件
    
    Args:
        input_path: 输入Excel文件路径
        output_path: 输出Excel文件路径，如不指定则自动生成
        enable_trace: 是否生成JSONL格式的trace文件
        
    Returns:
        输出文件路径
    """
    print(f"正在读取文件: {input_path}")

    df = pd.read_excel(input_path)
    print(f"读取完成，共 {len(df)} 行数据")

    print("正在进行欠料归因分析...")
    
    traces = []
    l2_results = []
    candidate_results = []
    
    for idx, row in df.iterrows():
        label_str, trace = analyze_row(row, return_trace=True)
        l2_results.append(label_str)
        
        candidate_str = '、'.join(trace['candidate_labels']) if trace['candidate_labels'] else ''
        candidate_results.append(candidate_str)
        
        if enable_trace:
            trace['sample_id'] = f'row-{idx}'
            trace['stage'] = 'rule-evaluation'
            traces.append(trace)
    
    df['L2分类结果'] = l2_results
    df['候选标签'] = candidate_results

    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_归因分析结果.xlsx"

    df.to_excel(output_path, index=False)
    print(f"分析完成，结果已保存至: {output_path}")

    if enable_trace:
        trace_path = Path(output_path).with_suffix('.jsonl')
        with trace_path.open('w', encoding='utf-8') as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        print(f"Trace文件已保存至: {trace_path}")

    return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_shortage.py <input_excel_path> [output_excel_path] [--trace]")
        print("\n示例:")
        print("  python analyze_shortage.py 欠料数据.xlsx")
        print("  python analyze_shortage.py 欠料数据.xlsx 输出结果.xlsx")
        print("  python analyze_shortage.py 欠料数据.xlsx 输出结果.xlsx --trace")
        print("\n选项:")
        print("  --trace    生成JSONL格式的trace文件用于审计分析")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    enable_trace = '--trace' in sys.argv

    analyze_excel(input_path, output_path, enable_trace)


if __name__ == "__main__":
    main()
---
name: 欠料归因分析
description: 基于欠料分析数据和规则，归因欠料原因。当用户需要分析欠料Excel数据，得到欠料归因结果时触发。接收用户指定的原始Excel表格，执行分析后生成带L2分类标签的新表格
---

# 欠料归因分析

对供应链欠料数据进行归因分析，识别异常类型。

## 快速使用

用户指定原始数据Excel后，运行分析脚本：

```
python scripts/analyze_shortage.py <input_excel_path> <output_excel_path>
```

分析完成后，输出带L2分类结果的新Excel。

## Trace 输出（用于审计）

使用 `--trace` 参数生成 JSONL 格式的追踪文件：

```
python scripts/analyze_shortage.py <input_excel_path> <output_excel_path> --trace
```

Trace 文件包含：
- sample_id：样本标识
- stage：规则评估阶段
- confirmed_labels：高置信标签
- candidate_labels：低置信候选标签
- rule_hits：每条规则的触发记录（包含条件和证据）

## L2分类标签

详细说明见 references/rules.md

若无匹配类别，则划分到 - 未匹配到分支

## 置信度分离

| 标签类型 | 置信度 | 说明 |
|---------|-------|------|
| 网容异常 | high | 直接判断，证据清晰 |
| 用量异常 | high | 数值比较，可审计 |
| 补库异常 | high | 数值比较，可审计 |
| 基线异常 | high | 数值比较，可审计 |
| 计划参数异常 | high | 数值比较，可审计 |
| 补库供应不及时 | high | 数值比较，可审计 |
| 责任库房异常 | high | 数值比较，可审计 |
| 替代交付异常 | low | 逻辑复杂，数据不完整，作为候选标签 |

替代交付异常规则因逻辑复杂且依赖多个不完整字段，被标记为低置信规则。分析时输出到"候选标签"列，供人工审核确认。

## 审计工作流

使用 skill-ops 技能进行审计：

```
python ../skill-ops/scripts/label_audit.py <output_excel_path> \
    --human-col "人工标注_L2分类结果" \
    --pred-col "L2分类结果" \
    --candidate-col "候选标签" \
    --low-confidence-labels "替代交付异常"
```

审计报告将输出：
- 主口径一致率
- 疑似人工漏标/误标数量
- 低置信规则差异数量
- 具体问题样本列表

## 当前审计结果

基于最新审计（2026-04-25）：
- 主口径一致率：84%
- 疑似人工漏标：11 条
- 疑似人工误标：5 条
- 低置信规则差异：7 条

详见 `insight_report.md` 和 `audit_report.md`

## 输出字段

| 字段名 | 说明 |
|-------|------|
| L2分类结果 | 高置信标签，用"、"分隔 |
| 候选标签 | 低置信标签，供人工审核 |

#

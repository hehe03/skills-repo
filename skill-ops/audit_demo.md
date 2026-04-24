# Audit Report: data_归因分析结果_审计结果.xlsx

- total_rows: 100
- primary_matches: 85
- primary_match_rate: 85.00%
- low_confidence_labels: 替代交付异常

## Discrepancy Types

- 一致: 77
- 疑似人工漏标: 10
- 低置信规则差异: 8
- 疑似人工误标: 5

## Label Counts

- 基线异常: human=67 pred=74
- 用量异常: human=58 pred=64
- 网容异常: human=20 pred=19
- 补库供应不及时: human=12 pred=13
- 补库异常: human=24 pred=26
- 计划参数异常: human=26 pred=28
- 责任库房异常: human=22 pred=23

## Suspected Label-Issue Rows

- sample_id=0 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=基线异常、用量异常、网容异常、计划参数异常 | note=预测新增命中: 基线异常、用量异常、网容异常、计划参数异常
- sample_id=9 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=网容异常 | note=预测新增命中: 网容异常
- sample_id=17 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=基线异常、用量异常 | note=预测新增命中: 基线异常、用量异常
- sample_id=18 | type=疑似人工误标 | human=基线异常、用量异常、责任库房异常 | pred=基线异常、用量异常 | note=人工多标但缺少支持证据: 责任库房异常
- sample_id=23 | type=疑似人工漏标 | human=基线异常 | pred=基线异常、用量异常 | note=预测新增命中: 用量异常
- sample_id=28 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=基线异常、用量异常、计划参数异常 | note=预测新增命中: 基线异常、用量异常、计划参数异常
- sample_id=29 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=基线异常、用量异常、责任库房异常 | note=预测新增命中: 基线异常、用量异常、责任库房异常
- sample_id=30 | type=疑似人工漏标 | human=基线异常 | pred=基线异常、补库异常 | note=预测新增命中: 补库异常
- sample_id=40 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=基线异常、用量异常、补库异常、责任库房异常 | note=预测新增命中: 基线异常、用量异常、补库异常、责任库房异常
- sample_id=41 | type=疑似人工漏标 | human=未匹配到异常分支 | pred=基线异常 | note=预测新增命中: 基线异常
- sample_id=57 | type=疑似人工误标 | human=网容异常 | pred=基线异常 | note=人工多标但缺少支持证据: 网容异常；预测新增命中: 基线异常
- sample_id=58 | type=疑似人工误标 | human=网容异常 | pred=基线异常 | note=人工多标但缺少支持证据: 网容异常；预测新增命中: 基线异常
- sample_id=71 | type=疑似人工漏标 | human=用量异常、计划参数异常 | pred=用量异常、补库供应不及时、计划参数异常 | note=预测新增命中: 补库供应不及时
- sample_id=78 | type=疑似人工误标 | human=基线异常、计划参数异常、责任库房异常 | pred=计划参数异常、责任库房异常 | note=人工多标但缺少支持证据: 基线异常
- sample_id=81 | type=疑似人工误标 | human=网容异常、补库异常、责任库房异常 | pred=补库异常、责任库房异常 | note=人工多标但缺少支持证据: 网容异常
- sample_id=85 | type=低置信规则差异 | human=用量异常 | pred=用量异常 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=86 | type=低置信规则差异 | human=责任库房异常 | pred=责任库房异常 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=87 | type=低置信规则差异 | human=未匹配到异常分支 | pred=未匹配到异常分支 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=88 | type=低置信规则差异 | human=计划参数异常 | pred=计划参数异常 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=89 | type=低置信规则差异 | human=基线异常 | pred=基线异常 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=90 | type=低置信规则差异 | human=基线异常、用量异常 | pred=基线异常、用量异常 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=91 | type=低置信规则差异 | human=基线异常、用量异常、网容异常、计划参数异常 | pred=基线异常、用量异常、网容异常、计划参数异常 | note=人工仅多出低置信规则: 替代交付异常
- sample_id=92 | type=低置信规则差异 | human=用量异常、补库异常 | pred=用量异常、补库异常 | note=人工仅多出低置信规则: 替代交付异常
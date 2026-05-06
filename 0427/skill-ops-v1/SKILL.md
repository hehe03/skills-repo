---
name: skill-ops-v1
description: 规则型 Agent skill 优化增强版。除 trace、评估、反事实分析外，还支持标注疑点排序和高频冲突模式挖掘。
---

# skill-ops-v1

`skill-ops-v1` 是 `skill-ops` 的增强版，目标是在原有能力基础上增加两类算法：

1. 标注疑点排序；
2. 高频冲突模式挖掘。

## 相比 skill-ops 的新增能力

### 1. 标注疑点排序

通过 `rank_label_issues.py` 对每个错例进行启发式打分，优先找出：

- 相同输入不同标注；
- 标注包含目标标签，但 trace 明确不支持；
- 预测与标注在目标标签上长期冲突；
- 满足高风险冲突模式的样本。

### 2. 高频冲突模式挖掘

通过 `mine_conflict_patterns.py`，对人工指定的派生特征做分组，输出：

- 模式支持度；
- 标注阳性率；
- 预测阳性率；
- 冲突缺口。

这样可以更快识别类似“组合替代关系为空 + 单一替代不足”这样的高频冲突簇。

## 标准流程

1. 对工作 skill 补 trace；
2. 重跑预测结果；
3. 运行 `evaluate_case.py` 建立基线；
4. 运行 `rank_label_issues.py` 找高置信度标注疑点；
5. 运行 `mine_conflict_patterns.py` 挖掘高频冲突模式；
6. 运行 `analyze_counterfactuals.py` 比较改规则和改标注的后果；
7. 基于上述证据决定是否修改规则，并将优化后案例输出到 `./opt/<case>-v1`。

## 推荐工具

- `scripts/evaluate_case.py`
- `scripts/analyze_counterfactuals.py`
- `scripts/rank_label_issues.py`
- `scripts/mine_conflict_patterns.py`

## 注意事项

1. `skill-ops-v1` 仍然优先追求泛化能力，而不是训练集刷分；
2. 若某类模式在“改标注”方案下收益显著高于“改规则”，应优先将其标为标注疑点；
3. 模式挖掘用于发现候选问题，不应直接替代业务规则确认。

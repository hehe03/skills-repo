---
name: trace-sorter
description: >-
  对 Agent trace 文件进行 goodcase / badcase 分类，并给出可解释的规则命中原因。适用于检查单个或批量
  Agent trace、运行批量实验、比较三层规则，或从无标注/有标注 trace 样例中萃取规则：（1）无先验的通用规则，
  （2）从无标注 trace 中提取的动态规则，（3）从有标注 train trace 中提取的动态规则。
---

# Trace Sorter

使用本 skill 将 Agent trace 分类为 `goodcase` 或 `badcase`，并输出命中的规则、分数和判定原因。

## 输入

输入可以是单个 `.json` trace 文件，也可以是包含多个 `.json` trace 的目录。

如果有 metadata，请使用 CSV，推荐列如下：

```text
name,label,source,split
```

- `name`：trace 文件名，例如 `case_001.json`。
- `label`：可选真实标签，取值为 `goodcase` 或 `badcase`。
- `source`：可选数据来源。
- `split`：可选数据划分，通常为 `train` 或 `test`。

## 当前方法如何从 Trace 中萃取规则

方法分为“特征抽取 -> 规则生成 -> 规则匹配 -> 分数合并”四步。

1. **抽取通用特征**：`scripts/features.py` 会遍历 trace JSON，识别 `plan_list`、`steps`、`events`、`messages`、`actions` 等常见结构，并抽取可解释特征，例如 `step_count`、`error_count`、`empty_result_ratio`、`max_consecutive_same_action`、`has_final_answer`、`nonempty_result_ratio`。
2. **使用静态通用规则**：`scripts/rules/static/general_rules.json` 写入不依赖任何数据集先验的规则，例如 JSON 解析失败、空 trace、无可观察步骤、错误文本、缺少最终回答、重复动作疑似循环。
3. **从无标注样例萃取规则**：`scripts/rule_generation.py` 统计一批 trace 的特征分布，对步数、空结果比例、重复动作深度、错误次数等风险特征取高分位阈值，生成“相对当前样本群异常”的 badcase 规则，写入 `scripts/rules/dynamic/unlabeled_rules.json`。
4. **从有标注样例萃取规则**：当 metadata 中存在 `split=train` 且同时包含 `goodcase` 和 `badcase` 时，脚本比较两类样本的特征均值。如果某个风险特征在 badcase 中显著更高，就生成区分阈值规则；如果最终回答等正向信号在 goodcase 中明显更常见，就生成 goodcase 支持规则。结果写入 `scripts/rules/dynamic/labeled_rules.json`。

默认不调用 LLM 生成规则。若需要引入 LLM，可让 LLM 读样例后产出同一 JSON schema 的候选规则，再人工审查后写入动态规则文件。

## 规则层级

按可用信息选择最强层级：

1. **通用规则（`general`）**：没有任何先验知识，只加载 `scripts/rules/static/general_rules.json`。
2. **无标注动态规则（`unlabeled`）**：从无标注 trace 样例中萃取群体异常阈值规则。
3. **有标注动态规则（`labeled`）**：从有标注 train 样例中萃取区分 goodcase / badcase 的规则。

## 常用命令

只用通用规则分类单个文件或目录：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir>
```

运行批量实验，并生成以“实验方法+时间”命名的 Markdown 报告：

```powershell
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer general
```

实验前生成无标注或有标注动态规则：

```powershell
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer unlabeled --generate-dynamic-rules unlabeled
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer labeled --generate-dynamic-rules labeled
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer all --generate-dynamic-rules auto
```

常用参数：

- `--bad-threshold`：判为 badcase 的最低风险分，默认 `0.60`。
- `--good-threshold`：判为 goodcase 的最低支持分，默认 `0.50`。
- `--output-dir`：实验报告输出目录，默认当前目录。
- `--max-rows`：Markdown 报告中最多展示的样本行数。

## 输出

每条预测包含：

- `name`
- `predicted_label`
- `bad_score`
- `good_score`
- `matched_rules`
- `reason`

当 metadata 中包含真实标签时，实验报告还会输出混淆矩阵，以及以 `badcase` 为正类的 precision、recall 和 F1。

## 维护约定

- 静态通用规则只维护在 `scripts/rules/static/general_rules.json`。
- 无标注样例生成的规则保存在 `scripts/rules/dynamic/unlabeled_rules.json`。
- 有标注样例生成的规则保存在 `scripts/rules/dynamic/labeled_rules.json`。
- 新增特征时先修改 `scripts/features.py`，再在规则 JSON 中引用该特征。
- 规则应保持保守和可解释：badcase 规则要对应具体失败风险，goodcase 规则要对应明确完成证据。

---
name: trace-boundary-analysis
description: 用于分析 Agent trace 并判断 goodcase / badcase 的自包含技能。适用于需要检查一个或多个 trace JSON、自动选择规则法/无监督特征法/有监督特征法、识别重复调用、循环执行、结果为空、缺少大纲生成等 badcase 信号，并输出可解释判定理由的场景。
---

# Trace 定界分析

这个 skill 用于分析 Agent trace，并将样本划分为 `goodcase` 或 `badcase`。skill 文件夹是自包含的：整体保留本目录，其他 Agent 就可以直接加载并使用。

## 文件结构

```text
trace-boundary-analysis/
├── SKILL.md
├── rules.md
├── 方法及优化.md
└── scripts/
    ├── run_trace_analysis.py
    ├── common.py
    ├── features.py
    ├── classify_rule.py
    ├── classify_unsupervised.py
    ├── classify_unsupervised_hybrid.py
    └── classify_supervised.py
```

- `run_trace_analysis.py`：统一入口，只负责参数解析、读取输入、自动选路和输出。
- `common.py`：公共数据结构、trace/metadata 读取、指标统计和结果序列化。
- `features.py`：公共特征抽取逻辑。
- `classify_rule.py`：规则法，从 `rules.md` 读取规则配置。
- `classify_unsupervised.py`：无监督特征评分法。
- `classify_unsupervised_hybrid.py`：混合无监督特征法，输出结构、重复、结果、流程、语义子风险；“生成大纲”是较大权重特征，但不是 100% 判定证据。
- `classify_supervised.py`：有监督特征法。

维护规则法时，先更新 `rules.md`；只有当现有配置表达不了新规则时，再修改 `scripts/classify_rule.py`。

## 适用场景

- 用户提供一个 trace JSON 文件，需要快速判断是 `goodcase` 还是 `badcase`。
- 用户提供一批 trace JSON，需要批量定界并输出原因。
- 用户提供 metadata CSV，希望利用已有 `train/test` 标注自动选择监督或无监督方法。
- 用户希望定位 badcase 风险信号，例如重复调用工具、循环执行、任务结果为空、没有生成交流大纲等。

## Trace 输入格式

每个 trace 是一个 JSON 对象，推荐结构如下：

```json
{
  "query": "用户输入",
  "plan_list": [
    {
      "task_name": "任务名",
      "command": {
        "name": "工具名",
        "args": {
          "arg1": "value"
        }
      },
      "task_id": "step-1",
      "result": "执行结果"
    }
  ]
}
```

最低可用字段是 `query` 和 `plan_list`。如果需要更稳定地识别重复工具调用和循环模式，尽量保留 `command.name` 与 `command.args`。

## Metadata 输入格式

metadata CSV 是可选的。推荐列如下：

```text
name,label,source,split
```

- `name`：trace JSON 文件名，例如 `case_001.json`
- `label`：可选，`goodcase` 或 `badcase`
- `source`：可选，样本来源
- `split`：可选，`train` 或 `test`

## 默认工作流

1. 先确认输入是单个 JSON 文件还是 JSON 文件夹。
2. 如果有 metadata，读取标签、来源与 `train/test` 划分。
3. 默认使用 `auto` 策略自动选路。
4. `run_trace_analysis.py` 根据选路结果调用对应的单独方法脚本。
5. 输出每个样本的预测标签、使用的方法和判定原因。
6. 如果样本带有真实标签，额外输出 badcase precision / recall / F1。

## 自动选路策略

- 若 metadata 中存在可用 `train` 样本，并且 `train` 同时包含 `goodcase` 与 `badcase`，使用 `supervised`。
- 否则，如果待分析样本数量不少于 3，使用 `unsupervised`。
- 否则，使用 `rule`。
- 不默认使用 LLM 方法；当前项目实验里 LLM baseline 容易把样本全部判成 `goodcase`。

## 方法说明

### rule

规则法适合单条或少量 trace 快速筛查。它主要检查：

- `plan_list` 是否存在且为 list
- 是否出现重复或循环任务
- 是否出现“生成大纲”相关任务且结果有效

### unsupervised

无监督特征法适合没有训练集但有一批 trace 的场景。它会抽取任务数量、任务多样性、工具多样性、结果非空比例、连续重复、循环模式、结果多样性、是否生成大纲等特征，然后组合 `behavior_prior` 与 `bad_risk` 做判断。

### unsupervised_hybrid

混合无监督特征法适合需要更强解释性的无监督实验。它将风险拆成 `structure_risk`、`repeat_risk`、`result_risk`、`flow_risk`、`semantic_risk`，并计算 `good_score`。其中 `生成大纲` 是 `flow_risk` 和 `good_score` 的高权重信号，但不会单独决定最终标签。

### supervised

有监督特征法适合已有 `train/test` 标注的场景。它会基于 `train` 样本拟合 goodcase / badcase 特征质心，再根据待测样本到两个质心的距离生成预测，并给出最近的正负样本作为解释。

## 使用命令

自动选路：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir>
```

带 metadata 自动选路：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --metadata <metadata.csv>
```

强制使用某种方法：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy rule
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy unsupervised
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy unsupervised_hybrid
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy supervised --metadata <metadata.csv>
```

输出 JSON 文件：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --metadata <metadata.csv> --output result.json
```

扫描 hybrid 阈值：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --metadata <metadata.csv> --sweep-hybrid
```

## 输出解释

终端默认输出：

```text
sample    predicted_label    strategy    reason
```

字段含义：

- `sample`：样本文件名
- `predicted_label`：预测标签
- `strategy`：实际使用的方法
- `reason`：判定依据或风险信号摘要

当 metadata 中包含真实标签时，还会输出混淆矩阵和 badcase 指标。这里把 `badcase` 作为正类。

## 使用建议

- 只有单条 trace：优先使用默认 `auto`，通常会落到 `rule`。
- 有一批无标注 trace：优先使用默认 `auto`，通常会落到 `unsupervised`。
- 有稳定标注数据：提供 metadata，让默认 `auto` 使用 `supervised`。
- 若用户更重视 badcase precision，可以提高 `--bad-risk-threshold` 或使用 `supervised` 并提高 `--supervised-threshold`。
- 若用户更重视 badcase recall，可以降低 `--bad-risk-threshold` 或降低分类阈值。

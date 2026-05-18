---
name: trace-sorter
description: >-
  对 Agent trace 文件进行 goodcase / badcase 分类，并输出可解释的规则命中原因。
  适用于检查单个或批量 trace、运行训练/测试分离的分类实验、比较非 LLM 与 LLM 方法、
  生成动态规则，以及执行 precision-first 的组件消融研究。
---

# Trace Sorter

使用本 skill 将 Agent trace 分类为 `goodcase` 或 `badcase`。核心架构保持三层：

```text
trace -> 特征抽取 -> 规则生成 -> 规则分类器
```

所有方法最终都加载显式 JSON 规则，并由统一规则引擎输出分类结果、分数和命中原因。

## 方法结构

方法按两个维度选择：

| 方法 | 方法族 | 训练输入场景 |
|---|---|---|
| `non_llm_no_train` | 非 LLM | 无训练数据 |
| `non_llm_unlabeled` | 非 LLM | 无标注 trace 作为训练数据 |
| `non_llm_labeled` | 非 LLM | 有标注 trace 作为训练数据 |
| `llm_no_train` | LLM | 无训练数据 |
| `llm_unlabeled` | LLM | 无标注 trace 作为训练数据 |
| `llm_labeled` | LLM | 有标注 trace 作为训练数据 |

旧别名仍可使用：`general -> non_llm_no_train`，`unlabeled -> non_llm_unlabeled`，`labeled -> non_llm_labeled`，`llm ->` 根据训练数据自动选择 LLM 场景。

## 普通实验入口

普通训练、测试、多方法对比使用：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --method non_llm_labeled
```

多方法对比：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods non_llm_no_train,non_llm_unlabeled,non_llm_labeled
```

`run_experiments.py` 只负责训练、测试和方法对比；组件消融统一交给 `run_ablation_study.py`。

## 组件消融入口

当用户要求“最佳消融实验”“组件贡献研究”“以 badcase precision 为主、兼顾 recall 的消融研究”时，优先使用：

```powershell
python .\scripts\run_ablation_study.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods non_llm_labeled
```

该入口会自动运行：

- baseline；
- leave-one-component-out；
- only-one-component；
- targeted subsets；
- precision 优先总榜；
- 推荐候选；
- case 变化和 false positive 摘要。

实验设计说明在 `ablation_experiment_plan.md`。主指标是 `precision(badcase)`，次级指标是 `recall(badcase)`。

## LLM 配置

LLM 相关运行参数统一写入 `scripts/llm_config.yaml`。入口脚本通过 `--llm-config` 指定配置文件；不要再把 provider、model、prompt budget、use_existing_rules 等 LLM 设置分散写成命令行参数。

示例：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --method llm_labeled --llm-config .\scripts\llm_config.yaml
```

`scripts/llm_config.yaml` 常用字段：

```yaml
provider:
model:
temperature: 0.0
use_existing_rules: false
output:
prompt_output:
report_output:
max_samples: 30
max_prompt_chars: 60000
max_trace_chars: 2000
max_dynamic_fields: 80
extra:
```

如果选择 `llm_*` 方法且 `use_existing_rules: false`，脚本会调用 `scripts/llm_rule_prompt.py::call_llm()`。该函数默认为空，需要用户自行接入 provider。

如果希望 Codex/OpenCode 当前 Agent 用自身模型生成规则，而不是让 Python 调用 `call_llm()`，使用专用入口：

```powershell
python .\scripts\run_agent_llm_workflow.py .\data\trace --metadata .\data\metadata.csv --train-split train --eval-split test --methods llm_no_train,llm_unlabeled,llm_labeled --output-dir .\results --report-output .\results\llm_methods_compare.md
```

该入口只生成 prompt 和任务清单，不调用外部模型。Agent 应读取 `results/agent_llm_tasks.md`，用自身 LLM 能力生成规则 JSON，写入清单指定的 `scripts/rules/dynamic/llm/*.json`，再运行清单里的评估命令。评估命令会使用生成的 `agent_llm_eval_config.yaml`，其中 `use_existing_rules: true`，因此不会触发 `call_llm()`。

## Final Answer 识别

`has_final_answer` 只由明确字段或 assistant 消息决定，不使用“任意长字符串”兜底。

证据优先级：

1. `--final-answer-item "key:value"`：用户指定，强证据，顶层字段和嵌套字段都算。
2. `--final-answer-config`：用户配置，强证据。
3. 默认字段扫描命中：中等证据。
4. LLM 发现字段并写入配置：中等证据。
5. 如果都没有命中，则 `has_final_answer` 不作为 good/bad 证据。

示例：

```powershell
python .\scripts\run_experiments.py .\traces --final-answer-item "your_final_answer_key:*"
```

## 规则目录

```text
scripts/rules/
|-- static/
|   |-- general_rules.json
|   `-- final_answer_config.json
`-- dynamic/
    |-- non_llm/
    |   |-- unlabeled_rules.json
    |   `-- labeled_rules.json
    `-- llm/
        |-- no_train_rules.json
        |-- unlabeled_rules.json
        `-- labeled_rules.json
```

清空上一轮动态规则：

```powershell
python .\scripts\clear_dynamic_rules.py
```

可选范围：

```powershell
python .\scripts\clear_dynamic_rules.py --methods non_llm
python .\scripts\clear_dynamic_rules.py --methods llm
python .\scripts\clear_dynamic_rules.py --methods all
```

## 规则汇聚

规则引擎使用同一套汇聚逻辑：

```text
bad_score = 命中的 badcase 规则权重之和
good_score = 命中的 goodcase 规则权重之和
```

默认判定：

```text
bad_score >= 0.60 且 bad_score >= good_score -> badcase
否则如果 good_score >= 0.50 -> goodcase
否则 -> goodcase
```

报告会输出命中规则、组件贡献、final-answer policy、指标和逐条预测结果。

## 辅助分类组件

第三阶段支持 `distance_aux`、`cluster_aux` 和 `ensemble_policy`。这些组件不是新的规则生成方法，而是分类器层面的实验性辅助证据：

- `distance_aux`：基于训练样本的标准化数值特征向量，计算测试样本到训练集中心或 good/bad 类中心的距离。
- `cluster_aux`：基于训练样本的轻量聚类或带标签原型，判断测试样本靠近哪个簇。
- `ensemble_policy`：控制辅助证据是否以及如何改变规则分类器结果。

普通实验默认 `rules_only`，不会启用辅助分类器。需要显式指定：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --method non_llm_labeled --aux-components all --ensemble-policy precision_guard
```

`run_ablation_study.py` 会自动加入 `distance_aux_only`、`cluster_aux_only`、`aux_only`、`rules_plus_aux`、`field_only_plus_aux` 等辅助分类变体，用于判断它们是否值得采纳。

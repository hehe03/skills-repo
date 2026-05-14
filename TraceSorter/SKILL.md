---
name: trace-sorter
description: >-
  对 Agent trace 文件进行 goodcase / badcase 分类，并输出可解释的规则命中原因。
  适用于检查单个或批量 trace、运行训练/测试分离的分类实验、比较非 LLM 与 LLM
  两大方法族，或在三种训练输入场景下生成规则：无训练数据、无标注 trace 训练数据、
  有标注 trace 训练数据。
---

# Trace Sorter

使用本 skill 将 Agent trace 分类为 `goodcase` 或 `badcase`。所有方法最终都产出或加载规则，并由同一个规则引擎执行分类。

## Agent 执行原则

当用户只给出 trace 路径、metadata 路径、训练/测试 split 和实验目标时，Agent 应自行选择脚本和流程，不要求用户说明内部函数或逐步命令。

如果用户要求比较 LLM 方法，且明确或隐含希望“由当前 Agent 自身生成规则”“不要配置外部 LLM/API”“不要依赖 `call_llm()`”，则 Agent 应自动执行以下策略：

1. 使用 `scripts/run_agent_llm_workflow.py` 生成 prompt 和任务清单。
2. 读取任务清单和每个 prompt。
3. 用当前 Agent 自身推理能力生成规则 JSON。
4. 写入任务清单指定的 `scripts/rules/dynamic/llm/*.json`。
5. 运行任务清单中的评估命令。
6. 向用户汇报报告路径和三种 LLM 方法的效果差异。

只有当用户明确要求 Python 自动调用外部模型时，才使用会触发 `call_llm()` 的 LLM 实验路径。

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

旧别名仍可用：`general -> non_llm_no_train`，`unlabeled -> non_llm_unlabeled`，`labeled -> non_llm_labeled`，`llm ->` 根据训练数据自动选择 LLM 场景。

## 代码流程

批量实验入口是 `scripts/run_experiments.py`。流程必须区分训练和测试：

1. 读取测试 trace：位置参数 `trace_path`。
2. 可选读取测试 metadata：`--metadata`。
3. 可选读取训练 trace：`--train-trace-path`，或从 `trace_path` 中用 `--train-split` 选出训练样本。
4. 可选选择测试 split：`--eval-split`。若不指定，测试 `trace_path` 中全部样本。
5. 训练阶段先抽取通用特征和动态字段特征，再根据方法生成规则文件。
6. 测试阶段重新抽取测试样本特征，加载规则文件，对测试样本输出预测和 Markdown 报告。

若既没有 `--train-trace-path`，也没有 `--train-split`，则视为无训练数据，只能使用 `non_llm_no_train` 或 `llm_no_train`。

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

静态通用规则只维护在 `scripts/rules/static/general_rules.json`。动态规则由训练阶段写入 `scripts/rules/dynamic/` 下对应方法族目录。

清空上一轮实验动态规则时运行：

```powershell
python .\scripts\clear_dynamic_rules.py
```

该脚本只把动态规则文件重置为空规则 payload，不删除目录，不修改静态规则。可用 `--methods llm`、`--methods non_llm` 或 `--methods all` 控制清理范围。

## 非 LLM 方法

`non_llm_no_train` 不训练，只加载通用规则。通用规则覆盖解析失败、空 trace、无可观察步骤、错误文本、空结果比例高、重复动作、步数过多、存在 final answer 等信号。

训练阶段不只依赖固定特征表。`features.py` 会同时抽取通用特征和动态字段特征：

- 通用特征：`step_count`、`error_count`、`empty_result_ratio`、`has_final_answer` 等。
- 动态字段特征：根据 trace 实际字段自动生成，例如 `field_exists:status`、`field_text:status`、`field_nonempty_ratio:plan_list[].result`、`field_number_mean:metrics.score`。

训练方法可以直接把动态字段特征写入规则，因此从训练数据中发现的业务字段、字段取值、字段缺失或字段数值规律，会立即在后续测试中生效。

`non_llm_unlabeled` 使用无标注训练 trace。它从训练样本抽取通用特征和动态字段特征，对风险特征计算 90 分位阈值，并为训练集中高频字段的缺失/为空生成批内异常型 badcase 规则。

`non_llm_labeled` 使用有标注训练 trace。它要求训练集同时包含 `goodcase` 和 `badcase`，比较两类样本的通用特征和动态字段特征，生成区分性阈值、字段出现、字段缺失、字段取值和数值大小规则。

## LLM 方法

LLM 方法不直接让模型判定每条测试 trace，而是让模型生成可执行 JSON 规则，再交给 `rule_engine.py` 统一测试。Prompt 会包含训练样本中的动态字段特征；如果 LLM 发现某个业务字段有判别力，应优先用 `field_*:<path>` 特征写成规则，这类规则会立即生效。

LLM 入口在 `scripts/llm_rule_prompt.py`：

- `build_prompt_from_records()` 根据 `no_train`、`unlabeled`、`labeled` 三种场景生成 prompt，并将训练数据压缩为字段概要、特征统计和代表样本。
- `call_llm()` 是预留钩子，默认返回空字符串。选择任一 `llm_*` 方法时，`run_experiments.py` 会默认触发它；若未实现，会报错提示补全。
- `write_llm_rule_report()` 生成中文 `llm_rule_repoert.md`，说明 LLM 发现了哪些规则、final-answer 字段和建议新增特征。

LLM 输出中的 `proposed_features` 只表示“尚未实现的新计算特征建议”，不会在当前运行中立即执行；使用 `field_exists:<path>`、`field_text:<path>`、`field_number_mean:<path>` 等动态字段特征生成的规则会立即执行。

LLM prompt 不默认传入全量 trace。它包含：

- `dataset_summary`：样本数量、标签分布、source/split 分布、长度统计和高频字段路径。
- `fixed_feature_stats`：通用特征统计。
- `label_contrasts`：有标注场景下的 goodcase/badcase 字段差异。
- `selected_samples`：代表样本，包含压缩特征和截断后的 `trace_excerpt`。

有标注场景必须尽量包含正负样例；无标注场景按长度、错误、final-answer 和字段路径差异做多样性采样。可通过 `--llm-max-samples`、`--llm-max-prompt-chars`、`--llm-max-trace-chars`、`--llm-max-dynamic-fields` 控制输入规模。

当用户要求“由当前 Agent 自身生成 LLM 规则，不依赖 Python 的 `call_llm()`”时，优先使用 Agent 专用入口：

```powershell
python .\scripts\run_agent_llm_workflow.py .\data\trace --metadata .\data\metadata.csv --train-split train --eval-split test --methods llm_no_train,llm_unlabeled,llm_labeled --output-dir .\results --report-output .\results\llm_methods_compare.md
```

该入口只生成 prompt 和任务清单，不调用外部模型。Agent 应按 `results/agent_llm_tasks.md` 逐个读取 prompt，用自身 LLM 能力生成规则 JSON，写入清单指定的 `scripts/rules/dynamic/llm/*.json`，然后运行清单里的评估命令。评估命令会带 `--llm-use-existing-rules`，因此不会触发 `call_llm()`。

`call_llm()` 签名：

```python
def call_llm(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    extra_args: Dict[str, Any] | None = None,
) -> str:
    return ""
```

Codex/OpenCode 作为 Agent 使用本 skill 时，可以由当前 Agent 自己阅读 prompt 并生成规则；如果希望 Python 脚本自动调用模型，需要在 `call_llm()` 中接入具体 provider。

## Final Answer 识别

`has_final_answer` 只由明确字段或 assistant 消息决定，不再使用“任意长字符串”兜底。

证据优先级：

1. `--final-answer-item "key:value"`：用户指定，强证据，顶层和嵌套字段都算。
2. `--final-answer-config`：用户配置，强证据，按 `top_level_keys` 和 `nested_keys` 执行。
3. 默认字段扫描命中：中等证据。
4. LLM 发现字段并写入配置：中等证据。
5. 若都没有命中，则 `has_final_answer` 不作为 good/bad 证据。

示例：

```powershell
python .\scripts\run_experiments.py .\traces --final-answer-item "your_final_answer_key:*"
```

上面的 `your_final_answer_key` 是占位字段名；实际使用时应替换成你的业务 trace 中代表最终结果的字段。`key:value`、`key: value`、`key : value` 都兼容，`*` 表示任意字符。

## 常用命令

无训练数据，运行非 LLM 通用规则：

```powershell
python .\scripts\run_experiments.py .\traces --method non_llm_no_train
```

用同一目录中的 `split=train` 训练、`split=test` 测试：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --method non_llm_unlabeled
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --method non_llm_labeled
```

使用独立训练目录和测试目录：

```powershell
python .\scripts\run_experiments.py .\test_traces --metadata .\test_metadata.csv --train-trace-path .\train_traces --train-metadata .\train_metadata.csv --method non_llm_labeled
```

比较多个方法：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods non_llm_no_train,non_llm_unlabeled,non_llm_labeled
```

运行 LLM 有标注方法：

```powershell
python .\scripts\run_experiments.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --method llm_labeled --llm-provider custom --llm-model my-model
```

只生成 LLM prompt：

```powershell
python .\scripts\llm_rule_prompt.py .\traces --metadata .\metadata.csv --split train --training-scenario labeled --output llm_rule_prompt.md --no-report
```

## 报告

`run_experiments.py` 默认把报告写到 `--output-dir`，文件名为“方法 + 时间”。`--output` 可以指定完整 Markdown 文件路径，并覆盖自动命名。

报告会说明：

- 方法族。
- 训练输入场景。
- 训练来源和训练样本数。
- 测试来源和测试样本数。
- final-answer policy。
- 指标和每条测试样本的预测结果。

## 规则汇聚

所有方法最终都使用同一规则引擎：

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

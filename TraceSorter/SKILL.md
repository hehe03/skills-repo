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

生成 LLM 规则 prompt，并默认输出中文报告 `llm_rule_repoert.md`：

```powershell
python .\scripts\llm_rule_prompt.py <trace_json_or_dir> --metadata <metadata.csv> --split train --output llm_rule_prompt.md
```

如果已经保存 LLM 返回结果，可用 `--llm-output` 让报告说明 LLM 实际发现了哪些规则：

```powershell
python .\scripts\llm_rule_prompt.py <trace_json_or_dir> --metadata <metadata.csv> --split train --output llm_rule_prompt.md --llm-output llm_response.json --report-output llm_rule_repoert.md
```

## 最终回答识别机制

`has_final_answer` 不再使用“trace 中存在长度大于等于 80 的字符串”这类宽松兜底。它先确定本次运行是否采用 final-answer 字段，再决定相关规则是否参与分类。

顶层字段和内部字段的区别：

- **顶层字段**：只检查 trace 根对象直接拥有的字段，例如 `trace["final_answer"]`。它更适合作为业务最终结果字段，误判风险较低。
- **内部字段**：递归检查 trace 任意内部节点中的字段，例如某个 step、message、event 里的 `answer`。它覆盖面更广，但可能命中中间工具结果，误判风险更高。

证据强度规则：

1. 如果用户通过 `--final-answer-item` 指定键值对模式，则这些 item 在顶层和内部递归位置都算作强证据，`final_answer_evidence_strength=strong`。
2. 如果用户通过 `--final-answer-config` 指定字段，则按配置中的 `top_level_keys` 和 `nested_keys` 执行；若希望顶层和内部都算，需要把字段同时写入两组。
3. 如果用户没有指定字段，脚本会先扫描输入 trace，检查可见字段是否命中默认候选字段。命中后采用这些字段作为中等强度证据，`final_answer_evidence_strength=medium`。
4. 如果是 LLM 方法，可以让大模型先判断能否找到业务 final-answer 字段，并将字段写入配置，设置 `evidence_source=llm`。这类字段也作为中等强度证据。
5. 如果没有用户指定、没有默认命中、也没有 LLM 发现，则 `has_final_answer` 不作为 good/bad 判定证据，相关规则不会命中。

默认候选来源：

1. 顶层字段候选：`final`、`final_answer`、`final_response`、`answer`、`response`、`output`、`result`。
2. 内部字段候选：`final_answer`、`final_response`、`answer`。
3. assistant 消息候选：`role=assistant` 且 `content` 非空。

业务相关字段通过配置加入，不要改硬编码。默认配置模板位于：

```text
scripts/rules/static/final_answer_config.json
```

快速添加业务 item：

```powershell
python .\scripts\run_experiments.py <trace_json_or_dir> --final-answer-item "business_result:*" --final-answer-item "status: *success*"
```

上述命令会同时检查 `trace["business_result"]` 这类顶层字段，以及任意内部节点中的 `business_result` / `status`。`*` 表示任意字符，`key:value`、`key: value`、`key : value` 都兼容。

使用完整配置：

```powershell
python .\scripts\run_experiments.py <trace_json_or_dir> --final-answer-config .\final_answer_config.json
```

配置示例：

```json
{
  "top_level_keys": ["final_answer", "business_result"],
  "nested_keys": ["final_answer", "business_result", "summary_text"],
  "final_answer_items": ["business_result:*", "status: *success*"],
  "assistant_roles": ["assistant"],
  "assistant_content_keys": ["content"],
  "min_chars": 1,
  "evidence_source": "user"
}
```

实验报告会在方法说明区域汇总 final-answer policy，例如：

```text
user/strong:business_result
default/medium:top_level:final_answer,nested:answer
none/none:none
```

## 通用规则汇聚方法

通用规则层使用单一加权方法：所有命中的 badcase 规则累加到 `bad_score`，所有命中的 goodcase 规则累加到 `good_score`。

通用规则的权重/阈值已按强弱信号重新核查：

- 解析失败、空 trace 属于硬失败信号，单独即可超过 badcase 阈值。
- 错误文本是强风险信号。
- 缺少最终回答、空结果比例高属于结果缺失风险。
- 重复动作和步数过多属于效率/循环风险。
- goodcase 支持规则只提供正向证据，不覆盖硬失败信号。

如果有 metadata，会额外输出指标；没有 metadata 时，只输出预测结果。

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

运行批量实验，并生成以“实验方法+时间”命名的 Markdown 报告。metadata 不是必选项，只有有标注/监督规则生成或评估才需要：

```powershell
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer general
```

实验前生成无标注或有标注动态规则：

```powershell
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer unlabeled --generate-dynamic-rules unlabeled
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer labeled --generate-dynamic-rules labeled
python .\scripts\run_experiments.py <trace_json_or_dir> --metadata <metadata.csv> --rule-layer all --generate-dynamic-rules auto
```

`run_experiments.py` 默认 `--generate-dynamic-rules auto`。当 `--rule-layer` 或 `--methods` 包含 `unlabeled` 时，且可用样本不少于 3 条，会自动生成 `unlabeled_rules.json`；当包含 `labeled` 且训练/可用标注样本同时包含 `goodcase` 和 `badcase` 时，会自动生成 `labeled_rules.json`。

常用参数：

- `--bad-threshold`：判为 badcase 的最低风险分，默认 `0.60`。
- `--good-threshold`：判为 goodcase 的最低支持分，默认 `0.50`。
- `--eval-split`：指定 metadata `split` 列中用于评估的样本，例如 `test`、`train`、`validation`。不指定时使用全部样本。
- `--methods`：选择多个方法做对比实验，例如 `general,unlabeled,llm` 或 `all`。
- `--output-dir`：自动命名报告的输出目录。
- `--output`：明确指定单个 Markdown 报告文件路径；设置后会覆盖 `--output-dir` 自动命名。
- `--max-rows`：Markdown 报告中最多展示的样本行数。
- `--final-answer-item`：追加业务相关最终回答键值对模式，格式为 `key:value`，可重复传入，`*` 匹配任意字符。
- `--final-answer-config`：使用 JSON 配置控制最终回答识别。

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

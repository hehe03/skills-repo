# TraceSorter 与 communication 分类方法对比

本文比较当前项目 `TraceSorter` 中的无标注、有标注方法，与 `D:\code\github\hehe03\skills-repo\communication` 项目下两个脚本的差异：

- `communication\unsupervised\classify_unsupervised.py`
- `communication\supervised\classify_supervised.py`

## 总体结论

`TraceSorter` 的核心定位是“从 trace 样本中萃取规则，并把规则沉淀为可复用的 JSON 规则库”，随后由统一的规则引擎做分类。

`communication` 项目的两个脚本更像“直接分类器”：它们从 trace 中抽取一组特征，然后用人工公式、批内统计或监督距离模型直接给出预测结果，并不产出独立的动态规则文件。

因此，两者最大的区别不是是否都用了特征，而是：

| 维度 | TraceSorter | communication |
|---|---|---|
| 方法目标 | 生成可解释、可复用、可组合的规则 | 对当前数据集直接预测 goodcase/badcase |
| 输出核心 | `rules/dynamic/*.json` 规则文件 + 实验报告 | 预测 TSV + 指标 Markdown |
| 分类方式 | 规则命中后累加 good/bad 权重，再按阈值判定 | 特征打分、风险分数、中心性修正或类中心距离 |
| 适配范围 | 尽量面向通用 Agent trace 结构 | 明显面向 `plan_list`、`query`、任务执行类 trace |
| 可解释性 | 解释为“哪些规则命中、权重是多少” | 解释为分数、风险、最近样本或摘要字段 |
| 是否沉淀知识 | 是，沉淀为静态规则和动态规则 | 否，主要在脚本公式或训练统计中即时使用 |

## 无标注方法对比

### TraceSorter 的无标注方法

当前项目的无标注方法位于 `scripts/rule_generation.py` 的 `generate_unlabeled_rules()`。

它的流程是：

1. 读取无标注 trace 样本。
2. 通过 `scripts/features.py` 抽取通用结构特征，例如：
   - `step_count`
   - `repeated_action_count`
   - `max_consecutive_same_action`
   - `error_count`
   - `empty_result_count`
   - `empty_result_ratio`
   - `has_final_answer`
3. 对风险类数值特征计算样本集合内的 90 分位阈值。
4. 将高于阈值的模式写成 badcase 规则，例如 `unlabeled_high_error_count`。
5. 如果样本集合中存在可识别的最终回答字段，则额外生成“缺失最终回答”的 badcase 规则。
6. 将规则保存到 `scripts/rules/dynamic/non_llm/unlabeled_rules.json`。
7. 实验运行时，规则引擎读取静态规则和动态规则，对每条 trace 执行规则匹配和权重汇聚。

这个方法的重点是“从无标注样本中生成规则阈值”，不是直接学习一个分类模型。它默认 badcase 风险通常表现为异常高的错误数、空结果比例、重复动作或过长执行链路。

### communication 的无监督方法

`communication\unsupervised\classify_unsupervised.py` 是一个直接分类脚本。它不会生成动态规则文件，而是在一次运行中完成特征抽取、打分、预测和指标输出。

它的主要流程是：

1. 读取 trace 和 metadata。
2. 重点解析 `plan_list`，把其中每个 task 看作执行步骤。
3. 抽取更偏业务和执行行为的特征，例如：
   - `step_count_log`
   - `unique_task_ratio`
   - `unique_command_ratio`
   - `result_nonempty_ratio`
   - `avg_result_log`
   - `final_result_log`
   - `max_same_run`
   - `max_loop_repeats`
   - `revisit_ratio`
   - `missing_command_ratio`
   - `has_outline`
   - `query_task_overlap`
   - `task_entropy`
   - `command_entropy`
   - `unique_result_ratio`
   - `max_same_result_run`
   - `final_result_nonempty`
   - `result_entropy`
4. 用 `behavior_prior()` 计算偏 goodcase 的行为先验分。
5. 用 `bad_risk_score()` 计算偏 badcase 的风险分。
6. 用批内 robust scaling 和样本中心性计算 `centrality_scores()`，对最终分数做修正。
7. 如果 bad 风险分超过阈值，直接判为 badcase；否则根据最终分数和阈值判定 goodcase/badcase。

这个方法比 TraceSorter 的无标注规则更“模型化”。它虽然没有训练标签，但有一套手写评分函数，且引入了批内中心性来判断样本是否像当前 batch 的主流模式。

### 无标注方法差异

| 维度 | TraceSorter 无标注规则 | communication 无监督分类 |
|---|---|---|
| 本质 | 从样本分布萃取阈值规则 | 手写评分函数 + 批内统计 |
| 是否生成规则文件 | 是，生成 `unlabeled_rules.json` | 否 |
| 是否需要 metadata | 不必需，除非需要标签评估或 split 选择 | 必需，脚本要求 metadata CSV |
| 对 trace 结构的假设 | 尝试兼容 `plan_list`、`steps`、`events`、`messages`、`turns`、`spans`、`actions` 等结构 | 强依赖 `plan_list` 和 task 字段 |
| 阈值来源 | 无标注样本的 90 分位 + 固定下限 | 人工设定公式参数和命令行阈值 |
| 解释方式 | 命中了哪些规则 | 行为先验、风险分、中心性、摘要 reason |
| 优点 | 规则可沉淀、可审计、可迁移到后续实验 | 特征更丰富，能表达连续风险和相对异常 |
| 风险 | 分位阈值简单，样本少或分布偏时规则不稳定 | 公式权重人工设定，且更依赖特定 trace schema |

## 有标注方法对比

### TraceSorter 的有标注方法

当前项目的有标注方法位于 `scripts/rule_generation.py` 的 `generate_labeled_rules()`。

它的流程是：

1. 读取带标签的 trace 样本。
2. 优先使用 `split=train` 的样本；如果没有 train，则使用全部有标签样本。
3. 分别抽取 goodcase 和 badcase 的特征。
4. 对风险类特征比较 badcase 均值和 goodcase 均值。
5. 如果 badcase 均值明显高于 goodcase 均值，则在两类均值中点生成阈值规则。
6. 如果 goodcase 中最终回答出现率明显高于 badcase，则生成：
   - 缺失最终回答的 badcase 规则；
   - 存在最终回答的 goodcase 支持规则。
7. 将规则保存到 `scripts/rules/dynamic/non_llm/labeled_rules.json`。
8. 分类时仍然通过统一规则引擎进行规则命中和权重汇聚。

这个方法的监督信息只用于“萃取规则”，而不是直接拟合一个分类器。它追求的是把标注样本中的差异转化为人可读、可复用的规则。

### communication 的监督方法

`communication\supervised\classify_supervised.py` 是少样本监督分类器。

它的主要流程是：

1. 从 metadata 中强制读取 `train` 和 `test` 两个 split。
2. 要求 train 中至少包含一个 goodcase 和一个 badcase。
3. 从 `plan_list` 抽取特征，例如：
   - `step_count_log`
   - `unique_task_ratio`
   - `unique_command_ratio`
   - `result_nonempty_ratio`
   - `avg_result_log`
   - `final_result_log`
   - `max_same_run`
   - `max_loop_repeats`
   - `revisit_ratio`
   - `missing_command_ratio`
   - `has_outline`
   - `outline_count_log`
   - `query_task_overlap`
   - `task_entropy`
   - `command_entropy`
4. 基于 train 样本计算每个特征的 median 和 scale，做 robust scaling。
5. 分别计算 goodcase 和 badcase 的类中心。
6. 对 test 样本计算到 good 类中心和 bad 类中心的距离。
7. 用 `sigmoid((distance_to_bad - distance_to_good) / distance_scale)` 得到 goodcase 分数。
8. 根据阈值输出预测，并记录最近的 good/bad 训练样本。

这个方法是典型的特征空间监督分类：标签直接参与构建类中心，预测时根据距离判定类别。

### 有标注方法差异

| 维度 | TraceSorter 有标注规则 | communication 监督分类 |
|---|---|---|
| 本质 | 从标注样本中抽取差异规则 | 用标注样本拟合特征空间类中心 |
| 标签用途 | 计算 good/bad 特征差异，生成规则 | 建立 good/bad 类中心并直接分类 |
| 是否生成规则文件 | 是，生成 `labeled_rules.json` | 否 |
| train/test 要求 | 有 train 则优先用 train；没有 train 可退化为全部有标签样本 | 必须同时存在 train 和 test |
| 类别要求 | 生成有区分度规则时需要 good/bad 都存在 | train 必须同时有 good/bad |
| 分类机制 | 规则命中、权重累加、阈值判定 | robust scaling + 类中心距离 + sigmoid |
| 解释方式 | 哪些标签萃取规则命中 | score、confidence、nearest_good、nearest_bad |
| 优点 | 更容易审计和迁移，规则可以被人工复核或组合 | 能利用多维特征整体相似度，少样本下也能直接预测 |
| 风险 | 只捕捉单特征阈值差异，表达能力较弱 | 可解释性弱于规则，且更依赖 train/test 切分质量 |

## 特征体系差异

`TraceSorter` 的特征更偏通用 Agent trace 健康度：

| 特征方向 | 示例 |
|---|---|
| 结构是否可解析 | `parse_error`, `is_empty_trace`, `has_steps` |
| 执行步骤规模 | `step_count`, `action_count` |
| 动作重复和循环 | `repeated_action_count`, `max_consecutive_same_action`, `unique_action_ratio` |
| 错误信号 | `error_count`, `has_error_text` |
| 结果缺失 | `empty_result_count`, `empty_result_ratio`, `nonempty_result_ratio` |
| 最终回答 | `has_final_answer`, `final_answer_evidence_strength`, `final_answer_adopted_fields` |
| 文本规模 | `text_chars` |

`communication` 的特征更偏某类任务执行过程，尤其是假设 trace 中有 `plan_list`：

| 特征方向 | 示例 |
|---|---|
| 任务计划结构 | `step_count_log`, `unique_task_ratio`, `task_entropy` |
| command 使用 | `unique_command_ratio`, `missing_command_ratio`, `command_entropy` |
| result 质量 | `result_nonempty_ratio`, `avg_result_log`, `final_result_log` |
| 循环和重复 | `max_same_run`, `max_loop_repeats`, `revisit_ratio` |
| 业务任务特征 | `has_outline`, `outline_count_log` |
| query 与任务相关性 | `query_task_overlap` |
| 结果多样性 | `unique_result_ratio`, `max_same_result_run`, `result_entropy` |

因此，`communication` 的特征更丰富，但迁移性更弱；`TraceSorter` 的特征较保守，但更适合做通用 SKILL 的默认基础。

## 规则法与特征法的边界

这两个项目都需要抽取特征，但方法边界仍然清楚：

| 类型 | 判断标准 | 本项目归属 |
|---|---|---|
| 规则法 | 特征只作为规则条件输入，最终知识以规则形式保存，可独立查看和复用 | TraceSorter |
| 特征打分法 | 特征直接进入公式、距离、模型或分数函数，输出预测，不产出规则库 | communication |

换句话说，是否使用特征不是分界线；是否把样本规律沉淀为显式规则，才是分界线。

## 输出与实验能力差异

| 维度 | TraceSorter | communication |
|---|---|---|
| 批量实验入口 | `scripts/run_experiments.py` | 两个分类脚本分别运行 |
| 多方法对比 | 支持 `--methods general,unlabeled,labeled,llm` | 不支持同一脚本内多方法对比 |
| metadata 是否必需 | 仅标注方法、split 筛选或指标评估需要 | 两个脚本都要求 metadata |
| split 筛选 | `--eval-split` 指定则筛选对应 split；不指定则全部样本 | 无监督脚本支持 `--split train/test`；监督脚本固定 train/test |
| 动态规则输出 | `scripts/rules/dynamic/*.json` | 无 |
| 报告命名 | 默认按“实验方法+时间”生成 Markdown | 默认固定 metrics 文件名 |
| LLM 规则路线 | 预留 `llm_rule_prompt.py::call_llm()`，选择 llm 方法时触发 | 无 |

## 适用场景建议

如果目标是构建一个可长期维护的 SKILL，并让 Agent 在不同 trace 数据源上复用分类逻辑，`TraceSorter` 当前路线更合适。它的优势是规则可读、可审计、可版本化，也方便把 LLM 或人工发现的业务规则逐步加入规则库。

如果目标是在 `communication` 这类固定结构数据上尽快获得较强的分类效果，尤其 trace 都有稳定的 `plan_list`、`query`、`command`、`result` 字段，那么 `communication` 的无监督和监督脚本更直接。它们的特征表达更细，监督方法还能利用多维特征空间中的整体相似性。

## 可以借鉴的改进方向

`TraceSorter` 可以从 `communication` 项目借鉴以下能力：

1. 引入更多连续特征，例如任务多样性、结果多样性、query 与任务重合度、结果长度分布。
2. 在无标注方法中增加批内相对异常分数，而不仅是单特征 90 分位阈值。
3. 在有标注方法中增加多特征组合规则或类中心辅助分析，但仍然把最终可解释知识写成规则。
4. 在报告中补充“规则生成依据”，例如每条动态规则对应的 good/bad 均值或无标注分位数。

`communication` 可以从 `TraceSorter` 项目借鉴以下能力：

1. 把人工公式或监督差异沉淀成独立规则文件，便于复用和审计。
2. 降低对 `plan_list` 的强依赖，增加对通用 trace 字段的兼容。
3. 增加 final answer 字段发现和用户指定机制，避免把“最终结果”固定理解为最后一个 task 的 result。
4. 增加多方法统一实验入口，方便同一批样本上比较不同方法。

## 简短结论

`TraceSorter` 是规则萃取与规则执行框架，适合作为可扩展 SKILL。

`communication` 是面向特定 trace schema 的分类实验脚本，适合在固定数据格式上快速得到预测效果。

两者不是简单的新旧版本关系，而是目标不同：前者重在可解释规则资产，后者重在当前数据集上的直接分类能力。

# TraceSorter 消融实验方案

本文档定义当前项目的默认消融研究流程。目标不是穷举所有规则组合，而是用较少实验回答三个问题：

1. 哪些组件能稳定提升 badcase precision。
2. 哪些组件在 precision 不下降的前提下补充 recall。
3. 哪些组件容易引入 false positive，应降权、禁用或保留为实验组件。

## 核心指标

本方案以 badcase 为正类。

主指标：

- `precision(badcase)`：第一优先级。它衡量被判为 badcase 的样本中有多少是真的 badcase。

次级指标：

- `recall(badcase)`：第二优先级。它衡量真实 badcase 中有多少被识别出来。
- `f1(badcase)`：辅助指标，用于 precision 和 recall 较平衡时比较。
- `fp`：重点观察。`fp` 越多，说明 goodcase 被误判成 badcase 越多。
- `fn`：兼顾观察。`fn` 越多，说明漏掉 badcase 越多。
- `changed vs baseline`：用于判断某个组件调整的影响范围。

排序原则：

```text
precision 越高越好
precision 相同或接近时，recall 越高越好
recall 相近时，f1 越高越好
仍相近时，fp 更少者优先
```

## 实验阶段

### 第一阶段：Baseline

运行完整方法，作为后续所有消融的对照。

回答的问题：

- 当前方法的 precision / recall / fp / fn 是多少。
- 当前规则组件有哪些。
- 每个组件命中了多少 case，贡献了多少 good/bad 分数。

### 第二阶段：Leave One Component Out

逐个禁用一个组件，其它组件保持不变。

回答的问题：

- 去掉某组件后 precision 是否上升。
- 去掉某组件后 recall 是否明显下降。
- 哪些组件是主要误伤来源。
- 哪些组件虽然单独不强，但对 recall 有互补价值。

判断方式：

```text
禁用后 precision 上升且 recall 基本不降 -> 该组件可能应降权或默认禁用。
禁用后 precision 下降 -> 该组件对 badcase 判断有正贡献。
禁用后 recall 大幅下降 -> 该组件负责抓住一批独特 badcase。
禁用后指标基本不变 -> 该组件可能冗余。
```

### 第三阶段：Only One Component

每次只保留一个组件。

回答的问题：

- 哪个组件单独就有较高 precision。
- 哪个组件只适合作为辅助证据。
- 哪个组件命中多但误伤高。

判断方式：

```text
单组件 precision 高、fp 少 -> 适合保留为高置信组件。
单组件 recall 高、precision 低 -> 适合降权或只在高召回模式使用。
单组件几乎不命中 -> 需要检查是否冗余或数据集中不适用。
```

### 第四阶段：Targeted Subsets

基于组件类型运行少量固定组合，避免盲目全组合搜索。

默认组合包括：

- `static_only`：只保留静态通用规则。
- `dynamic_only`：只保留训练或 LLM 生成的动态规则。
- `without_final_answer`：禁用 final-answer 相关规则。
- `without_dynamic_fields`：禁用动态字段规则。
- `without_generated_dynamic`：禁用训练或 LLM 生成的动态规则。
- `numeric_only`：只保留数值阈值类规则。
- `field_only`：只保留动态字段类规则。
- `distance_aux_only`：只使用距离辅助分类器。
- `cluster_aux_only`：只使用聚类辅助分类器。
- `aux_only`：只使用距离和聚类辅助分类器。
- `rules_plus_aux`：完整规则集叠加辅助分类器，并使用 precision-first 的 ensemble policy。
- `field_only_plus_aux`：字段类规则叠加辅助分类器，用于观察辅助分类器是否能在稳健字段规则上补 recall。

回答的问题：

- 当前数据上是否主要依赖静态规则。
- 动态字段规则是否带来泛化风险。
- final-answer 规则是否过强。
- 字段规则和数值规则哪个更可靠。
- 辅助分类器是独立有效，还是只能作为规则分类器的弱补充。
- ensemble policy 是否能在不显著增加 FP 的前提下补充 badcase recall。

### 第五阶段：推荐候选

脚本会根据指标自动给出候选方案。

默认推荐条件：

```text
precision >= baseline precision
recall >= baseline recall - recall_drop_tolerance
```

其中 `recall_drop_tolerance` 默认是 `0.05`。这表示允许为了提升 precision 牺牲少量 recall，但不接受 recall 大幅下降。

## 运行入口

推荐使用新入口：

```powershell
python .\scripts\run_ablation_study.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods non_llm_labeled
```

多方法一起研究：

```powershell
python .\scripts\run_ablation_study.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods non_llm_no_train,non_llm_labeled
```

使用已有 LLM 规则，不触发 `call_llm()`：

```powershell
python .\scripts\run_ablation_study.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods llm_labeled --llm-config .\scripts\llm_config.yaml
```

此时请在 `scripts/llm_config.yaml` 或传入的配置文件中设置 `use_existing_rules: true`。

指定输出：

```powershell
python .\scripts\run_ablation_study.py .\traces --metadata .\metadata.csv --train-split train --eval-split test --methods non_llm_labeled --report-output .\results\ablation_summary.md
```

## 总报告内容

总报告会聚合以下内容：

- 实验配置。
- 方法和数据来源。
- 按 precision 优先排序的总榜。
- 每个方法的 baseline 指标。
- 每个组件的 leave-one-out 和 only-one-component 结果。
- targeted subsets 结果。
- 相对 baseline 新修正的 case。
- 相对 baseline 新弄错的 case。
- false positive case 摘要。
- 推荐采用、继续观察、暂不采用的候选方案。

## 采纳标准

建议默认采用的组件或组合需要满足：

- precision 不低于 baseline。
- recall 不低于 baseline 超过容忍范围。
- fp 不增加，或新增 fp 能被解释并接受。
- 在不同 source 或 split 上表现稳定。
- 规则数量没有显著膨胀。

建议保留为实验组件的情况：

- precision 高但 recall 太低。
- 只在部分 source 上有效。
- 命中较少但解释性强。
- 与其它组件组合后才有价值。

建议禁用或降权的情况：

- 禁用后 precision 明显提升。
- 该组件贡献了大量 false positive。
- 组件主要依赖 source-specific 字段，跨 source 不稳。
- only-one-component 时 precision 很低且命中很多。

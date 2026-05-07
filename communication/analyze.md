# 当前分类结果分析与改进建议

## 结果概览

当前以 badcase 为正类，各方法的 F1-score 如下：

| 方法 | 设置 | badcase F1 |
| --- | --- | ---: |
| 规则方法 `classify_traces.py` | 不使用循环检测 | 0.556 |
| 规则方法 `classify_traces.py` | `--repeat-threshold=2` | 0.703 |
| 直接 LLM 分类 `classify_llm.py` | 默认提示词 | 0.000 |
| 无监督方法 `unsupervised/classify_unsupervised.py` | 默认参数 | 0.421 |
| 少量监督方法 `supervised/classify_supervised.py` | `--threshold=0.85` | 0.631 |

目前最强基线是规则方法加循环检测，badcase F1 为 0.703。其余方法都没有超过该基线。

## 方法分析

### 1. 规则方法

规则方法在加入 `--repeat-threshold=2` 后从 0.556 提升到 0.703，说明你的 badcase 中有相当一部分具有明确的行为异常形态，尤其是连续重复任务或循环任务。

这个结果很重要：它说明 badcase 并不完全依赖语义理解，很多失败可以从 trace 执行结构中直接识别。规则方法的优势是高确定性、可解释、不会受 LLM 输出格式影响；缺点是只能覆盖已经显式定义的失败模式，对隐性的“完成质量差”“答非所问”“表面完成但结果不可用”覆盖不足。

### 2. 直接 LLM 分类

`classify_llm.py` 把所有样本都识别为 goodcase，badcase F1 为 0。这通常有几类原因：

- 提示词对 badcase 的定义不够强，模型倾向认为“有执行步骤、有输出”就是 goodcase。
- 没有要求模型先列出失败证据再给结论，导致模型直接做宽松判断。
- goodcase/badcase 的业务边界没有例子，模型默认采用通用助理评估标准，而不是你的数据集标准。
- 如果 batch 中有多个样本，模型可能被整体上下文带偏，倾向输出一致标签。
- 结果解析可能没有问题，因为“全部 goodcase”更像判定偏置，而不是解析失败。

这说明单轮“提示词 + trace -> 标签”的方式太弱，需要把分类改成更强约束的审查流程，尤其要让模型主动寻找 badcase 证据。

### 3. 无监督方法

无监督方法 F1 为 0.421，低于规则基线，说明当前特征评分没有学到真实 badcase 边界。无监督方法的问题在于，它主要依赖行为形态和分数阈值，但 badcase 可能有多种异质类型：

- 结构型 badcase：循环、重复、空结果、缺少关键步骤。
- 语义型 badcase：步骤看似正常，但结果不满足 query。
- 质量型 badcase：生成了内容，但内容过短、泛泛、不可执行。
- 路径型 badcase：任务拆解方向错误，后续步骤都建立在错误理解上。

无监督方法对结构型 badcase 有帮助，但对语义型和质量型 badcase 较弱；如果数据集中 goodcase 和 badcase 的 trace 结构相似，它很容易误判。

### 4. 少量监督方法

监督方法在阈值 0.85 时 F1 达到 0.631，说明提高判 goodcase 的门槛有助于提升 badcase 召回，但仍未超过规则基线。主要原因可能是：

- 当前特征仍以结构特征为主，语义信息不足。
- 原型分类过于简单，只能学习“整体像不像”，不能学习多类 badcase 模式。
- badcase 内部差异大，一个 bad 原型无法覆盖所有失败类型。
- 阈值提高后可能提升 badcase 召回，但会牺牲 goodcase precision，F1 仍受限。

## 为什么现有方法都未超过规则基线

规则基线之所以强，是因为它命中了数据中最稳定的一类失败模式：循环和重复。其他方法试图用更泛化的特征或 LLM 判断覆盖更多情况，但目前没有充分利用这条强规则，也没有可靠识别语义型 badcase。

可以把当前问题理解为：强结构规则已经能抓住一部分 badcase，但模型类方法还没有学会在规则之外找到新的 badcase，因此没有带来增量收益。

## 提升建议

### 1. 采用级联策略，而不是替代规则基线

建议把 `classify_traces.py --repeat-threshold=2` 作为第一层 hard rule。凡是命中循环/重复规则的样本直接判 badcase。

第二层只处理未命中规则的样本，再使用 LLM 或 Agent 方法识别语义型 badcase。这样新方法只需要贡献规则之外的增量，不需要重新学习已经很强的结构规则。

推荐流程：

```text
if 命中循环/重复规则:
    badcase
else:
    使用 Agent/LLM 做语义审查
```

### 2. 直接 LLM 分类改成 Agent 式审查

不要让模型直接输出标签，而是要求它先完成以下步骤：

1. 理解用户 query 的目标。
2. 检查 plan_list 是否覆盖目标。
3. 检查每个关键步骤 result 是否有效。
4. 检查是否存在循环、重复、空结果、无关动作。
5. 判断最终是否真的完成目标。
6. 基于证据输出 goodcase/badcase。

这可以显著降低模型“默认乐观”的倾向。新建的 `classify_agent.py` 就是为这个方向准备的。

### 3. 为 LLM 提供少量反例和边界规则

建议在 LLM prompt 中加入 3 到 5 条 badcase 判定准则，例如：

- 只要出现连续重复或循环，即使有输出，也优先判 badcase。
- 生成了大纲但大纲为空、过短、泛泛，仍可判 badcase。
- 每一步都有 result 不等于完成任务，必须检查 result 是否对 query 有实际贡献。
- 如果 trace 只是搜索/重试/工具调用，没有形成最终有效答案，判 badcase。
- 若 task_name 看似正确但 result 与 task_name 不匹配，判 badcase。

### 4. 监督方法从单原型升级为多原型

当前 `classify_supervised.py` 只有 good 原型和 bad 原型，建议改成 badcase 多原型：

- 循环重复型 badcase。
- 输出缺失型 badcase。
- 答非所问型 badcase。
- 质量不足型 badcase。

即使每类只有少量样本，也比一个 bad 原型更合理。预测时只要接近任一 bad 原型，就判 badcase。

### 5. 输出 badcase 细分类原因

无论是 LLM 还是 Agent 方法，都建议不仅输出 label，还输出 badcase reason，例如：

```json
{
  "label": "badcase",
  "badcase_type": "loop",
  "evidence": ["连续两次执行相同搜索任务", "没有产生最终有效答案"]
}
```

这可以用于后续统计：哪些 badcase 类型最多、哪些类型被漏判最多。没有这个分析，很难继续提升 F1。

### 6. 优先优化 badcase recall，再控制 precision

当前目标是提升 badcase F1，而 badcase 漏判通常更严重。建议先让 Agent 方法偏保守：只要出现明确失败证据就判 badcase。然后再通过阈值、few-shot 示例或规则白名单减少 goodcase 误杀。

## 推荐下一步实验

1. 使用规则方法 `--repeat-threshold=2` 作为 hard baseline。
2. 对未命中规则的样本运行 `classify_agent.py`。
3. 统计 Agent 方法新增识别出的 badcase 数量，以及误杀 goodcase 数量。
4. 对 Agent 输出的 `badcase_type` 做分布统计，找出主要漏判类型。
5. 将高频漏判类型固化成规则或 few-shot 示例。

最终目标不是让 LLM 完全替代规则，而是形成组合分类器：

```text
强规则兜底结构型 badcase + Agent 审查语义型 badcase + 少量监督校准边界
```

这个方向更有机会超过当前 0.703 的规则基线。


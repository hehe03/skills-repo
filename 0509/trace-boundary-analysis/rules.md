# 规则配置与问题收集

本文档是规则法的唯一规则来源。`scripts/classify_rule.py` 会读取本文档中的“规则配置区”，不要把业务关键词直接硬编码进 Python。

## 1. 需要用户补充的问题

请优先补充这些问题。补充后，再把可以执行的部分沉淀到第 2 节的规则配置中。

### 1.1 目标完成信号

1. 哪些 `task_name` 关键词表示 trace 已经完成用户目标？
2. 是否所有任务都必须最终进入“大纲生成”才算 goodcase？
3. 如果用户只要求要素澄清或要素推荐，是否允许没有大纲生成？

当前状态：

- 暂不假设“生成大纲”是必需规则。
- `success_task_keywords` 默认留空。

### 1.2 失败信号

1. 工具结果中哪些词一定表示工具失败？
2. 哪些失败是允许继续执行下一步的？
3. 哪些失败必须先向用户展示并确认，不能继续调用新工具？

当前候选失败词：

- `error`
- `failed`
- `exception`
- `traceback`
- `null`
- `none`
- `失败`
- `报错`
- `异常`
- `未找到`
- `无权限`
- `超时`
- `无法获取`
- `无结果`
- `没有查询到`
- `请求失败`

### 1.3 重复调用

1. 同一个 `task_name + command.name + command.args` 连续出现几次可以判为 badcase？
2. 同一个工具但参数不同，是否应视为重复？
3. 如果 result 有新增信息，重复调用是否可以接受？

当前假设：

- 为了优先 precision，重复/循环阈值先使用 3。
- 判断重复时默认纳入 `command.args`。

### 1.4 用户确认

1. trace 中是否有字段能表示用户确认？
2. 用户确认通常出现在 `task_name`、`result` 还是其它字段？
3. 如果没有用户确认字段，是否可以仅根据工具失败后继续执行判 badcase？

当前假设：

- 先不识别用户确认字段。
- 工具失败后继续出现新的 command，作为高置信 badcase 候选。

## 2. 规则配置区

以下 JSON 配置会被代码读取。修改时请保持 JSON 合法。

<!-- RULE_CONFIG_START -->
```json
{
  "version": "rule-v0",
  "default_label_when_no_bad_rule_matches": "goodcase",
  "final_result_fields": [
    "final_result",
    "answer",
    "response"
  ],
  "success_task_keywords": [],
  "failure_markers": [
    "error",
    "failed",
    "exception",
    "traceback",
    "null",
    "none",
    "失败",
    "报错",
    "异常",
    "未找到",
    "无权限",
    "超时",
    "无法获取",
    "无结果",
    "没有查询到",
    "请求失败"
  ],
  "badcase_rules": {
    "plan_list_required": true,
    "empty_plan_without_final_result": true,
    "all_step_results_empty": true,
    "failed_step_continue_with_new_command": true,
    "repeated_or_loop_tasks": {
      "enabled": true,
      "repeat_threshold": 3,
      "include_command_args": true
    }
  }
}
```
<!-- RULE_CONFIG_END -->

## 3. 通用规则说明

### 3.1 高置信 badcase 规则

- `plan_list_required`：`plan_list` 缺失或不是 list。
- `empty_plan_without_final_result`：`plan_list` 为空，并且没有任何最终结果字段有内容。
- `all_step_results_empty`：有任务步骤，但所有步骤结果都为空。
- `failed_step_continue_with_new_command`：某一步结果命中失败词，后面仍继续调用新 command。
- `repeated_or_loop_tasks`：连续重复或循环任务达到阈值。

### 3.2 可选 goodcase 规则

- `success_task_keywords`：如果未来确认某些任务代表目标完成，可以填入关键词。
- 当前为空，因此规则法不会要求必须出现某个成功任务。

## 4. 实验后待更新

运行实验后，请把典型 FP/FN 的样本名和 reason 贴到 `方法及优化.md`。下一轮根据误判类型更新本文件：

- 增加或删除失败词。
- 调整重复阈值。
- 增加成功任务关键词。
- 增加允许继续执行的失败类型。
- 增加用户确认字段或关键词。

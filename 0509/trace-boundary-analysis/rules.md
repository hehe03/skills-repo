# 规则配置与问题收集

本文档是规则法的唯一规则来源。`scripts/classify_rule.py` 会读取下方“规则配置区”中的 JSON，并按规则层级逐步启用规则。

## 1. 背景

### 1.1 trace 推荐结构

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

### 1.2 生成 trace 的任务描述

用户输入“一句话任务描述”或者“策划报告文件”，Agent 需要收集和澄清要素，然后生成面向客户高层交流的交流大纲：

1. 收到用户输入后，首先检查要素完备性，跟用户澄清关键要素，并输出要素报告。
2. 在用户同意情况下帮助用户收集关键要素。
3. 根据要素报告，生成交流大纲。

## 2. 分层规则说明

规则按“可使用的先验知识”分为三层。实验时可以通过 `--rule-layer` 或 `--sweep-rule-layers` 控制启用范围；层级是累积的，即 `trace_format` 包含 `general`，`domain_prior` 包含前两层。

### 2.1 `general`: 不知道任何 trace 先验

这一层只把输入看成普通 JSON 结果，不假设 `plan_list`、`task_name`、`command`、`result` 等字段存在。

- `root_object_required`: JSON 根节点必须是对象。
- `non_empty_root_required`: JSON 根对象不能为空。
- `all_leaf_values_empty`: 所有叶子值都为空时判为 `badcase`。
- `failure_markers`: 通用失败词库，用于后续层判断失败结果；这些词本身不依赖 trace 格式。

### 2.2 `trace_format`: 知道 trace 基本格式

这一层使用 1.1 中的基础结构先验，重点检查 trace 的结构完整性、步骤结果和执行流。

- `plan_list_required`: `plan_list` 缺失或不是 list 时判为 `badcase`。
- `empty_plan_without_final_result`: `plan_list` 为空，且没有任何最终结果字段有内容时判为 `badcase`。
- `all_step_results_empty`: 有任务步骤，但所有步骤 `result` 都为空时判为 `badcase`。
- `failed_step_continue_with_new_command`: 某一步结果为空或包含失败词，后续仍调用新的 `command` 时判为 `badcase`。
- `repeated_or_loop_tasks`: 连续重复或循环任务达到阈值时判为 `badcase`。
- `final_result_fields`: 允许作为最终结果证据的顶层字段。

### 2.3 `domain_prior`: 知道额外领域先验

这一层使用当前任务领域的成功证据。它不覆盖前两层的高置信 badcase 判断，只在没有命中坏例规则时提供 goodcase 证据。

- `success_task_keywords`: 如果某个 `task_name` 命中成功任务关键词，且该步骤 `result` 有效，则判为 `goodcase`。
- 当前关键词：`生成大纲`。

## 3. 规则配置区

<!-- RULE_CONFIG_START -->
```json
{
  "version": "rule-v1-layered",
  "default_label_when_no_bad_rule_matches": "goodcase",
  "rule_layers": {
    "general": {
      "description": "不知晓 trace 结构，只使用通用 JSON 质量规则。",
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
        "root_object_required": true,
        "non_empty_root_required": true,
        "all_leaf_values_empty": true
      }
    },
    "trace_format": {
      "description": "知晓 trace 的 query/plan_list/task_name/command/result 基本格式。",
      "final_result_fields": [
        "final_result",
        "answer",
        "response"
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
    },
    "domain_prior": {
      "description": "知晓当前任务的额外成功先验。",
      "success_task_keywords": [
        "生成大纲"
      ]
    }
  }
}
```
<!-- RULE_CONFIG_END -->

## 4. 实验后待更新

运行实验后，请把典型 FP/FN 的样本名和 reason 贴到 `方法及优化.md`。下一轮根据误判类型更新本文档：

- 增加或删除失败词。
- 调整重复阈值。
- 增加成功任务关键词。
- 增加允许继续执行的失败类型。
- 增加用户确认字段或关键词。

# Trace 定界分析 Skill

这个目录是一个可直接交给其它 Agent 加载使用的自包含 skill，用于分析 Agent trace，并判断样本是 `goodcase` 还是 `badcase`。

## 文件结构

```text
trace-boundary-analysis/
├── SKILL.md
├── README.md
├── 方法及优化.md
└── scripts/
    ├── run_trace_analysis.py
    ├── common.py
    ├── features.py
    ├── classify_rule.py
    ├── classify_unsupervised.py
    └── classify_supervised.py
```

## 文件说明

- `SKILL.md`：Agent 加载 skill 时读取的中文说明，包含适用场景、输入输出规范、自动选路策略和使用命令。
- `README.md`：面向维护者的目录说明，只描述当前 skill 文件夹。
- `方法及优化.md`：无监督方法的实验记录和迭代方案，用于记录规则、特征、LLM 复核等优化实验。
- `scripts/run_trace_analysis.py`：统一入口，负责参数解析、读取 trace/metadata、自动选择方法、调用具体分类脚本和输出结果。
- `scripts/common.py`：公共数据结构、metadata 读取、trace 读取、指标计算、结果打印和 JSON 序列化。
- `scripts/features.py`：公共特征抽取逻辑，包括任务数量、任务多样性、工具多样性、结果非空比例、重复/循环模式、大纲生成信号等。
- `scripts/classify_rule.py`：规则法，适合单条或少量 trace 快速筛查。
- `scripts/classify_unsupervised.py`：无监督特征评分法，适合没有训练集但有一批 trace 的场景。
- `scripts/classify_supervised.py`：有监督特征法，适合 metadata 中已有可用 `train/test` 标注的场景。

## 输入规范

trace 输入可以是单个 JSON 文件，也可以是包含多个 JSON 文件的目录。推荐结构：

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

metadata CSV 是可选的，推荐列：

```text
name,label,source,split
```

- `name`：trace JSON 文件名
- `label`：`goodcase` 或 `badcase`
- `source`：样本来源
- `split`：`train` 或 `test`

## 使用方式

进入 `trace-boundary-analysis/` 后运行：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir>
```

带 metadata 自动选路：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --metadata <metadata.csv>
```

强制指定方法：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy rule
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy unsupervised
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --strategy supervised --metadata <metadata.csv>
```

输出 JSON：

```powershell
python .\scripts\run_trace_analysis.py <trace_json_or_dir> --metadata <metadata.csv> --output result.json
```

## 自动选路策略

- metadata 中存在可用 `train` 样本，且 `train` 同时包含 `goodcase` 和 `badcase`：使用 `supervised`。
- 没有可用训练集，但待分析样本数不少于 3：使用 `unsupervised`。
- 其它场景：使用 `rule`。

## 输出格式

终端默认输出：

```text
sample    predicted_label    strategy    reason
```

当 metadata 中包含真实标签时，会额外输出 badcase 混淆矩阵和 precision / recall / F1。这里以 `badcase` 作为正类。

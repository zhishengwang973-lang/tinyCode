# TinyCode 评测集

每个 YAML 用例会在隔离的临时工作区中运行真实 `AgentLoop`。执行模型负责完成任务；
评测模型只读取脱敏后的任务结果、工具序列、文件变更和指标，不会参与执行。

当前基线共 30 条用例：8 条直接回答、8 条工具路由/文件任务、6 条小修复、4 条长任务、4 条
安全与恢复任务。用 `--tag direct-answer`、`--tag tool-routing`、`--tag small-fix`、`--tag long-task`
或 `--tag safety-recovery` 可执行其中一个分组。

运行：

```bash
tinyCode eval evals/cases/fix_greeting.yaml --executor deepseek --judge gpt
# 或顺序执行整个目录；可用 --tag smoke 过滤
tinyCode eval evals/cases --executor deepseek --judge gpt --tag smoke
```

执行 Provider 和评测 Provider 必须配置为不同的 `model`。报告默认写入
`evals/results/`，包含总分、各条确定性断言、模型评分、Token、轮次、工具统计与 Trace 路径。
加 `--keep-workspace` 可保留该次临时工作区，便于复盘。

报告还包含任务实际产生的 `workspace_diff`：新增、修改、删除文件以统一 diff 记录，并注入独立
评测模型作为代码质量评分依据。每个文件和总 diff 都有长度上限；二进制、超大文件、`.tinyCode/`
运行时文件及敏感配置只保留“未提供 diff”的说明，不会泄露内容。

执行期间终端会持续显示总进度百分比、当前用例、执行轮次与当前阶段。该百分比是评测阶段的
进度，不把不确定的模型响应耗时伪装成剩余时间预测。

评分标准固定为 100 分：结果正确性 45 分、工具过程的确定性检查 15 分、效率 10 分，
以及独立评测模型给出的工具过程 5 分、指令遵循 15 分、代码质量 10 分。评测模型不可用时
仍会输出报告，但模型评分为 0，并明确记录异常。

用例中的 `assertions.tests` 会在隔离工作区执行。因此评测集应和代码一样受版本控制与审查，
不要直接运行来源不明的 YAML 用例。评测没有交互界面，写入和命令确认会在本次临时工作区内
自动批准；路径沙箱、敏感文件保护和危险命令黑名单仍然生效。

```yaml
name: 修复 greeting
tags: [coding, regression]
prompt: 修复 greeting.py 中的问候函数，并补充测试。
fixture: fixtures/greeting_bug
assertions:
  tests:
    - command: python3 -m unittest
      timeout_seconds: 30
  file_contains:
    - path: greeting.py
      text: "return f\"你好，{name}\""
  final_text_contains:
    - 测试
  tools_used: [read_file]
  tools_not_used: [delete_file]
  no_errors: true
budgets:
  max_rounds: 12
  max_model_requests: 18
  max_tokens: 80000
  max_duration_seconds: 300
```

# TinyCode — Tasks

## 任务总览（11 阶段，覆盖全部子系统）

```
 1. 项目脚手架
 2. 配置 + Provider + 对话  ← 并行
 3. 工具系统 + ReAct 循环
 4. TUI + 主流程接入
 5. 安全纵深防御
 6. MCP 协议客户端
 7. 上下文双层管理
 8. 命令系统 + 指令 + 笔记 + 会话
 9. Skill + Hook + 子 Agent
10. Worktree + Team 编排
11. 端到端验证
```

---

## 阶段 1：基础设施

### 任务 1.1 — 项目脚手架
**影响文件**：`pyproject.toml`, `tinyCode/__init__.py`, `tinyCode/__main__.py`
**依赖**：无
**关注点**：prompt-toolkit / httpx / pyyaml / pydantic 依赖声明；console_scripts 入口点

### 任务 1.2 — 配置系统
**影响文件**：`tinyCode/config/models.py`, `tinyCode/config/loader.py`
**依赖**：1.1
**关注点**：ProviderConfig / AppConfig 数据模型；三级配置发现；YAML 解析 + 校验

### 任务 1.3 — Provider 抽象 + Anthropic / OpenAI / DeepSeek
**影响文件**：`tinyCode/providers/base.py`, `anthropic.py`, `openai.py`, `deepseek.py`
**依赖**：1.2
**关注点**：BaseProvider 抽象类；SSE 流式解析；extended thinking / reasoning_content；ToolCall 数据类

### 任务 1.4 — 对话历史
**影响文件**：`tinyCode/conversation/history.py`
**依赖**：1.3
**关注点**：ConversationHistory（无 system prompt）；token 估算；无效消息过滤

---

## 阶段 2：核心循环

### 任务 2.1 — 工具系统
**影响文件**：`tinyCode/tools/base.py`, `registry.py`, `executor.py`, `read_file.py`, `write_file.py`, `edit_file.py`, `delete_file.py`, `run_command.py`, `glob.py`, `grep.py`, `tool_result_search.py`, `tool_result_read.py`
**依赖**：1.4
**关注点**：BaseTool → 9 个启动时基础工具；ToolCategory READ/WRITE 分组；超时 + 错误拦截

### 任务 2.2 — Agent 事件流
**影响文件**：`tinyCode/agent/events.py`
**依赖**：2.1
**关注点**：11 种事件类型 + AgentEvent 联合类型

### 任务 2.3 — ReAct 循环
**影响文件**：`tinyCode/agent/loop.py`
**依赖**：2.2
**关注点**：调 LLM → 工具调用 → 分批执行（读并发/写串行）→ 回填；plan-only 白名单；取消信号

---

## 阶段 3：TUI + 接入

### 任务 3.1 — 消息渲染
**影响文件**：`tinyCode/tui/render.py`
**依赖**：2.3
**关注点**：用户/AI/thinking/error/warning 格式；Style 样式表

### 任务 3.2 — TUI 主应用
**影响文件**：`tinyCode/tui/app.py`
**依赖**：3.1
**关注点**：Rich 单向输出；Prompt Toolkit 输入；Enter 分流；Tab 补全；瞬时进度；HITL 输入

### 任务 3.3 — 主流程接入
**影响文件**：`tinyCode/main.py`
**依赖**：3.2
**关注点**：单一 async event loop；41 个组件注入；--mode / --resume CLI

---

## 阶段 4：安全

### 任务 4.1 — 安全模型 + 黑名单 + 沙箱
**影响文件**：`tinyCode/security/models.py`, `blacklist.py`, `sandbox.py`
**依赖**：3.3
**关注点**：SecurityLevel 枚举；危险命令黑名单；PathSandbox

### 任务 4.2 — 安全策略 + Guard
**影响文件**：`tinyCode/security/policy.py`, `guard.py`
**依赖**：4.1
**关注点**：三档模式默认行为；会话/项目/全局规则优先级；HITL 管线

---

## 阶段 5：MCP 协议

### 任务 5.1 — JSON-RPC + 传输层
**影响文件**：`tinyCode/mcp/protocol.py`, `transport/base.py`, `stdio.py`, `http.py`
**依赖**：3.3
**关注点**：请求/响应/通知编解码；id→Future 异步匹配；SSE 解析

### 任务 5.2 — MCP 客户端 + 适配器 + 池 + 管理器
**影响文件**：`tinyCode/mcp/client.py`, `adapter.py`, `pool.py`, `manager.py`, `config.py`
**依赖**：5.1
**关注点**：initialize→initialized→list→call；并行 connect_all；延迟 resource/prompt 发现；命名前缀

---

## 阶段 6：上下文管理

### 任务 6.1 — 工具结果截断（层1）
**影响文件**：`tinyCode/conversation/truncator.py`
**依赖**：3.3
**关注点**：50K/条、200K/轮阈值；写盘留预览；返回截断信息

### 任务 6.2 — 结构化摘要 + 熔断（层2）
**影响文件**：`tinyCode/conversation/summarizer.py`
**依赖**：6.1
**关注点**：9 段摘要 Prompt；draft 草稿；边界消息；CircuitBreaker

### 任务 6.3 — 压缩协调器
**影响文件**：`tinyCode/conversation/compression.py`
**依赖**：6.2
**关注点**：两层统一入口；backward-compat CompressionResult

---

## 阶段 7：命令 + 会话 + 指令 + 笔记

### 任务 7.1 — 命令系统核心
**影响文件**：`tinyCode/commands/types.py`, `registry.py`, `parser.py`, `dispatcher.py`
**依赖**：3.3
**关注点**：CommandType / UIControl / CommandRegistry / 别名冲突检测 / Tab 补全列表

### 任务 7.2 — 内置命令（16 条）
**影响文件**：`tinyCode/commands/builtin/help_cmd.py` ~ `review_cmd.py`
**依赖**：7.1
**关注点**：/help /compress /clear /mode /session /memory /permission /status /config /exit /prompt /review /skill /tasks /worktree /team

### 任务 7.3 — JSONL 会话 + 迁移
**影响文件**：`tinyCode/storage/sessions.py`
**依赖**：7.1
**关注点**：追加写 O(1)；meta JSON；损坏行跳过；未配对 tool_use 截断；时间跨度提醒；旧格式迁移

### 任务 7.4 — 项目指令 + @include
**影响文件**：`tinyCode/instructions/loader.py`
**依赖**：3.3
**关注点**：TINYCODE.md 两级加载；@include 嵌套 3 层；路径越界拦截

### 任务 7.5 — 自动笔记
**影响文件**：`tinyCode/notes/manager.py`, `categories.py`
**依赖**：7.1
**关注点**：每 5 轮调 LLM；四分类；增量去重；Ctrl+C 退出时触发

---

## 阶段 8：Skill + Hook + 子 Agent

### 任务 8.1 — Skill 系统
**影响文件**：`tinyCode/skills/loader.py`, `registry.py`, `tool.py`, `executor.py`, `models.py`, `builtin/commit.md`, `review.md`, `test.md`
**依赖**：7.2
**关注点**：三级目录；YAML frontmatter + MD；两阶段加载；工具白名单交集；/skill 命令

### 任务 8.2 — Hook 引擎
**影响文件**：`tinyCode/hooks/models.py`, `conditions.py`, `templates.py`, `actions.py`, `loader.py`, `engine.py`
**依赖**：3.3
**关注点**：12 种事件；4 种操作符 ALL/ANY；4 种动作；集中校验；tool_pre_exec 拦截

### 任务 8.3 — 子 Agent 系统
**影响文件**：`tinyCode/subagent/models.py`, `runner.py`, `filter.py`, `manager.py`, `tool.py`, `roles/loader.py`, `builtin/explorer.md`, `planner.md`, `general.md`
**依赖**：8.1 (角色复用 RoleLoader)
**关注点**：Fork vs 定义模式；三层工具过滤；后台管理器；/tasks 命令

---

## 阶段 9：Worktree + Team

### 任务 9.1 — Git Worktree 管理
**影响文件**：`tinyCode/worktree/manager.py`, `validator.py`, `initializer.py`, `cleaner.py`, `models.py`
**依赖**：3.3
**关注点**：完整生命周期；名称安全校验；环境初始化；变更保护；后台清理；--resume 恢复

### 任务 9.2 — Agent Team 编排
**影响文件**：`tinyCode/teams/models.py`, `lead.py`, `member.py`, `tasks.py`, `mailbox.py`, `merger.py`, `scheduler.py`, `tools.py`, `persistence.py`
**依赖**：9.1
**关注点**：Lead 拆解/分配/合并；6 个协作工具；点对点消息；增量 merge + LLM 裁决；双锁调度模式

---

## 阶段 10：接入 + 补全

### 任务 10.1 — Skill/Hook/子Agent 接入 AgentLoop
**影响文件**：`tinyCode/agent/loop.py`, `tinyCode/main.py`
**依赖**：8.1, 8.2, 8.3
**关注点**：Skill 指令注入 _assemble_messages；工具白名单 _build_tool_defs；Hook fire() 5 个节点

### 任务 10.2 — Worktree/Team 接入主流程
**影响文件**：`tinyCode/main.py`, `tinyCode/tui/app.py`, `tinyCode/commands/__init__.py`
**依赖**：9.1, 9.2
**关注点**：BackgroundCleaner 启动/停止；worktree_manager → TUI → /worktree 命令；task_manager → /tasks 命令

---

## 阶段 11：验证

### 任务 11.1 — 端到端验证
**影响文件**：无（纯验证）
**依赖**：10.2
**关注点**：按 checklist.md 逐项验证；配置 → 启动 → 对话 → 工具调用 → 上下文压缩 → 退出恢复完整链路

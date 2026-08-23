# TinyCode — Spec

## 背景

从零构建一个终端 AI 编程助手（类似 Claude Code），使用 Python 开发。支持多 Provider、工具调用、安全策略、MCP 协议、Skill 系统、Hook 引擎、子 Agent 协作等完整能力。

## 目标用户

在终端中工作的开发者，习惯 CLI 交互，需要 AI 辅助编码、文件操作、代码审查、任务协作。

## 能力清单

### 基础对话
1. **TUI 交互界面**：Rich 负责单向流式输出，Prompt Toolkit 负责输入与补全；终端原生滚动，临时进度不进入回答正文
2. **单行输入**：Enter 提交，Tab 补全命令
3. **流式逐字打印**：大模型回复通过 SSE 流式返回，逐 token 渲染
4. **多轮对话**：AI 能记住本轮会话中之前的所有消息

### Provider 层
5. **多 Provider 后端**：支持 Anthropic Claude / OpenAI / DeepSeek，通过 YAML 配置切换
6. **Provider 抽象层**：统一 `BaseProvider` 接口 + `ToolCall` 数据类 + `make_tool_calls_message` / `make_tool_result_message`
7. **Extended Thinking / Reasoning**：Claude extended thinking + DeepSeek reasoning_content，TUI 仅用临时进度提示推理阶段
8. **配置管理**：YAML 多 Provider 列表，protocol / model / base_url / api_key 字段
9. **配置发现**：环境变量 `TINYCODE_CONFIG` → `.tinyCode.yaml` → `~/.tinyCode/config.yaml`

### 工具系统
10. **内置工具体系**：核心工具 read_file / write_file / edit_file / delete_file / run_command / glob / grep，运行时再接入 skill_loader / sub_agent / MCP / Team 工具
11. **统一 Tool 接口**：name / description / category（READ/WRITE） / parameters / execute
12. **工具注册中心**：按名查找、转 OpenAI/Anthropic 格式
13. **工具执行器**：超时控制、错误捕获、结构化结果回灌
14. **edit_file 语义**：原文精确匹配替换（`str.count` 判断唯一性，0/多次均报错含上下文）

### Agent 循环
15. **ReAct 范式**：调 LLM → 解析工具调用 → 分批执行（读并发/写串行）→ 回填 → 继续
16. **事件流**：UserMessage / ThinkingEvent / TextDeltaEvent / ToolCallEvent / ToolResultEvent / ToolBlockedEvent / AgentDoneEvent / ErrorEvent / RoundStartEvent / RoundLimitReachedEvent / RoundLimitExtendedEvent / SteeringAppliedEvent / TruncationEvent / HITLRequestEvent
17. **Steering、取消与超时**：任务运行时继续接收用户输入，在完整模型响应或工具结果批次结束后的协议安全边界注入；`/cancel` 打断当前任务并清理尚未注入的追加指令
18. **弹性轮次预算**：`max_rounds` 是初始软预算；耗尽后按 `round_limit_action` 询问、自动续跑或暂停，续跑步长由 `round_extension` 控制，且始终受 `hard_max_rounds`（1–100）约束；会话内可通过 `/config` 调整

### 安全系统
19. **硬安全边界**：危险命令黑名单、Provider 凭据文件保护和命令环境凭据清理，始终生效不依赖档位
20. **路径沙箱**：拒绝绝对路径、`..` 遍历、项目目录边界检查
21. **三档权限模式**：严格（白名单内读取放行、其余询问）/ 默认（文件读取放行、Shell 与写入询问）/ 放行（普通操作放行，硬安全边界与沙箱仍生效）
22. **可配置规则**：会话级 > 项目级 > 全局级，工具 + 路径 + 命令细粒度匹配
23. **人在回路（HITL）**：A/S/P/D 四键决策，本次/会话/永久允许或拒绝
24. **失败信息回灌**：拒绝原因作为 ToolResult error 字段返回模型，Agent 据此调整

### 会话管理
25. **JSONL 追加存储**：O(1) 写、崩溃只丢最后一行、Meta 文件存概要
26. **会话恢复**：损坏行跳过、未配对 tool_use 截断、时间跨度 > 30 分钟提醒
27. **旧格式迁移**：`default.json` → JSONL 一次性迁移

### 上下文管理
28. **层1 截断**：单条工具结果 > 50K 字符写盘留 2K 预览；单轮合计 > 200K 截断最大的
29. **层2 摘要**：token ≥ 70% 窗口时警告并触发 9 段结构化 LLM 摘要；TUI 在 70%/90% 分别用黄色/红色提示占用
30. **摘要 Prompt**：禁止工具调用（首尾强调）+ 先出 draft 草稿再出正式摘要
31. **边界消息**：提示模型重新读取文件不要脑补
32. **熔断**：摘要连续失败 2 次停止自动触发

### MCP 协议
33. **JSON-RPC 2.0 客户端**：initialize → initialized → tools/list → tools/call
34. **两种传输**：Stdio 子进程 + Streamable HTTP（POST /mcp + SSE）
35. **并行连接**：启动时 `asyncio.gather` 连接所有 server，失败不阻塞
36. **三类能力**：Tools（MCPToolAdapter）、Resources（MCPResourceAdapter，延迟发现）、Prompts（MCPPromptAdapter，延迟发现）
37. **命名前缀**：清洗为 Provider 兼容的 `mcp_<server>_tool_<name>`，超长名称带稳定摘要并限制为 64 字符

### Skill 系统
38. **YAML frontmatter + Markdown 正文**：name / description / mode / tools / model / history_carry
39. **三级存储**：项目 (`.tinyCode/skills/`) > 用户 (`~/.tinyCode/skills/`) > 内置
40. **两阶段加载**：Phase 1 注入名字+描述 → Phase 2 激活完整 SOP
41. **工具白名单**：Skill 声明 tools_allow / tools_deny，多个 Skill 取交集
42. **系统工具豁免**：`skill_loader` 始终可用，不受白名单约束
43. **自动注册命令**：每个 Skill → `/<name>` 斜杠命令
44. **内置 3 个参考 Skill**：commit（Conventional Commits）/ review（代码审查）/ test（测试生成）

### Hook 系统
45. **事件 + 条件 + 动作**三要素规则：条件可省略，事件和动作必须有
46. **12 种生命周期事件**：会话/轮次/消息/工具/系统
47. **4 种操作符**：exact / not / regex / glob，ALL 或 ANY 二选一逻辑
48. **4 种动作**：shell / prompt_inject / http / sub_agent
49. **拦截能力**：tool_pre_exec 可返回拒绝原因 → LLM 收到调整
50. **执行控制**：once / async / timeout；拦截事件强制同步
51. **错误隔离**：Hook 失败只记日志，不中断 Agent
52. **集中校验**：YAML 加载时检查事件名、必填字段、async 约束

### 子 Agent
53. **统一入口**：`sub_agent(task, role?, background?)` 单工具
54. **两种模式**：定义式（空白对话 + 角色 SOP）+ Fork 式（继承父历史 + 注入强硬指令）
55. **Fork 指令**：不创建子 worker / 不对话 / 直接干活 / 结构化报告 ≤ 500 字
56. **工具过滤三层防线**：全局禁止 sub_agent → 角色 allow/deny → 后台只读白名单
57. **后台任务**：Fork 强制后台，完成自动注入 tool 消息
58. **管理器**：/tasks 命令查看、终止、详情
59. **内置 3 个角色**：explorer / planner / general

### Agent Team
60. **Team 定义**：JSON 格式存用户目录，Lead + 多个 Member
61. **协作工具 6 个**：team_create_task / list / view / update / send_message / broadcast，仅 Team 成员可见
62. **运行入口**：/team run <名称> <目标> → run_team → LeadAgent.execute
63. **点对点消息**：名称注册表 + 邮箱文件（JSONL）
64. **增量合并**：成员完成 → git merge worktree → LLM 裁决冲突 → 失败回滚
65. **纯调度模式**：双锁剥夺读写/命令工具 → 注入 10 阶段工作流指引
66. **成员恢复**：idle 后通过消息恢复上下文继续工作，不重 spawn

### 工作目录隔离
67. **Git worktree**：统一放 `.tinyCode/worktrees/`，不被 Git 追踪
68. **名称校验**：`[a-zA-Z0-9_-]`、段 ≤ 64 字符、总长 ≤ 255、拒绝 `.` `..`
69. **环境初始化**：复制 config、symlink 依赖、git hooks
70. **变更保护**：有未提交修改默认拒绝删除，需 --force
71. **后台清理**：每 5 分钟扫描过期无修改 worktree
72. **--resume**：进程退出后恢复上次 Worktree（不自动恢复聊天会话）

### 命令系统
73. **统一注册中心**：名称/别名/描述/用法/类型/参数提示/处理函数，别名冲突检测
74. **输入分流**：`/` 前缀 → 命令分发，否则 → AI
75. **Tab 补全**：单匹配补全、多匹配列表、隐藏命令不参与
76. **UIControl 接口**：命令通过稳定接口控制 TUI，不依赖具体渲染实现
77. **16 个内置命令模块**：help / compress / clear / mode / session / memory / permission / status / config / exit / prompt / review / skill / tasks / worktree / team；Skill 还能自动注册 `/<name>` 专属命令

### 项目指令与笔记
78. **指令文件**：`TINYCODE.md`（项目级）+ `~/.tinyCode/instructions.md`（用户级），`@include` 嵌套 3 层
79. **自动笔记**：每 5 轮调 LLM，四分类串行增量更新，LLM 去重；安全退出前做限时刷新

## 非功能要求

- 全程异步 I/O（asyncio + httpx），不阻塞 UI
- Provider 初始化失败启动即报错
- 配置解析失败可读错误 + 定位到具体规则
- 压缩后日志记录压缩事件
- `/clear` 顺带清空激活 Skill

## 设计骨架

```
tinyCode/
├── main.py                      # 启动编排（41 个组件注入）
├── config/                      # YAML 配置
│   ├── loader.py                # 三级发现 + 校验
│   └── models.py                # ProviderConfig / AppConfig
├── providers/                   # LLM 后端
│   ├── base.py                  # BaseProvider + ToolCall + Message
│   ├── anthropic.py             # SSE + cache_control + thinking
│   ├── openai.py                # SSE + tool_calls
│   └── deepseek.py              # OpenAI 兼容 + reasoning
├── agent/                       # ReAct 循环
│   ├── loop.py                  # AgentLoop + Prompt + Hook + Skill
│   └── events.py                # 12 种事件类型
├── conversation/                # 对话管理
│   ├── history.py               # ConversationHistory（无 system prompt）
│   ├── truncator.py             # 层1 工具结果截断
│   ├── summarizer.py            # 层2 结构化摘要 + 熔断 + 边界消息
│   └── compression.py           # 两层协调器
├── prompts/                     # Prompt 系统
│   ├── builder.py               # PromptBuilder + cache_control
│   ├── injector.py              # 每轮动态注入
│   ├── environment.py           # 环境信息收集
│   └── modules/                 # 7 个 .txt 优先级模块
├── tools/                       # 工具系统
│   ├── base.py                  # BaseTool + ToolResult + ToolCategory
│   ├── registry.py              # ToolRegistry + 格式转换
│   ├── executor.py              # ToolExecutor + 超时
│   └── [7 个核心工具].py
├── tui/                         # 终端 UI
│   ├── app.py                   # TinyCodeTUI(UIControl) + 命令分流 + Rich 流式输出
│   └── render.py                # 消息样式渲染
├── storage/sessions.py          # JSONL 持久化 + 恢复 + 迁移
├── security/                    # 安全防御
│   ├── blacklist.py             # 硬编码危险命令
│   ├── sandbox.py               # PathSandbox
│   ├── policy.py                # SecurityPolicy + 规则评估
│   └── guard.py                 # SecurityGuard 管线
├── mcp/                         # MCP 客户端
│   ├── protocol.py              # JSON-RPC 2.0
│   ├── transport/stdio.py       # 子进程传输
│   ├── transport/http.py        # HTTP SSE 传输
│   ├── client.py                # MCPClient 生命周期
│   ├── adapter.py               # Tool/Resource/Prompt 适配器
│   ├── pool.py                  # 并行连接池
│   └── manager.py               # MCPManager
├── commands/                    # 命令系统
│   ├── types.py                 # CommandType + UIControl
│   ├── registry.py              # CommandRegistry
│   ├── parser.py                # /name args 解析
│   ├── dispatcher.py            # 分流器
│   └── builtin/                 # 16 个内置命令模块
├── skills/                      # Skill 系统
│   ├── loader.py                # 三级扫描 + YAML 解析
│   ├── registry.py              # 两阶段加载
│   ├── tool.py                  # skill_loader 工具
│   ├── executor.py              # isolated 执行
│   └── builtin/                 # commit / review / test
├── hooks/                       # Hook 引擎
│   ├── models.py                # Rule / Condition / Action
│   ├── conditions.py            # 条件求值器
│   ├── templates.py             # {{var}} 替换
│   ├── actions.py               # 4 种动作执行器
│   ├── loader.py                # YAML 加载 + 集中校验
│   └── engine.py                # HookEngine
├── subagent/                    # 子 Agent
│   ├── runner.py                # SubAgentRunner
│   ├── filter.py                # 三层工具过滤
│   ├── manager.py               # BackgroundTaskManager
│   ├── tool.py                  # sub_agent 工具
│   └── roles/builtin/           # explorer / planner / general
├── worktree/                    # Git 工作目录
│   ├── manager.py               # 完整生命周期 + 会话持久化
│   ├── validator.py             # 名称安全校验
│   ├── initializer.py           # 环境初始化
│   └── cleaner.py               # 后台清理
├── teams/                       # Agent Team
│   ├── lead.py                  # LeadAgent 编排
│   ├── orchestrator.py          # /team run 到 LeadAgent 的组装服务
│   ├── member.py                # TeamMember 协程执行
│   ├── tasks.py                 # SharedTaskList + 依赖
│   ├── mailbox.py               # Mailbox 消息通信
│   ├── merger.py                # GitMerger + LLM 裁决
│   ├── scheduler.py             # DispatchScheduler 双锁
│   ├── tools.py                 # 6 个团队协作工具
│   └── persistence.py           # Team 定义加载
├── instructions/loader.py       # @include 指令加载
└── notes/                       # 自动笔记
    ├── manager.py               # AutoNoteManager
    └── categories.py            # 四分类 + Prompt 模板
```

## Out of Scope（本版本不做）

- 非 YAML 配置格式
- pip 包发布
- Skill 市场与分发 / 版本管理
- Agent 递归调用（子 Agent 调用子 Agent）
- 终端窗格后端（BackendTerminal 占位）
- 多行输入
- Syntax highlighting

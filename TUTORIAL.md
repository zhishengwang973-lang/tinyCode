# TinyCode 使用教程

## 目录

1. [快速开始](#1-快速开始)
2. [基本对话](#2-基本对话)
3. [Provider 配置](#3-provider-配置)
4. [工具系统](#4-工具系统)
5. [命令系统](#5-命令系统)
6. [对话管理](#6-对话管理)
7. [安全系统](#7-安全系统)
8. [MCP 协议](#8-mcp-协议)
9. [Skill 技能](#9-skill-技能)
10. [Hook 钩子](#10-hook-钩子)
11. [子 Agent](#11-子-agent)
12. [工作目录隔离](#12-工作目录隔离)
13. [Agent Team](#13-agent-team)
14. [项目指令与笔记](#14-项目指令与笔记)
15. [完整快捷键](#15-完整快捷键)

---

## 1. 快速开始

### 安装

```bash
python3 -m pip install -e /path/to/mewcode-main
```

**要求**：Python ≥ 3.10，Git ≥ 2.30

这是可编辑安装。源码修改会立即反映到 `tinyCode` 命令，通常不需要再次执行安装。

### 最小配置

在项目根目录创建 `.tinyCode.yaml`：

```yaml
providers:
  - name: claude
    protocol: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    context_window: 200000

active_provider: claude
```

### 启动

```bash
python -m tinyCode

# 指定配置文件
TINYCODE_CONFIG=/path/to/config.yaml python -m tinyCode

# 安全等级
python -m tinyCode --mode strict

# 恢复上次进入的 Git Worktree
python -m tinyCode --resume

# 明确信任当前项目的 MCP/Hook 可执行配置
python -m tinyCode --trust-project-config
```

安装后可以在任意目录直接运行 `tinyCode`。建议把通用 Provider 配置放在
`~/.tinyCode/config.yaml`；项目目录中的 `.tinyCode.yaml` 只覆盖普通配置项。
如果项目配置包含 `providers`，整份 Provider 列表会替换全局列表，不会继承全局密钥。
`TINYCODE_CONFIG` 指向的文件是独立配置，不与全局或项目配置合并。

---

## 2. 基本对话

启动后进入保留终端滚动历史的流式 TUI：

```
TinyCode · claude/claude-sonnet-4-6 · MCP 1 · 就绪
›
```

### 操作

| 操作 | 方式 |
|------|------|
| 提交消息 | 输入内容 → **Enter** |
| 安全退出 | 输入 `/exit`（别名 `/quit`、`/q`） |
| 中断/退出 | **Ctrl+C** 或 **Ctrl+D** |
| Tab 补全命令 | 输入 `/hel` → **Tab** → `/help` |
| 运行进度 | 等待模型或执行工具时显示临时进度；开始输出回答后自动收起 |

AI 会连续流式输出回复。模型调用工具时，工具名只显示在临时进度中，不会插进回答正文：

```
You: 读一下 README.md

TinyCode: README.md 的内容显示这是一个 AI 编程助手项目...
✓ 本轮已正常完成
本轮统计 · Turn: 2 · 模型请求: 2 次 · 消耗 Token: 1,234 · 耗时: 2.31 秒 · 工具调用: 1 次 · 成功率: 100.0%
上下文   · ≈18.6k / 128.0k（15%）· 剩余 ≈109.4k
```

实际统计行还会显示 `Turn` 和“模型请求”次数。Turn 是本轮经历的 Agent/ReAct
轮数；模型请求包含重试和任务内上下文压缩，因此可能大于 Turn。Token 数来自
Provider 返回的实际 usage，并累计本任务内所有请求；兼容服务未返回 usage 时显示“不可用”。
上下文占用是下一轮可用历史的估算快照：低于 70% 使用弱化颜色，70% 起显示黄色，90% 起显示红色。

---

## 3. Provider 配置

支持三种协议，可在同一配置文件中定义多个并按需切换。

```yaml
providers:
  # Anthropic Claude
  - name: claude
    protocol: anthropic
    model: claude-sonnet-4-6
    base_url: https://api.anthropic.com      # 可选，不填用默认
    api_key_env: ANTHROPIC_API_KEY
    context_window: 200000

  # OpenAI
  - name: gpt
    protocol: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}

  # DeepSeek
  - name: deepseek
    protocol: deepseek
    model: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY

  # DeepSeek 推理模型
  - name: deepseek-r1
    protocol: deepseek
    model: deepseek-reasoner
    api_key_env: DEEPSEEK_API_KEY

  # 本地模型（Ollama / vLLM / LiteLLM）
  - name: local
    protocol: openai
    model: llama-3
    base_url: http://localhost:11434/v1
    api_key: ollama

active_provider: claude   # ← 当前使用哪个
max_rounds: 30            # 单个任务最大 Agent 轮数，范围 1-100
security_level: normal    # strict / normal / permissive
notes_enabled: true       # 持久笔记开关，仅在全局配置中生效
```

密钥支持三种互斥写法：`api_key_env: ENV_NAME`、`api_key: ${ENV_NAME}`，或安装
`tinyCode[keyring]` 后使用 `api_key_keyring: service/username`。不建议把真实密钥写入
项目文件。未显式配置密钥时，三个协议会分别尝试 `ANTHROPIC_API_KEY`、
`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`。

启动时只要求 `active_provider` 的密钥实际可用；其他 Provider 的字段结构仍会校验，
切换为 active 后才要求相应环境变量或 Keyring 条目存在。因此可以保留多家 Provider
模板，而不必一次配置所有密钥。

**配置规则**：`TINYCODE_CONFIG` 最高优先且单独加载；否则先加载
`~/.tinyCode/config.yaml`，再用当前目录 `.tinyCode.yaml` 覆盖。项目 Provider 列表
整体替换全局 Provider 列表，避免项目提供的 `base_url` 偷用全局密钥。项目配置可以
提高全局安全等级，但不能把显式的全局 `security_level` 降级；需要临时降级时必须由用户
亲自传入 `--mode`。`notes_enabled` 只接受 `true` 或 `false`，并且只从全局
`~/.tinyCode/config.yaml` 读取，项目 `.tinyCode.yaml` 不能启用或关闭用户的持久笔记。

**Claude Extended Thinking**：在代码中通过 `provider.enable_thinking(budget_tokens=8192)` 开启。推理期间 TUI 使用临时进度提示，不把推理状态插入回答正文。

**DeepSeek Reasoner**：`deepseek-reasoner` 模型的推理过程自动以 `[Reasoning]` 标签渲染。

---

## 4. 工具系统

TinyCode 启动时先注册 9 个基础工具，之后再加入 Skill、子 Agent 和 MCP 工具。
明确独立的算法、解释或代码片段请求不会暴露工作区工具，并且只向模型发送最新请求，
避免上一轮算法答案污染新请求；项目任务和“继续”类请求仍保留完整会话上下文。

| 工具 | 功能 | 类型 |
|------|------|------|
| `read_file` | 读取文件内容（UTF-8/GBK/Latin-1 自动尝试） | 读 |
| `write_file` | 写入新文件（已存在则报错） | 写 |
| `edit_file` | 精确匹配替换（原文必须唯一出现） | 写 |
| `delete_file` | 删除工作目录内的单个文件 | 写 |
| `run_command` | 执行 Shell 命令（工作目录内） | 写 |
| `glob` | 按模式查找文件（如 `**/*.py`） | 读 |
| `grep` | 正则搜索代码内容 | 读 |
| `tool_result_search` | 搜索已落盘的大型工具结果 | 读 |
| `tool_result_read` | 分段读取已落盘的大型工具结果 | 读 |

### 工具调用示例

```
> 帮我在 src/ 下找到所有定义了 main 函数的文件

🔧 grep(pattern='def main')
  → src/cli.py:42: def main():
  → src/server.py:15: def main():

TinyCode: 找到了两个文件：src/cli.py:42 和 src/server.py:15
```

### 工具截断

单个工具结果超过 **50K 字符**时，完整内容写入 `~/.tinyCode/tool_results/`，对话中只保留 2K 字符预览：

```
📎 read_file 结果过大（125,000 字符）→ 存盘 ~/.tinyCode/tool_results/20260601_read_file.txt
```

---

## 5. 命令系统

输入 `/` 开头触发命令，Tab 可补全。未知命令自动引导到 `/help`。

| 命令 | 别名 | 用途 |
|------|------|------|
| `/help [命令]` | `h ?` | 列出命令或查看详情 |
| `/clear` | `cls reset` | 清空当前对话 |
| `/compress` | `zip` | 手动触发上下文压缩 |
| `/mode [plan\|security]` | — | 切换模式 |
| `/status` | `st info` | 显示综合状态 |
| `/config max-rounds [N]` | `cfg` | 查看或修改当前会话最大轮次 |
| `/prompt [部分]` | `system-prompt sp` | 查看当前实际生效的系统提示词 |
| `/exit` | `quit q` | 保存会话并安全退出 |
| `/session [list\|load\|new\|delete]` | `sess` | 管理会话 |
| `/memory [show\|clear\|edit]` | `mem notes` | 管理自动笔记 |
| `/permission` | `perm acl` | 安全权限状态 |
| `/review [路径]` | `cr audit` | 请求代码审查 |
| `/skill [list\|reload\|clear]` | `skills` | Skill 管理 |
| `/tasks [list\|detail\|kill]` | `bg` | 后台任务管理 |
| `/worktree [status\|list\|create\|enter\|exit]` | `wt` | Git 工作目录 |
| `/team [list\|show\|dir\|run]` | `tm` | Team 管理和执行 |

`/prompt` 只读取当前系统上下文，不会请求模型。可查看 `all`、`base`、
`instructions`、`skills`、`environment` 或 `injection`；可连续执行，不会消耗对话轮次。
`/config max-rounds 50` 只影响当前启动会话，重启后恢复 YAML 中的 `max_rounds`。

---

## 6. 对话管理

### 上下文压缩

估算上下文达到模型窗口 **70%** 时发出警告并尝试自动压缩。任务结束后的上下文
快照在 70% 起显示黄色、90% 起显示红色。

压缩产出一个 **9 段结构化摘要**：主要请求、关键概念、文件与代码、错误与修复、解决过程、用户原话、待办事项、当前工作、下一步。

压缩后附加边界消息提示模型重新读取文件而非脑补细节。

**手动触发**：`/compress` 或 **Ctrl+Q**

**熔断保护**：连续 2 次摘要失败自动停止自动压缩。

### 会话持久化

每轮对话后自动保存到 `~/.tinyCode/sessions/{id}.jsonl`（追加写，O(1)）。每次
启动默认创建新会话；使用 `/session list` 和 `/session load <ID>` 恢复对话内容。

恢复时自动处理：
- 损坏行跳过
- 未配对的 tool_use 截断
- 时间跨度 > 30 分钟注入提醒

```bash
# 在应用内恢复一段对话
/session list
/session load <ID>
```

启动参数 `--resume` 恢复的是上次进入的 Git Worktree，不等同于加载聊天会话。

---

## 7. 安全系统

三层权限档位：**Ctrl+S** 循环切换，或 `/mode security <level>`。

| 档位 | 读类工具 | 写类工具 | 路径限制 |
|------|---------|---------|---------|
| **严格** | 仅白名单路径 | 全部询问 | `.tinyCode-security.yaml` 声明的路径 |
| **默认** | 直接放行 | 写入时询问 | 项目目录内放行 |
| **放行** | 直接放行 | 直接放行 | 仍限制在项目目录并拦截敏感配置、黑名单命令 |

### 人在回路（HITL）

当安全策略无法自动判断时，弹出确认提示：

```
⚠ 安全确认: run_command(command='pip install pandas')
  当前模式: normal
  [A]llow once  [S]ession allow  [P]ermanent allow  [D]eny
```

- **A**：本次允许
- **S**：会话允许（本次启动内有效）
- **P**：永久允许（写入 `.tinyCode-security.yaml`）
- **D**：拒绝

### 安全规则文件

```yaml
# .tinyCode-security.yaml
rules:
  - tool: run_command
    command_pattern: "pip install *"
    action: allow

  - tool: write_file
    path_pattern: "*.env"
    action: deny
```

优先级：会话级 > 项目级 > 全局级

默认安全等级可写在全局 `~/.tinyCode/config.yaml` 的
`security_level: strict|normal|permissive` 中，项目 `.tinyCode.yaml` 和启动参数
`--mode` 可依次覆盖它。全局安全规则位于 `~/.tinyCode/security.yaml`，项目规则位于
`.tinyCode-security.yaml`。选择 P 永久允许时写入当前项目规则文件。

---

## 8. MCP 协议

连接外部 MCP Server，扩展工具集。

### 配置

```yaml
# .tinyCode-mcp.yaml（项目级，覆盖全局 ~/.tinyCode/mcp.yaml）
servers:
  # Stdio 传输：本地子进程
  - name: filesystem
    transport: stdio
    command: npx
    args: [-y, "@anthropic/mcp-filesystem", /allowed/path]
    timeout: 30

  # HTTP 传输：远程服务
  - name: remote
    transport: http
    url: http://localhost:8080
    headers:
      Authorization: Bearer xxx
    timeout: 30
```

### 工具命名

MCP 名称会清洗成各 Provider 都接受的最长 64 字符标识符：

```
mcp_filesystem_tool_read_file
mcp_filesystem_tool_write_file
mcp_remote_tool_search
mcp_filesystem_resource
mcp_filesystem_prompt
```

启动时并行连接所有 Server，失败的不阻塞启动，掉线后调用会解析重连后的客户端。
项目级 MCP/Hook 配置能够启动进程或发起网络请求，因此默认忽略；确认信任仓库后使用
`tinyCode --trust-project-config`。全局 `~/.tinyCode/mcp.yaml` 不受这个开关影响。

---

## 9. Skill 技能

Skill 是用 YAML frontmatter + Markdown 正文定义的专业 SOP。

### 创建 Skill

```markdown
---
name: my-skill
description: 我的自定义技能
mode: shared
tools: [read_file, glob, grep]
---

# My Skill SOP

1. 使用 glob 了解项目结构
2. 使用 grep 搜索关键模式
3. 输出分析报告
```

### 存放位置

| 优先级 | 路径 |
|--------|------|
| 项目级 | `.tinyCode/skills/*.md` |
| 用户级 | `~/.tinyCode/skills/*.md` |
| 内置 | `tinyCode/skills/builtin/*.md` |

同名 Skill 按优先级覆盖。

### 内置 Skill

| Skill | 模式 | 描述 |
|-------|------|------|
| `commit` | shared | 生成 Conventional Commits 提交信息 |
| `review` | isolated | 全面代码审查（正确性/安全/性能/风格） |
| `test` | shared | 分析变更并生成/运行测试 |

### 使用 Skill

Agent 调用 `skill_loader(name="commit")` 激活 Skill，或使用命令：

```
/skill list              # 列出可用 Skill
/skill commit            # 查看 Skill 详情
/commit                  # 自动注册的命令，激活并执行
```

激活后 Skill 指令**钉在环境上下文**中，每轮 LLM 调用都可见。

---

## 10. Hook 钩子

事件驱动的自动化规则。

### 配置

```yaml
# .tinyCode-hooks.yaml
hooks:
  - name: block-rm-rf
    event: tool_pre_exec
    condition:
      match: ALL
      rules:
        - field: tool_name
          operator: exact
          value: run_command
        - field: params.command
          operator: regex
          value: "rm\\s+-rf"
    actions:
      - type: prompt_inject
        text: "拦截: '{{params.command}}' 包含危险操作"
    control:
      async: false

  - name: log-writes
    event: tool_post_exec
    condition:
      match: ANY
      rules:
        - field: tool_name
          operator: exact
          value: write_file
    actions:
      - type: shell
        command: "echo {{tool_name}} {{params.path}} >> ~/.tinyCode/log.txt"
    control:
      async: true
      once: false
```

### 12 种事件

| 层级 | 事件 | 可拦截 |
|------|------|--------|
| 会话 | `session_start` `session_end` | — |
| 轮次 | `round_start` `round_end` | — |
| 消息 | `message_pre_send` `message_post_receive` | — |
| 工具 | `tool_pre_exec` | **✓** |
| 工具 | `tool_post_exec` | — |
| 系统 | `system_startup` `system_shutdown` `system_error` `system_compress` | — |

### 四种动作

| 动作类型 | 用途 |
|---------|------|
| `shell` | 执行命令 |
| `prompt_inject` | 向 LLM 注入文本（拦截事件用此反馈拒绝原因） |
| `http` | 发起 HTTP 请求 |
| `sub_agent` | 启动真实后台子 Agent，并将结果注入后续上下文 |

---

## 11. 子 Agent

### 定义模式（指定角色）

```
> 用 explorer 角色探索项目结构

🔧 sub_agent(task='探索项目结构', role='explorer')
  → ## 项目结构 / ## 关键模块 / ## 代码模式...
```

### Fork 模式（继承当前对话）

不指定 `role` 参数，自动继承当前对话历史 + 复用工具集：

```
> 帮我分析刚才读到的所有文件

🔧 sub_agent(task='分析刚才读到的文件')
  → [Fork 模式] 分析结果...
```

Fork 模式**强制后台运行**，完成结果自动注入对话。

### 内置角色

| 角色 | 工具 | 用途 |
|------|------|------|
| `explorer` | read_file, glob, grep | 探索代码结构 |
| `planner` | + run_command | 制定执行计划 |
| `general` | 全部 | 通用综合任务 |

### 自定义角色

在 `.tinyCode/roles/` 下创建 Markdown 文件（同名覆盖内置）：

```markdown
---
name: my-role
description: 自定义角色
tools_allow: [read_file, glob, grep, run_command]
max_rounds: 5
---

# My Role SOP
...
```

### 后台任务管理

```
/tasks list       # 列出所有后台任务
/tasks detail id  # 查看详情
/tasks kill id    # 终止任务
```

---

## 12. 工作目录隔离

基于 `git worktree` 的物理隔离，每个子 Agent 可在独立工作目录中操作。

### 命令

```
/worktree status                  # 当前状态
/worktree list                    # 列出所有工作目录
/worktree create fix-bug          # 创建 worktree（自动复制配置+链接依赖）
/worktree enter fix-bug           # 切换到工作目录
/worktree exit fix-bug            # 退出（有修改默认拒绝删除）
/worktree exit fix-bug --force    # 强制退出
```

### 变更保护

退出 worktree 时，默认检查 `git status`：

- **有未提交修改** → 拒绝删除，提示用 `--force`
- **无修改** → 正常删除 worktree + 分支

### 后台清理

每 5 分钟自动清理过期（24h+）且无修改的 worktree。

在普通非 Git 新目录中 TinyCode 仍可正常对话和操作文件，只会禁用 Worktree 后台清理及
相关命令并给出原因；需要这些能力时先执行 `git init`。

### 恢复

```bash
python -m tinyCode --resume
# → 恢复到上次的 worktree 会话
```

---

## 13. Agent Team

长期存在的协作小组，Leader 拆解目标、分配成员、合并结果。

### Team 定义

```json
// ~/.tinyCode/teams/example.json
{
  "name": "example",
  "description": "示例 Team",
  "lead_role": "general",
  "members": [
    {"name": "alice", "role": "explorer", "worktree": "alice-wt", "backend": "coro"},
    {"name": "bob", "role": "planner", "worktree": "bob-wt", "backend": "coro"},
    {"name": "carol", "role": "general", "worktree": "carol-wt", "backend": "coro"}
  ],
  "dispatch_mode": false
}
```

### 协作工具

Team 成员拥有专属的 6 个协作工具（主 Agent 不可见）：

| 工具 | 用途 |
|------|------|
| `team_create_task` | 创建任务（可指定依赖） |
| `team_list_tasks` | 列出所有任务及状态 |
| `team_view_task` | 查看单任务详情 |
| `team_update_task` | 更新任务状态/结果 |
| `team_send_message` | 点对点消息 |
| `team_broadcast` | 广播到全组 |

### 纯调度模式

在 Team JSON 中设置 `"dispatch_mode": true` 后，Leader 会注入纯调度工作流：

- Lead 只负责模型辅助拆解、依赖分析、分配、进度跟踪和结果合并
- 文件读写与命令执行交给配置了独立工作树的成员
- 注入 10 阶段工作流指引

### 合并策略

成员完成 → Lead 增量合并 worktree：
- 无冲突 → 自动 `git merge --commit`
- 有冲突 → LLM 逐文件裁决
- 裁决失败 → 回滚，记录不可解决的冲突

---

## 14. 项目指令与笔记

### 项目指令

在项目根目录创建 `TINYCODE.md`：

```markdown
# 项目规范
- 使用 Python 3.12+，类型注解必须完整
- 测试用 pytest，覆盖率 ≥ 80%
- 禁止使用 `print`，统一走 `logging`

@include(sub/ci-rules.md)
```

`@include` 支持最大 3 层嵌套，拒绝越界路径。

启动时自动读取并作为 System 消息注入对话开头。

全局指令放在 `~/.tinyCode/instructions.md`（优先级低于项目级）。

### 自动笔记

每 5 轮对话自动调 LLM 更新笔记，四个分类分别存储：

| 分类 | 存放位置 |
|------|---------|
| 用户偏好 | `~/.tinyCode/notes/user_preferences.md` |
| 纠正反馈 | `~/.tinyCode/notes/corrections.md` |
| 项目知识 | `.tinyCode/notes/project_knowledge.md` |
| 参考资料 | `.tinyCode/notes/references.md` |

```
/memory show                # 查看各分类大小
/memory show 项目知识        # 查看具体内容
/memory clear 纠正反馈       # 清空
/memory edit 用户偏好        # 显示文件路径
```

---

## 15. 完整快捷键

| 按键 | 功能 |
|------|------|
| **Enter** | 提交消息 / 执行命令 |
| **Ctrl+C / Ctrl+D** | 中断输入并退出（自动保存+笔记更新） |
| **Ctrl+P** | 切换 plan-only 模式 |
| **Ctrl+S** | 循环切换安全等级 |
| **Ctrl+Q** | 手动触发上下文压缩 |
| **Tab** | 补全 `/` 开头的命令名 |
| **A/S/P/D + Enter** | HITL 确认（允许/会话/永久/拒绝） |

### 启动参数

```bash
python -m tinyCode --mode strict     # 安全等级
python -m tinyCode --resume          # 恢复上次 Worktree（不是聊天会话）
python -m tinyCode --trust-project-config  # 信任项目 MCP/Hook 配置
TINYCODE_CONFIG=/path/to/config.yaml python -m tinyCode
```

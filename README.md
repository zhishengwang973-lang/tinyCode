# TinyCode

终端 AI 编程助手（类似 Claude Code），用 Python 开发。

## 特性

- **多 Provider 支持**：Anthropic Claude / OpenAI / DeepSeek，通过 YAML 配置切换
- **流式 TUI**：Rich 连续输出 + Prompt Toolkit 输入补全；进度瞬时刷新，不混入回答正文
- **内置工具体系**：读/写/编辑/删除文件、执行命令、Glob/Grep 搜索，并在运行时接入 Skill 和子 Agent 工具
- **16 个内置斜杠命令模块**：`/help` `/clear` `/compress` `/mode` `/status` `/config` `/prompt` `/exit` `/review` `/skill` `/team` 等，Skill 还可自动注册专属命令
- **纵深安全防御**：黑名单拦截、路径沙箱、人在回路确认、三档权限模式
- **MCP 协议**：支持 Stdio 和 HTTP 传输，连接外部工具服务器
- **YAML+MD Skill 系统**：可编程 SOP 指令，三级优先级覆盖
- **事件 Hook 引擎**：12 种生命周期事件 + 条件匹配 + 4 种动作
- **子 Agent + Team 编排**：Fork 模式继承上下文、后台任务、共享任务清单，`/team run` 可触发团队编排
- **Git Worktree 隔离**：子 Agent 在独立工作目录中操作，退出自动清理
- **两层 Token 管理**：工具结果截断（层1）+ 结构化 LLM 摘要（层2）
- **JSONL 会话持久化**：追加写 O(1)、崩溃恢复、损坏行跳过

## 快速开始

```bash
python3 -m pip install -e .
cp example.tinyCode.yaml .tinyCode.yaml
# 编辑 .tinyCode.yaml，推荐用 api_key_env 引用环境变量
python3 -m tinyCode
# 或安装后在任意目录运行
tinyCode
```

`-e` 是可编辑安装：源码修改会立即生效，通常无需重复安装。若要在任意新目录启动，
将 Provider 配置放到 `~/.tinyCode/config.yaml`，进入目标目录后直接执行 `tinyCode`。
普通新目录无需是 Git 仓库；只有 Worktree 和 Team 多工作树功能依赖 Git。

## 开发验证

```bash
python3 -m pip install -e '.[dev]'
python3 -m unittest discover -s tests
ruff check tinyCode tests
mypy tinyCode
coverage run -m unittest discover -s tests && coverage report --fail-under=68
```

## 项目结构

```
tinyCode/
├── main.py              # 启动编排
├── config/              # YAML 配置
├── providers/           # LLM 后端（Anthropic/OpenAI/DeepSeek）
├── agent/               # ReAct 循环 + 事件流
├── conversation/        # 历史/截断/摘要/压缩
├── prompts/             # 模块化 Prompt
├── tools/               # 内置工具 + 注册中心
├── tui/                 # Rich 输出 + Prompt Toolkit 输入
├── storage/             # JSONL 会话存储
├── security/            # 纵深防御
├── mcp/                 # MCP 协议客户端
├── commands/            # 内置斜杠命令
├── skills/              # YAML+MD Skill 系统
├── hooks/               # 事件 Hook 引擎
├── subagent/            # 子 Agent 系统
├── worktree/            # Git 工作目录管理
├── teams/               # Agent Team 编排
├── instructions/        # 项目指令加载
└── notes/               # 自动笔记
```

## 文档

- [使用教程](TUTORIAL.md)
- [Spec](spec.md)
- [任务列表](tasks.md)
- [验收清单](checklist.md)


## 要求

- Python ≥ 3.10
- Git ≥ 2.30（Worktree 功能需要）

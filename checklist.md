# TinyCode — Checklist

每项可勾选、可观测。

---

## 配置系统

- [ ] 启动时当前目录无 `.tinyCode.yaml`，`~/.tinyCode/config.yaml` 也不存在 → 打印配置文件缺失提示并退出
- [ ] `protocol` / `model` / `api_key` 分别缺失 → 打印 "Provider 'X' 缺少字段: Y" 并退出
- [ ] `protocol: unknown` → 打印 "不支持的协议: unknown"
- [ ] 环境变量 `TINYCODE_CONFIG` 指向的路径优先于 `.tinyCode.yaml`
- [ ] 当前目录 `.tinyCode.yaml` 优先于 `~/.tinyCode/config.yaml`
- [ ] `base_url` 不填：Anthropic 默认 `api.anthropic.com`，OpenAI 默认 `api.openai.com`，DeepSeek 默认 `api.deepseek.com`
- [ ] `providers` 配置 3 个 provider → `active_provider` 选到正确的

## Provider 层

- [ ] `BaseProvider` 不可直接实例化 → TypeError
- [ ] `create_provider(protocol="anthropic")` → AnthropicProvider
- [ ] `create_provider(protocol="openai")` → OpenAIProvider
- [ ] `create_provider(protocol="deepseek")` → DeepSeekProvider
- [ ] Anthropic SSE 流式返回 "Hello world" → 逐字出现在 TUI
- [ ] OpenAI SSE 流式返回 "Hello world" → 逐字出现
- [ ] DeepSeek reasoner → reasoning_content 以 `[Reasoning]` 灰色渲染
- [ ] API key 无效 → 红色错误显示，程序不崩溃

## 对话历史

- [ ] "My name is Alice" → AI 回复 → "What is my name?" → 回复含 "Alice"
- [ ] 空对话 token 估算 = 0
- [ ] 10 条消息后 token 估算增长
- [ ] `clear()` 后 `get_messages()` 为空列表

## 工具系统

- [ ] `read_file("README.md")` → 返回完整内容
- [ ] `read_file("nonexistent.txt")` → `ToolResult(success=False, error="文件不存在")`
- [ ] `write_file("new.txt", "hello")` → 创建文件
- [ ] `write_file("new.txt", "again")` → "文件已存在" 错误
- [ ] `edit_file("new.txt", "hello", "hi")` → 替换成功，"已编辑文件: 1 处"
- [ ] `edit_file("new.txt", "not_found", "x")` → "未找到匹配的原文" 含文件预览
- [ ] `edit_file("new.txt", "h", "x")` → "找到 N 处匹配，请提供更长的上下文"
- [ ] `glob("**/*.py")` → 返回匹配列表
- [ ] `grep("def main")` → 返回文件名:行号:内容
- [ ] `delete_file("new.txt")` → 删除文件；目录或不存在路径返回结构化错误
- [ ] `run_command("echo hello")` → 退出码 0，stdout 含 "hello"
- [ ] `run_command("rm -rf /")` → 黑名单拦截
- [ ] 大型工具结果 → 可通过 `tool_result_search` / `tool_result_read` 分页检索
- [ ] 工具超时 → `ToolResult(success=False, error="超时")`

## Agent 循环

- [ ] 用户发消息 → ReAct 循环启动 → LLM 调工具 → 工具结果回填 → LLM 继续 → 无工具调用终止
- [ ] 多工具调用 → 读并发/写串行
- [ ] cancel 信号 → AgentDoneEvent("cancelled")
- [ ] 超过 max_rounds → AgentDoneEvent("max_rounds")

## TUI 交互

- [ ] 启动显示 Provider + Model + MCP server 数
- [ ] Enter 提交 → AI 流式回复逐字出现
- [ ] 用户消息绿色加粗，AI 回复蓝色加粗
- [ ] 工具调用在临时进度行显示，且不混入回答正文
- [ ] 模型开始输出时进度行收起，回答连续且每个分片只出现一次
- [ ] 正常结束显示 `✓ 本轮已正常完成`
- [ ] Ctrl+C → 退出，打印 "Goodbye!"
- [ ] Ctrl+P → "Plan-only 模式: ON/OFF"
- [ ] Ctrl+S → 循环 "严格/默认/放行"
- [ ] Ctrl+Q → 手动压缩
- [ ] Tab → 补全 `/` 命令
- [ ] 等待模型/执行工具时显示单条可擦除进度

## 命令系统

- [ ] `/help` → 列出所有可见命令
- [ ] `/help compress` → 显示压缩命令详情
- [ ] `/clear` → 对话清空 + Skill 激活清空
- [ ] `/compress` → 手动触发压缩并显示结果
- [ ] `/mode plan` → 切换 plan-only
- [ ] `/mode security strict` → 安全等级切换
- [ ] `/status` → 显示工作目录/OS/Plan/Sec/token
- [ ] `/review` → 注入代码审查提示词
- [ ] `/skill list` → 列出 Skills
- [ ] `/tasks list` → 列出后台任务
- [ ] `/worktree status` → 当前工作目录状态
- [ ] `/team list` → 列出 Team 定义
- [ ] 未知命令 `/xyz` → "未知命令: xyz。输入 /help 查看可用命令列表"
- [ ] 非 `/` 输入 → 发给 AI

## 安全系统

- [ ] `rm -rf /` → 黑名单拦截（所有档位生效）
- [ ] 路径 `../outside/file.txt` → 沙箱拦截
- [ ] 严格模式：写工具全部询问
- [ ] 默认模式：读放行写询问
- [ ] 放行模式：普通操作放行；黑名单、凭据文件保护和项目路径沙箱仍始终生效
- [ ] HITL 弹窗：按 A → "允许(本次)"、按 S → "允许(本会话)"、按 P → "允许(永久)"、按 D → "拒绝"
- [ ] 永久允许 → `.tinyCode-security.yaml` 写入新规则
- [ ] 下次同样操作 → 规则命中，不再询问

## MCP 协议

- [ ] `.tinyCode-mcp.yaml` 配置 stdio server → 启动时 `npx ...` 子进程连接成功
- [ ] HTTP server → `POST /mcp` 连接成功
- [ ] Server 连接失败 → stderr 打印警告，不阻塞启动
- [ ] 连接成功后 → `list_tools()` 返回工具列表 → MCPToolAdapter 注册
- [ ] MCP 工具命名：清洗为 Provider 兼容的 `mcp_<server>_tool_<name>`（最长 64 字符）
- [ ] `mcp_<server>_resource` 首次调用 → 延迟 `list_resources()`
- [ ] `mcp_<server>_prompt` 首次调用 → 延迟 `list_prompts()`

## Skill 系统

- [ ] 启动时扫描 `.tinyCode/skills/` + `~/.tinyCode/skills/` + 内置
- [ ] 项目级 Skill 覆盖同名内置 Skill
- [ ] 白名单工具不存在 → fail-fast 退出
- [ ] Phase 1：名字+描述注入 Instructions
- [ ] Phase 2：`skill_loader(name="commit")` → 完整 SOP 钉入环境上下文
- [ ] 工具白名单生效：Skill 声明 `tools: [read_file, glob]` → Agent 只能看到这两个工具 + skill_loader
- [ ] `/skill list` → 列出所有 Skill
- [ ] `/clear` → 激活 Skill 列表清空

## Skill 内置

- [ ] `/commit` → 激活 commit Skill → Agent 执行 git diff → 生成 Conventional Commits
- [ ] `/review` → 注入代码审查提示词 → isolated 模式执行
- [ ] `/test` → 激活 test Skill → 查找测试框架 → 生成测试

## Hook 系统

- [ ] `tool_pre_exec` + regex 匹配 `rm -rf` → prompt_inject 返回拦截原因 → Agent 收到错误调整
- [ ] `tool_post_exec` + write_file → shell 动作记录日志
- [ ] 拦截事件 `async: true` → YAML 加载时报错
- [ ] 未知事件名 → YAML 加载时报错 + 定位到规则
- [ ] Hook 动作执行失败 → stderr 日志，Agent 主流程不中断
- [ ] `once: true` → 第二次触发跳过

## 子 Agent

- [ ] `sub_agent(task="探索项目", role="explorer")` → 返回结构化报告
- [ ] Explorer 只能使用 read_file/glob/grep
- [ ] Fork 模式（不指定 role）→ 后台运行 → 结果自动注入
- [ ] 子 Agent 调用 `sub_agent` → 工具不可见（全局禁止）
- [ ] `/tasks list` → 显示后台任务状态
- [ ] `/tasks kill <id>` → 终止运行中任务

## 上下文管理

- [ ] 工具结果 > 50K 字符 → 写盘 `~/.tinyCode/tool_results/` → 对话留 2K 预览 → TUI 显示 `📎 结果过大 → 存盘`
- [ ] 单轮合计 > 200K → 截断最大的结果
- [ ] Token ≥ 70% 窗口 → 警告并自动尝试压缩 → 显示压缩前后 token 估算
- [ ] 任务结束显示上下文已用/总量/百分比/剩余量，≥70% 黄色、≥90% 红色
- [ ] 压缩后对话能正确回答相关问题
- [ ] 连续 2 次摘要失败 → 熔断 → "压缩熔断——已停止自动压缩"
- [ ] `/compress` 手动触发 → 重置熔断

## Worktree

- [ ] `/worktree create test-wt` → 创建 worktree + 复制 config + symlink 依赖
- [ ] `/worktree list` → 显示所有工作目录（● 当前）
- [ ] `/worktree enter test-wt` → 切换目录
- [ ] `/worktree exit test-wt` → 有修改拒绝删除 → `--force` 强制
- [ ] 名称含 `..` → 校验拒绝
- [ ] `--resume` → 恢复上次 worktree 会话

## Agent Team

- [ ] `~/.tinyCode/teams/example.json` 定义 Team → `/team list` 显示
- [ ] Lead 拆分目标 → SharedTaskList 创建任务
- [ ] 成员完成 → 通知 Lead → Lead 增量合并 worktree
- [ ] 合并冲突 → LLM 逐文件裁决
- [ ] 裁决失败 → 回滚 + 上报
- [ ] 双锁调度模式 → Lead 失去 read_file/write_file/edit_file/run_command
- [ ] `team_send_message` → 点对点消息写入目标邮箱
- [ ] `team_broadcast` → 广播到所有成员

## 指令与笔记

- [ ] 项目根目录 `TINYCODE.md` 存在 → 内容作为 System 消息注入
- [ ] `@include(sub/file.md)` → 嵌套解析（≤3 层）
- [ ] `@include(../../etc/passwd)` → 越界拦截
- [ ] 每 5 轮 → 四个笔记文件更新
- [ ] `/memory show` → 列出各分类大小
- [ ] `/memory show 项目知识` → 查看内容
- [ ] Ctrl+C → 最终笔记更新触发

## 会话持久化

- [ ] 对话 3 轮后 Ctrl+C → `~/.tinyCode/sessions/{id}.jsonl` 写入
- [ ] Meta JSON 含 id/title/created_at/last_active_at/message_count/model/provider
- [ ] 重新启动 → 创建新会话，不意外重放上次任务
- [ ] `/session list` + `/session load <id>` → 手动恢复后 AI 可引用恢复前内容
- [ ] JSONL 损坏行 → 跳过继续
- [ ] 未配对 tool_use → 截断到最后完整位置
- [ ] 时间跨度 > 30 分钟 → 注入 `[时间跨度提醒]`
- [ ] 旧 `default.json` → 启动时自动迁移为 JSONL

## 端到端验收

- [ ] **核心场景**：配置 Anthropic → 启动 → "读 README.md" → 流式输出 → "第一点再详细说一下" → AI 记住上文 → 安全退出 → 重启后用 `/session load <id>` 恢复 → "我刚才让你介绍什么？" → AI 提到 README

- [ ] **Provider 切换**：改 `active_provider` → 重启 → OpenAI 流式返回正常

- [ ] **工具调用完整链路**："帮我找到所有 Python 文件并搜索 main 函数" → glob → grep → 汇总结果

- [ ] **安全拦截**：默认模式 → "删除 build 目录" → Agent 调 run_command → HITL 弹窗 → D 拒绝 → Agent 收到原因调整

- [ ] **上下文压缩**：连续 30+ 条消息 → 压缩触发 → 摘要显示 → 对话继续正常

- [ ] **MCP 集成**：配置 filesystem server → 启动显示 "MCP: 1 server" → Agent 调用 MCP 工具

- [ ] **Skill 激活**：`/commit` → Skill 激活 → "生成提交信息" → Agent 按 SOP 执行

- [ ] **子 Agent**："用 explorer 探索项目" → 子 Agent 运行 → 报告注入主对话

- [ ] **Worktree**：`/worktree create wt1` → `/worktree enter wt1` → 在新目录修改文件 → `/worktree exit wt1 --force`

- [ ] **错误恢复**：错误 api_key → 输入消息 → 显示 401 错误 → 不崩溃 → 可继续修改配置重试

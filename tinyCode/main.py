"""Main entry point — wires config, prompts, security, MCP, agent loop, history, and TUI."""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from tinyCode.agent.loop import AgentLoop
from tinyCode.cli import CLIOptions, parse_cli_args
from tinyCode.config.loader import ConfigError, load_config
from tinyCode.instructions import InstructionsLoader
from tinyCode.mcp.manager import MCPManager
from tinyCode.notes import AutoNoteManager
from tinyCode.hooks import load_hooks, HookEngine, HookEvent
from tinyCode.skills import SkillLoader, SkillRegistry, SkillTool
from tinyCode.subagent import RoleLoader, SubAgentRunner, BackgroundTaskManager, SubAgentTool
from tinyCode.teams import run_team
from tinyCode.worktree import GitWorktreeManager, BackgroundCleaner
from tinyCode.providers.base import create_provider
from tinyCode.conversation.history import ConversationHistory
from tinyCode.conversation.compression import ContextCompressor
from tinyCode.conversation.truncator import ToolResultTruncator
from tinyCode.prompts import (
    PromptBuilder, PromptInjector, collect_current_time, collect_environment,
)
from tinyCode.security import SecurityGuard, SecurityPolicy, PathSandbox, SecurityLevel
from tinyCode.storage.sessions import SessionStore
from tinyCode.tools import (
    ToolRegistry,
    ToolExecutor,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ApplyPatchTool,
    DeleteFileTool,
    RunCommandTool,
    GlobTool,
    GrepTool,
    ToolResultSearchTool,
    ToolResultReadTool,
    RequestUserInputTool,
    WebSearchTool,
    WebFetchTool,
)
from tinyCode.tui import create_tui, fullscreen_supported
from tinyCode.tui.app import TinyCodeTUI
from tinyCode.tracing import TraceRecorder


class _CleanupStack:
    """Best-effort async cleanup that never hides the primary failure."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, Callable[[], Awaitable[None]]]] = []

    def add(self, label: str, cleanup: Callable[[], Awaitable[None]]) -> None:
        self._steps.append((label, cleanup))

    async def close(self) -> None:
        cancellation: asyncio.CancelledError | None = None
        for label, cleanup in reversed(self._steps):
            try:
                await cleanup()
            except asyncio.CancelledError as exc:
                # One subsystem cancelling its own cleanup must not strand all
                # resources registered before it.  Preserve cancellation, but
                # only re-raise after every best-effort cleanup has run.
                cancellation = cancellation or exc
            except Exception as exc:
                print(
                    f"{label} 清理失败: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        self._steps.clear()
        if cancellation is not None:
            raise cancellation


def _create_tool_registry(
    tool_result_storage_dir: Path | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ApplyPatchTool())
    registry.register(DeleteFileTool())
    registry.register(RunCommandTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(ToolResultSearchTool(tool_result_storage_dir))
    registry.register(ToolResultReadTool(tool_result_storage_dir))
    registry.register(RequestUserInputTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    return registry


async def _run_application(options: CLIOptions, cleanup: _CleanupStack) -> int:
    # Project MCP/Hook files are executable extension points.  A repository
    # must not gain local-code execution merely by being opened.
    trust_project_config = options.trust_project_config
    if not trust_project_config:
        ignored = [
            name for name in (".tinyCode-mcp.yaml", ".tinyCode-hooks.yaml")
            if (Path.cwd() / name).exists()
        ]
        if ignored:
            print(
                "安全提示: 已忽略项目可执行配置 " + ", ".join(ignored)
                + "；确认信任该项目后可使用 --trust-project-config",
                file=sys.stderr,
            )
    # 1. Load configuration
    try:
        app_config = load_config()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # 2. Find active provider
    active_config = None
    for p in app_config.providers:
        if p.name == app_config.active_provider:
            active_config = p
            break
    if active_config is None:
        print(f"active_provider '{app_config.active_provider}' 未找到", file=sys.stderr)
        return 2

    # 3. Create provider
    try:
        provider = create_provider(active_config)
    except ValueError as e:
        print(f"无法创建 provider: {e}", file=sys.stderr)
        return 2
    cleanup.add("provider", provider.close)

    # 4. Conversation history + session store + migration
    history = ConversationHistory()
    try:
        session_store = SessionStore()
    except OSError as exc:
        print(
            f"无法初始化会话存储: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    session_store.migrate_old_format()  # one-time: default.json → JSONL
    if session_store.last_migration_error:
        print(
            f"旧会话迁移失败: {session_store.last_migration_error}",
            file=sys.stderr,
        )
    try:
        session_store.new_session()  # fresh session ID each startup
    except OSError as exc:
        print(
            f"无法创建会话: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    # Old sessions are NOT auto-restored — use /session load <id> manually

    # 4.5. Instructions
    instructions_loader = InstructionsLoader()
    instruction_parts: list[str] = []
    for label, load_instruction in (
        ("项目指令", instructions_loader.load_project),
        ("用户指令", instructions_loader.load_user),
    ):
        try:
            text = load_instruction()
            if text:
                instruction_parts.append(text)
        except (OSError, UnicodeError, ValueError) as exc:
            print(
                f"{label}加载失败，已跳过: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    instructions_text = "\n\n".join(instruction_parts)

    # 5. Prompt system
    prompt_builder = PromptBuilder()
    prompt_injector = PromptInjector()
    environment_text = collect_environment

    # 6. Context compressor + truncator
    compressor = ContextCompressor(model=active_config.model, provider=provider)
    truncator = ToolResultTruncator()
    if truncator.storage_error:
        print(
            f"工具结果缓存不可用，将仅保留截断预览: {truncator.storage_error}",
            file=sys.stderr,
        )
    note_manager: AutoNoteManager | None = None
    if app_config.notes_enabled:
        note_manager = AutoNoteManager(provider=provider, interval=5)
        for error in note_manager.last_errors:
            print(f"笔记初始化: {error}", file=sys.stderr)

    # 7. Tool registry (create early — needed by skills MCP subagent)
    tool_registry = _create_tool_registry(truncator.storage_dir)
    trace_recorder = TraceRecorder(app_config.tracing, Path.cwd())

    # 7.5. Skills (needs tool_registry for whitelist validation)
    skill_loader = SkillLoader()
    skill_registry = SkillRegistry(skill_loader)
    skill_metas = skill_registry.load_all()
    # Register skill_loader as always-available system tool
    skill_tool = SkillTool(skill_registry)
    tool_registry.register(skill_tool)

    # Skill summaries for the prompt
    skill_summaries = "\n".join(
        f"- `skill_loader(name=\"{m.name}\")`: {m.description}" for m in skill_metas
    ) if skill_metas else ""
    if skill_summaries:
        instructions_text = (
            f"## 可用 Skills\n"
            f"使用 skill_loader 工具激活 Skill 以获取 SOP 指令。"
            f"激活后每轮 LLM 调用都会看到 Skill 的完整指令。\n"
            f"{skill_summaries}\n\n" + instructions_text
        )

    # 8. MCP — discover external servers and register their tools
    mcp_manager = MCPManager(tool_registry)
    cleanup.add("MCP", mcp_manager.shutdown)
    mcp_manager.load_config(include_project=trust_project_config)
    for error in mcp_manager.config_errors:
        print(f"MCP 配置: {error}", file=sys.stderr)
    if mcp_manager.is_configured:
        mcp_count = await mcp_manager.discover_and_register()
        if mcp_count > 0:
            print(f"MCP: 已注册 {mcp_count} 个远端工具/资源/提示词", file=sys.stderr)

    # 9. Security system
    security_level = SecurityLevel(app_config.security_level)
    sandbox = PathSandbox()
    policy = SecurityPolicy(level=security_level)
    for error in policy.load_errors:
        print(f"安全配置: {error}", file=sys.stderr)
    security_guard = SecurityGuard(
        policy=policy,
        sandbox=sandbox,
        level=security_level,
    )

    # 10. Tool executor & Agent Loop
    tool_executor = ToolExecutor(default_timeout=30.0)
    # Hooks
    hook_rules = load_hooks(include_project=trust_project_config)
    hook_engine = HookEngine(hook_rules)

    # Sub-agent system
    role_loader = RoleLoader()
    roles = role_loader.load_all()
    sub_runner = SubAgentRunner(
        provider,
        tool_registry,
        tool_executor,
        roles,
        trace_recorder=trace_recorder,
    )
    task_manager = BackgroundTaskManager()
    cleanup.add("sub-agent", task_manager.shutdown)
    sub_agent_tool = SubAgentTool(sub_runner, task_manager, roles, history)
    tool_registry.register(sub_agent_tool)

    # Validate only after built-ins, MCP and sub_agent are all registered so
    # skills may intentionally whitelist those runtime-provided tools.
    for meta in skill_metas:
        for tool_name in meta.tools or []:
            if not tool_registry.get(tool_name):
                print(
                    f"Skill [{meta.name}]: 白名单工具 '{tool_name}' 不存在",
                    file=sys.stderr,
                )
                return 2

    async def hook_sub_agent(task: str) -> str:
        result = await sub_agent_tool.execute(task=task, background=True)
        if not result.success:
            raise RuntimeError(result.error)
        return result.content

    hook_engine.set_sub_agent_handler(hook_sub_agent)

    lifecycle_started = {"system": False, "session": False}

    async def shutdown_hooks() -> None:
        if lifecycle_started["session"]:
            await hook_engine.fire(HookEvent.SESSION_END, {
                "session_id": session_store.current_id or "",
            })
        if lifecycle_started["system"]:
            await hook_engine.fire(
                HookEvent.SYSTEM_SHUTDOWN, {"cwd": str(Path.cwd())},
            )
        await hook_engine.shutdown(cancel=False)

    cleanup.add("hooks", shutdown_hooks)

    # Worktree management
    worktree_manager = GitWorktreeManager()
    cleaner = BackgroundCleaner(worktree_manager)
    cleaner.start()
    cleanup.add("worktree cleaner", cleaner.stop)

    # --resume: restore previous worktree session
    if options.resume:
        session = worktree_manager.load_session()
        if session and session.get("active_worktree"):
            resumed, resume_error = await worktree_manager.enter(
                session["active_worktree"]
            )
            if resumed:
                security_guard.set_project_root(Path.cwd())
                trace_recorder.set_project_root(Path.cwd())
                if note_manager is not None:
                    note_manager.set_cwd(Path.cwd())
            else:
                print(f"恢复 Worktree 失败: {resume_error}", file=sys.stderr)

    agent_loop = AgentLoop(
        provider=provider,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        prompt_injector=prompt_injector,
        security_guard=security_guard,
        truncator=truncator,
        note_manager=note_manager,
        skill_registry=skill_registry,
        hook_engine=hook_engine,
        instructions_text=instructions_text,
        environment_text=environment_text,
        current_time_text=collect_current_time,
        max_rounds=app_config.max_rounds,
        round_extension=app_config.round_extension,
        hard_max_rounds=app_config.hard_max_rounds,
        round_limit_action=app_config.round_limit_action,
        compressor=compressor,
        trace_recorder=trace_recorder,
    )

    tui_ref: dict[str, TinyCodeTUI] = {}

    async def team_progress(message: str) -> None:
        current_tui = tui_ref.get("tui")
        if current_tui is not None:
            current_tui.update_progress(message)

    async def team_runner(name: str, goal: str) -> str:
        # The periodic cleaner cannot distinguish a team-owned worktree from
        # an idle one using Git state alone. Pause it for the full team run so
        # it can never delete a clean, reused member workspace between model
        # planning and the member's first write.
        await cleaner.stop()
        try:
            return await run_team(
                name,
                goal,
                provider=provider,
                tool_registry=tool_registry,
                tool_executor=tool_executor,
                roles=roles,
                preapproved=True,
                progress=team_progress,
            )
        finally:
            cleaner.start()

    # Apply --mode CLI flag if any
    if options.mode:
        level = SecurityLevel(options.mode)
        security_level = level
        security_guard.set_level(level)
        agent_loop.set_security_level(level)

    # 11. Launch the configured TUI. Fullscreen rendering requires a real
    # terminal; IDE consoles and redirected stdin retain the proven stream UI.
    if app_config.ui_mode == "fullscreen" and not fullscreen_supported():
        print(
            "全屏 UI 需要真实终端，已自动降级为 stream 模式",
            file=sys.stderr,
        )
    tui = create_tui(
        ui_mode=app_config.ui_mode,
        agent_loop=agent_loop,
        history=history,
        compressor=compressor,
        session_store=session_store,
        note_manager=note_manager,
        provider_name=active_config.name,
        model=active_config.model,
        security_level=security_level,
        mcp_server_count=len(mcp_manager.connected_servers),
        skill_registry=skill_registry,
        task_manager=task_manager,
        worktree_manager=worktree_manager,
        team_runner=team_runner,
        trace_recorder=trace_recorder,
    )
    cleanup.add("TUI", tui.shutdown)
    tui_ref["tui"] = tui
    request_input_tool = tool_registry.get("request_user_input")
    if isinstance(request_input_tool, RequestUserInputTool):
        request_input_tool.set_handler(tui.request_tool_input)
    await hook_engine.fire(HookEvent.SYSTEM_STARTUP, {"cwd": str(Path.cwd())})
    lifecycle_started["system"] = True
    await hook_engine.fire(HookEvent.SESSION_START, {
        "session_id": session_store.current_id or "",
    })
    lifecycle_started["session"] = True
    await tui.run_async()
    return 0


async def main(options: CLIOptions | None = None) -> int:
    parsed = options or parse_cli_args()
    cleanup = _CleanupStack()
    try:
        return await _run_application(parsed, cleanup)
    except Exception as exc:
        # Last-resort process boundary: subsystems should already return
        # structured errors, but a damaged optional file or third-party client
        # must still produce a readable non-zero exit instead of an unexplained
        # traceback that looks like the TUI crashed.
        print(
            f"TinyCode 未能继续运行: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        await cleanup.close()


def entry_point() -> None:
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 130
    if exit_code:
        raise SystemExit(exit_code)

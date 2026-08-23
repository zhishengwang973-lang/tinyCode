"""Rich streaming CLI with prompt_toolkit input and completion."""

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.status import Status
from rich.text import Text

from tinyCode.agent.events import (
    AgentDoneEvent,
    ErrorEvent,
    HITLRequestEvent,
    RoundLimitDecision,
    RoundLimitDecisionAction,
    RoundLimitExtendedEvent,
    RoundLimitReachedEvent,
    RoundStartEvent,
    ProgressWarningEvent,
    SteeringAppliedEvent,
    TaskStalledDecision,
    TaskStalledDecisionAction,
    TaskStalledEvent,
    ThinkingEvent,
    TextDeltaEvent,
    ToolBlockedEvent,
    ToolCallEvent,
    ToolResultEvent,
    TruncationEvent,
    ContextCompressionEvent,
)
from tinyCode.agent.runtime import TurnRuntime
from tinyCode.commands import CommandDispatcher, CommandRegistry, UIControl, register_builtins
from tinyCode.providers.base import TokenUsage
from tinyCode.security.models import HITLDecision, SecurityLevel
from tinyCode.tui.render import STYLE
from tinyCode.tui.metrics import TurnMetrics
from tinyCode.tui.workspace_changes import WorkspaceChanges, WorkspaceSnapshot

if TYPE_CHECKING:
    from tinyCode.agent.loop import AgentLoop
    from tinyCode.conversation.compression import ContextCompressor
    from tinyCode.conversation.history import ConversationHistory
    from tinyCode.notes.manager import AutoNoteManager
    from tinyCode.skills.registry import SkillRegistry
    from tinyCode.storage.sessions import SessionStore


class _StreamTextNormalizer:
    """Normalize a fully escaped Markdown fence without corrupting source code.

    Some OpenAI-compatible gateways return an entire fenced response with
    literal ``\\n`` separators.  Replacing those sequences unconditionally is
    unsafe because valid code can itself contain ``"\\n"``.  Detect the broken
    wire form only from the opening Markdown fence and keep normal streams
    byte-for-byte intact.
    """

    _ESCAPED_FENCE = re.compile(r"^\s*```[^\n\\]{0,32}\\(?:r\\n|n)")

    def __init__(self) -> None:
        self._probe = ""
        self._escaped: bool | None = None

    def normalize(self, text: str) -> str:
        if self._escaped is None:
            self._probe = (self._probe + text)[:256]
            if "\n" in self._probe or "\r" in self._probe:
                self._escaped = False
            elif self._ESCAPED_FENCE.match(self._probe):
                self._escaped = True
            elif len(self._probe) >= 256:
                self._escaped = False
        if not self._escaped:
            return text
        return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


@dataclass
class _PendingControlInput:
    prompt: list[tuple[str, str]]
    future: asyncio.Future[str]


class _CommandCompleter(Completer):
    """Tab completer for slash commands."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(self, document, complete_event):
        text = document.text
        if text.startswith("/"):
            prefix = text[1:]
            for name in self._registry.get_completions(prefix):
                # Registry completions already include the leading slash.
                yield Completion(name, start_position=-len(text))


class _PlainInputSession:
    """Minimal input fallback for IDE consoles whose stdin is not a TTY."""

    def __init__(self, message: str = "› ") -> None:
        self._message = message

    async def prompt_async(self, message=None) -> str:
        prompt = self._message
        if callable(message):
            message = message()
        if isinstance(message, (list, tuple)):
            prompt = "".join(fragment[1] for fragment in message)
        elif isinstance(message, str):
            prompt = message
        return await asyncio.to_thread(input, prompt)


class TinyCodeTUI(UIControl):
    """Line-oriented terminal UI.

    Rich owns output and writes each stream delta exactly once. prompt_toolkit is
    intentionally limited to the input line so terminal scrollback remains the
    source of truth and progress redraws can never become conversation content.
    """

    def __init__(
        self,
        agent_loop: "AgentLoop",
        history: "ConversationHistory",
        compressor: "ContextCompressor",
        session_store: "SessionStore",
        note_manager: "AutoNoteManager | None",
        provider_name: str,
        model: str,
        security_level: SecurityLevel = SecurityLevel.NORMAL,
        mcp_server_count: int = 0,
        skill_registry: "SkillRegistry | None" = None,
        task_manager=None,
        worktree_manager=None,
        team_runner=None,
        console: Console | None = None,
        prompt_session: Any | None = None,
    ) -> None:
        self._agent_loop = agent_loop
        self._runtime = TurnRuntime(agent_loop)
        self._history = history
        self._compressor = compressor
        self._session_store = session_store
        self._note_manager = note_manager
        self._skill_registry = skill_registry
        self._provider_name = provider_name
        self._model = model
        self._mcp_server_count = mcp_server_count

        self._cmd_registry = CommandRegistry()
        register_builtins(
            self._cmd_registry,
            ui=self,
            note_manager=note_manager,
            skill_registry=skill_registry,
            task_manager=task_manager,
            worktree_manager=worktree_manager,
            team_runner=team_runner,
        )
        if skill_registry:
            for meta in skill_registry.list_available():
                self._register_skill_command(meta)
        self._cmd_dispatcher = CommandDispatcher(self._cmd_registry, ui=self)
        self._completer = _CommandCompleter(self._cmd_registry)
        self._input_keybindings = self._build_input_keybindings()

        self._console = console or Console(highlight=False)
        self._stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self._uses_prompt_toolkit = prompt_session is None and self._stdin_is_tty
        if prompt_session is not None:
            self._prompt_session = prompt_session
        elif self._stdin_is_tty:
            self._prompt_session = PromptSession(
                message=[("class:prompt", "› ")],
                completer=self._completer,
                style=STYLE,
                complete_while_typing=False,
                key_bindings=self._input_keybindings,
            )
        else:
            self._prompt_session = _PlainInputSession()
        self._progress: Status | None = None
        self._progress_text: str | None = None
        self._status_text = "就绪 · 可输入任务"
        self._note_update_task: asyncio.Task | None = None
        self._foreground_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._command_active = False
        self._pending_control_input: _PendingControlInput | None = None
        self._input_loop_active = False
        self._security_level = security_level
        self._closed = False
        self._exit_requested = False

    # -- UIControl implementation -----------------------------------------

    def show_system_message(self, text: str) -> None:
        self._print_info(text)

    async def request_tool_input(
        self, question: str, options: list[str],
    ) -> str | None:
        """Collect one model-requested answer without creating a new turn."""
        self._stop_progress()
        self._console.print()
        self._console.print(f"需要你确认：{question}", style="bold cyan", highlight=False)
        for index, option in enumerate(options, start=1):
            self._console.print(f"  {index}. {option}", highlight=False)
        prompt = "请输入选项序号或答案 › " if options else "请回答 › "
        while True:
            try:
                answer = (
                    await self._read_control_input(
                        [("class:prompt", prompt)]
                    )
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not answer:
                self._print_warning("回答不能为空")
                continue
            if options and answer.isascii() and answer.isdigit():
                selected = int(answer)
                if 1 <= selected <= len(options):
                    answer = options[selected - 1]
                else:
                    self._print_warning(f"请输入 1–{len(options)} 的序号，或直接输入答案")
                    continue
            self._start_progress("已收到回答 · 继续执行")
            return answer

    def send_to_conversation(self, text: str) -> None:
        """Inject a prompt-command result into the foreground conversation."""
        self._start_user_input(text, display_user=False)

    def toggle_plan_mode(self) -> bool:
        return self._agent_loop.toggle_plan_only()

    def set_security_level(self, level_name: str) -> str:
        try:
            level = SecurityLevel(level_name.lower())
            self._security_level = level
            self._agent_loop.set_security_level(level)
            return level.value
        except ValueError:
            return self._security_level.value

    def get_token_count(self) -> int:
        return self._history.estimated_token_count()

    def clear_conversation(self) -> None:
        self._history.clear()
        if self._skill_registry:
            self._skill_registry.clear_activated()
        self._console.clear()
        self._do_save()

    async def trigger_compress(self) -> str:
        self._compressor.reset_circuit()
        self._compressor.reset_warning()
        result = await self._compressor.check_and_compress(
            self._history, self._agent_loop.provider,
        )
        if result.was_compressed:
            released = max(
                0,
                result.estimated_tokens_before - result.estimated_tokens_after,
            )
            return (
                "上下文已压缩："
                f"≈{self._format_compact_tokens(result.estimated_tokens_before)} → "
                f"≈{self._format_compact_tokens(result.estimated_tokens_after)}，"
                f"释放 ≈{self._format_compact_tokens(released)}"
            )
        if result.error:
            return f"上下文压缩失败：{result.error}"
        if self._compressor.circuit_open:
            return "压缩熔断——已停止自动压缩"
        return "当前无需压缩"

    def get_session_list(self) -> list[dict]:
        return self._session_store.list_sessions()

    def load_session(self, session_id: str) -> str:
        restored = self._session_store.load(session_id)
        if restored is None:
            return f"会话 {session_id[:8]} 不存在"
        restored_history, _provider, _model = restored
        self._history.replace_messages(restored_history.get_messages())
        self._console.clear()
        return f"已加载会话 {session_id[:8]} ({len(restored_history)} 条消息)"

    def new_session(self) -> str:
        sid = self._session_store.new_session()
        self._history.clear()
        self._console.clear()
        return f"新会话已创建: {sid}"

    def delete_session(self, session_id: str) -> str:
        if self._session_store.delete(session_id):
            return f"会话 {session_id} 已删除"
        return f"会话 {session_id} 不存在"

    def get_plan_only(self) -> bool:
        return self._agent_loop.plan_only

    def get_security_level(self) -> str:
        return self._security_level.value

    def get_runtime_status(self) -> dict:
        snapshot = self._runtime.snapshot()
        return {
            "state": snapshot.state.value,
            "turn_id": snapshot.turn_id,
            "last_outcome": (
                snapshot.last_outcome.value if snapshot.last_outcome else "none"
            ),
        }

    def get_max_rounds(self) -> int:
        return self._agent_loop.max_rounds

    def set_max_rounds(self, value: int) -> int:
        return self._agent_loop.set_max_rounds(value)

    def get_round_extension(self) -> int:
        return self._agent_loop.round_extension

    def set_round_extension(self, value: int) -> int:
        return self._agent_loop.set_round_extension(value)

    def get_hard_max_rounds(self) -> int:
        return self._agent_loop.hard_max_rounds

    def set_hard_max_rounds(self, value: int) -> int:
        return self._agent_loop.set_hard_max_rounds(value)

    def get_round_limit_action(self) -> str:
        return self._agent_loop.round_limit_action

    def set_round_limit_action(self, value: str) -> str:
        return self._agent_loop.set_round_limit_action(value)

    def request_exit(self) -> None:
        self._exit_requested = True

    def cancel_active_turn(self) -> bool:
        cancelled = self._runtime.cancel()
        if cancelled:
            discarded = self._history.discard_steering_messages()
            self._status_text = "正在取消当前任务"
            if discarded:
                self._print_info(f"已丢弃 {discarded} 条尚未注入的追加指令")
            self._invalidate_input_prompt()
        return cancelled

    def get_system_prompt(self, section: str = "all") -> str:
        return self._agent_loop.get_system_prompt(section)

    async def confirm_action(self, prompt: str) -> bool:
        self._stop_progress()
        self._print_warning(prompt)
        decision = await self._prompt_for_approval()
        return decision != HITLDecision.DENY

    def workspace_changed(self) -> None:
        from pathlib import Path

        workspace = Path.cwd().resolve()
        self._agent_loop.set_workspace(workspace)
        if self._note_manager is not None:
            self._note_manager.set_cwd(workspace)

    def update_progress(self, text: str) -> None:
        self._start_progress(text)

    # -- commands ----------------------------------------------------------

    def _build_input_keybindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-p")
        def _toggle_plan(event) -> None:
            enabled = self._agent_loop.toggle_plan_only()
            self._print_info(f"Plan-only 模式: {'ON' if enabled else 'OFF'}")

        @bindings.add("c-s")
        def _cycle_security(event) -> None:
            levels = [
                SecurityLevel.STRICT,
                SecurityLevel.NORMAL,
                SecurityLevel.PERMISSIVE,
            ]
            current = levels.index(self._security_level)
            self._security_level = levels[(current + 1) % len(levels)]
            self._agent_loop.set_security_level(self._security_level)
            labels = {
                SecurityLevel.STRICT: "严格",
                SecurityLevel.NORMAL: "默认",
                SecurityLevel.PERMISSIVE: "放行",
            }
            self._print_info(f"安全等级: {labels[self._security_level]}")

        @bindings.add("c-q")
        def _compress(event) -> None:
            if not self._runtime.active and not self._command_active:
                self._spawn(self._manual_compress())

        return bindings

    def _register_skill_command(self, meta) -> None:
        from tinyCode.commands.types import CommandMeta, CommandType

        async def _handler(args: list[str]) -> str:
            if self._skill_registry:
                skill = self._skill_registry.activate(meta.name)
                if skill:
                    return f"Skill '{meta.name}' 已激活。\n\n{skill.body[:1000]}"
            return f"Skill '{meta.name}' 激活失败"

        if self._cmd_registry.lookup(meta.name) is None:
            self._cmd_registry.register(CommandMeta(
                name=meta.name,
                description=meta.description,
                usage=f"/{meta.name}",
                cmd_type=CommandType.UI,
                handler=_handler,
            ))

    async def _handle_command(self, text: str) -> None:
        # Commands such as /compress and /team can also use the provider.
        # Avoid overlapping them with the delayed automatic-note request on
        # gateways that only permit one stream per client.
        await self._join_note_update(self._cancel_note_update())
        self._command_active = True
        self._start_progress("运行命令")
        try:
            _was_command, result = await self._cmd_dispatcher.dispatch(text)
            if result:
                # Command results, especially /prompt, can span many terminal
                # rows. Stop Rich's live status before printing so its final
                # refresh cannot erase or visually swallow repeated output.
                self._stop_progress()
                self._print_info(result)
        finally:
            self._command_active = False
            if not self._runtime.active:
                self._stop_progress()

    async def _manual_compress(self) -> None:
        await self._join_note_update(self._cancel_note_update())
        self._command_active = True
        self._start_progress("压缩上下文")
        try:
            self._print_info(await self.trigger_compress())
        except Exception as exc:
            self._print_error(f"上下文压缩失败: {type(exc).__name__}: {exc}")
        finally:
            self._command_active = False
            self._stop_progress()

    # -- message handling --------------------------------------------------

    def _start_user_input(self, text: str, *, display_user: bool = True) -> bool:
        """Reserve the foreground slot before scheduling an async turn."""
        if not self._runtime.reserve():
            self._print_warning("已有任务正在执行")
            return False
        note_task = self._cancel_note_update()
        task = self._spawn(self._on_user_input(
            text,
            display_user=display_user,
            cancelled_note_task=note_task,
        ))
        self._foreground_task = task
        task.add_done_callback(self._finish_foreground)
        return True

    def _finish_foreground(self, task: asyncio.Task) -> None:
        if self._foreground_task is task:
            self._foreground_task = None
        self._invalidate_input_prompt()

    async def _wait_for_foreground(self) -> None:
        task = self._foreground_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def _input_prompt(self) -> list[tuple[str, str]]:
        pending = self._pending_control_input
        if pending is not None and not pending.future.done():
            return pending.prompt
        if self._runtime.active:
            return [("class:warning", "↪ 追加指令（/cancel 取消）› ")]
        return [("class:prompt", "› ")]

    def _invalidate_input_prompt(self) -> None:
        app = getattr(self._prompt_session, "app", None)
        if app is not None and getattr(app, "is_running", False):
            try:
                app.invalidate()
            except RuntimeError:
                pass

    async def _read_control_input(
        self, prompt: list[tuple[str, str]],
    ) -> str:
        """Read a foreground decision through the single TUI input owner."""
        if not self._runtime.active or not self._input_loop_active:
            return await self._prompt_session.prompt_async(prompt)
        if self._pending_control_input is not None:
            raise RuntimeError("已有交互输入正在等待用户回答")
        future = asyncio.get_running_loop().create_future()
        request = _PendingControlInput(prompt=prompt, future=future)
        self._pending_control_input = request
        self._invalidate_input_prompt()
        try:
            return await future
        finally:
            if self._pending_control_input is request:
                self._pending_control_input = None
            if not future.done():
                future.cancel()
            self._invalidate_input_prompt()

    @staticmethod
    def _is_cancel_command(text: str) -> bool:
        return text.strip().lower() == "/cancel"

    async def _handle_active_input(self, text: str) -> None:
        if self._cmd_dispatcher.is_command(text):
            self._print_warning(
                "任务执行中仅支持 /cancel；普通文字会作为追加指令排队"
            )
            return
        self._history.queue_steering_message(text)
        count = getattr(self._history, "steering_count", 1)
        self._print_info(
            f"↪ 已排队追加指令（当前等待 {count} 条），"
            "将在本轮模型/工具步骤完成后注入"
        )

    async def _handle_cancel_input(self, text: str) -> None:
        _was_command, result = await self._cmd_dispatcher.dispatch(text)
        if result:
            self._print_info(result)
        await self._wait_for_foreground()

    async def _on_user_input(
        self,
        text: str,
        *,
        display_user: bool = True,
        cancelled_note_task: asyncio.Task | None = None,
    ) -> None:
        if not self._runtime.active and not self._runtime.reserve():
            return
        if not self._runtime.claim():
            return
        metrics = TurnMetrics()
        workspace_snapshot: WorkspaceSnapshot | None = None
        response_started = False
        stream_line_open = False
        current_response = ""
        round_recorded = False
        stream_normalizer = _StreamTextNormalizer()
        try:
            note_task = cancelled_note_task or self._cancel_note_update()
            await self._join_note_update(note_task)
            if display_user:
                self._print_user(text)
            self._history.flush_deferred()
            self._history.add_user_message(text)
            self._save_checkpoint()
            self._start_progress("准备本轮任务")

            async for event in self._runtime.run(self._history):
                if isinstance(event, RoundStartEvent):
                    metrics.record_round(event.round_number)
                    self._start_progress(
                        f"第 {event.round_number}/{event.max_rounds} 轮 · 等待模型"
                    )

                elif isinstance(event, TextDeltaEvent):
                    # A spinner is useful while waiting, but must never split the
                    # response itself. Stop it before writing the first byte.
                    self._stop_progress()
                    if not response_started:
                        response_started = True
                        self._print_ai_prefix()
                    normalized = stream_normalizer.normalize(event.text)
                    current_response += normalized
                    self._print_stream(normalized)
                    if normalized:
                        stream_line_open = not normalized.endswith("\n")

                elif isinstance(event, ThinkingEvent):
                    self._start_progress(event.label)

                elif isinstance(event, ToolCallEvent):
                    if workspace_snapshot is None:
                        workspace_snapshot = await asyncio.to_thread(
                            WorkspaceSnapshot.capture,
                            Path.cwd(),
                        )
                    metrics.record_tool_call()
                    stream_line_open = self._close_stream_line(stream_line_open)
                    self._start_progress(f"执行工具 {event.tool_call.name}")

                elif isinstance(event, ToolResultEvent):
                    metrics.record_tool_result(event.result.success)
                    self._save_checkpoint()
                    state = "已返回" if event.result.success else "失败，交给模型处理"
                    self._start_progress(f"工具 {event.tool_name} {state}")

                elif isinstance(event, ToolBlockedEvent):
                    self._save_checkpoint()
                    self._start_progress(f"工具 {event.tool_name} 已被拦截")

                elif isinstance(event, TruncationEvent):
                    self._start_progress(
                        f"已压缩 {event.tool_name} 结果（{event.original_chars:,} 字符）"
                    )

                elif isinstance(event, ContextCompressionEvent):
                    if event.warning_issued:
                        self._print_warning(
                            "上下文接近窗口上限，正在执行任务内压缩"
                        )
                    if event.was_compressed:
                        released = max(
                            0,
                            event.estimated_tokens_before - event.estimated_tokens_after,
                        )
                        self._print_info(
                            "上下文已压缩 · "
                            f"≈{self._format_compact_tokens(event.estimated_tokens_before)} → "
                            f"≈{self._format_compact_tokens(event.estimated_tokens_after)} · "
                            f"释放 ≈{self._format_compact_tokens(released)}"
                        )
                    elif event.error:
                        self._print_warning(f"上下文压缩未完成：{event.error}")

                elif isinstance(event, HITLRequestEvent):
                    stream_line_open = self._close_stream_line(stream_line_open)
                    self._stop_progress()
                    self._print_warning(event.prompt)
                    decision = await self._prompt_for_approval()
                    self._resolve_hitl(decision)
                    self._start_progress("已确认 · 继续执行")

                elif isinstance(event, RoundLimitReachedEvent):
                    stream_line_open = self._close_stream_line(stream_line_open)
                    self._stop_progress()
                    if event.stalled:
                        self._print_warning(
                            "最近 3 轮重复了相同工具调用，已暂停自动续跑"
                        )
                    self._print_warning(
                        f"任务尚未完成，已用完轮次预算 "
                        f"{event.round_number}/{event.current_limit}"
                        f"（硬上限 {event.hard_limit}）"
                    )
                    decision = await self._prompt_for_round_limit(event)
                    self._runtime.resolve_round_limit(decision)
                    if decision.action == RoundLimitDecisionAction.STOP:
                        self._start_progress("正在暂停任务")
                    else:
                        self._start_progress("轮次预算已确认 · 继续执行")

                elif isinstance(event, RoundLimitExtendedEvent):
                    mode = "自动续跑" if event.automatic else "本次续跑"
                    self._print_info(
                        f"轮次预算已扩展：{event.previous_limit} → "
                        f"{event.new_limit}（{mode} · 硬上限 {event.hard_limit}）"
                    )
                    self._start_progress(
                        f"预算 {event.new_limit} 轮 · 继续执行"
                    )

                elif isinstance(event, ProgressWarningEvent):
                    reason = "；".join(event.reasons)
                    labels = {
                        "slow": "任务进展变慢",
                        "stalled": "任务没有有效进展",
                        "oscillating": "任务方案往返震荡",
                        "hard_stuck": "任务重复错误或超时",
                    }
                    self._print_warning(
                        f"{labels.get(event.state, '任务进展异常')}：{reason}"
                    )
                    self._start_progress("已要求模型更换重复步骤")

                elif isinstance(event, SteeringAppliedEvent):
                    stream_line_open = self._close_stream_line(stream_line_open)
                    self._save_checkpoint()
                    if event.continued:
                        self._print_info(
                            f"↪ 已注入 {event.message_count} 条追加指令，继续当前任务"
                        )
                        self._start_progress("已接收追加指令 · 等待模型")
                    else:
                        self._print_warning(
                            f"已保存 {event.message_count} 条追加指令，"
                            "但任务已达到轮次硬上限"
                        )

                elif isinstance(event, TaskStalledEvent):
                    stream_line_open = self._close_stream_line(stream_line_open)
                    self._stop_progress()
                    labels = {
                        "stalled": "没有有效进展",
                        "oscillating": "方案往返震荡",
                        "hard_stuck": "重复错误或超时",
                    }
                    self._print_warning(
                        f"检测到任务{labels.get(event.state, '进展异常')}"
                    )
                    for reason in event.reasons:
                        self._print_info(f"  - {reason}")
                    decision = await self._prompt_for_stalled_task(event)
                    self._runtime.resolve_progress(decision)
                    if decision.action == TaskStalledDecisionAction.STOP:
                        self._start_progress("正在暂停任务")
                    elif decision.action == TaskStalledDecisionAction.STRATEGY:
                        self._start_progress("已要求更换策略 · 继续执行")
                    else:
                        self._start_progress("已确认继续观察")

                elif isinstance(event, AgentDoneEvent):
                    self._stop_progress()
                    stream_line_open = self._close_stream_line(stream_line_open)
                    await self._print_workspace_changes(workspace_snapshot)
                    if event.reason in {"max_rounds", "round_budget_stopped"}:
                        self._status_text = "就绪 · 任务因轮次预算暂停"
                        self._print_warning(
                            "任务尚未完成，已暂停并保留当前进度；可输入“继续”恢复"
                        )
                    elif event.reason == "hard_max_rounds":
                        self._status_text = "就绪 · 任务达到轮次硬上限"
                        self._print_error(
                            "任务尚未完成，已达到轮次硬上限；"
                            "当前进度已保留，可调整配置后输入“继续”"
                        )
                    elif event.reason == "stalled":
                        self._status_text = "就绪 · 任务因无进展暂停"
                        self._print_warning(
                            "任务因持续无进展已暂停，当前进度已保留；"
                            "可补充信息或输入“继续”"
                        )
                    elif event.reason == "cancelled":
                        self._status_text = "就绪 · 本轮已取消"
                        self._print_info("本轮已取消")
                    else:
                        self._status_text = "就绪 · 上一轮已正常完成"
                        cache = " · cache ✓" if self._agent_loop.cache_hit else ""
                        self._console.print(
                            f"✓ 本轮已正常完成{cache}", style="bold green", highlight=False
                        )
                    self._print_turn_metrics(
                        metrics=metrics,
                        model_requests=getattr(
                            self._agent_loop, "turn_model_requests", 0
                        ),
                    )
                    self._print_context_snapshot()
                    if current_response:
                        self._agent_loop.record_round(text, current_response)
                        round_recorded = True
                    if round_recorded:
                        self._schedule_note_update()
                    break

                elif isinstance(event, ErrorEvent):
                    self._stop_progress()
                    stream_line_open = self._close_stream_line(stream_line_open)
                    await self._print_workspace_changes(workspace_snapshot)
                    self._status_text = "就绪 · 上一轮失败"
                    self._print_error(event.message)
                    self._print_turn_metrics(
                        metrics=metrics,
                        model_requests=getattr(
                            self._agent_loop, "turn_model_requests", 0
                        ),
                    )
                    self._print_context_snapshot()
                    break

        except asyncio.CancelledError:
            self._stop_progress()
            self._status_text = "就绪 · 本轮已取消"
            await self._print_workspace_changes(workspace_snapshot)
            raise
        except Exception as exc:
            self._runtime.fail_preparation(str(exc))
            self._stop_progress()
            stream_line_open = self._close_stream_line(stream_line_open)
            await self._print_workspace_changes(workspace_snapshot)
            self._status_text = "就绪 · 上一轮失败"
            self._print_error(f"对话执行失败: {type(exc).__name__}: {exc}")
        finally:
            self._runtime.release()
            try:
                self._do_save()
            except Exception as exc:
                self._print_warning(f"会话保存失败: {type(exc).__name__}: {exc}")

    async def _prompt_for_approval(self) -> HITLDecision:
        choices = {
            "a": HITLDecision.ALLOW_ONCE,
            "s": HITLDecision.ALLOW_SESSION,
            "p": HITLDecision.ALLOW_PERMANENT,
            "d": HITLDecision.DENY,
        }
        while True:
            try:
                answer = await self._read_control_input(
                    [("class:warning", "确认 [A本次/S会话/P永久/D拒绝] › ")]
                )
            except (EOFError, KeyboardInterrupt):
                return HITLDecision.DENY
            decision = choices.get(answer.strip().lower())
            if decision is not None:
                return decision
            self._print_warning("请输入 A、S、P 或 D")

    async def _prompt_for_round_limit(
        self, event: RoundLimitReachedEvent,
    ) -> RoundLimitDecision:
        prompt = (
            f"轮次 [A再执行{event.extension}轮/C持续执行/S停止，"
            "也可输入 +N 或目标轮数] › "
        )
        while True:
            try:
                answer = (
                    await self._read_control_input(
                        [("class:warning", prompt)]
                    )
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return RoundLimitDecision(RoundLimitDecisionAction.STOP)

            if answer == "a":
                return RoundLimitDecision(
                    RoundLimitDecisionAction.EXTEND,
                    min(event.hard_limit, event.current_limit + event.extension),
                )
            if answer == "c":
                return RoundLimitDecision(RoundLimitDecisionAction.AUTO)
            if answer == "s":
                return RoundLimitDecision(RoundLimitDecisionAction.STOP)

            value_text = answer
            update_session_default = False
            for prefix in ("/rounds ", "/config max-rounds ", "/config max_rounds "):
                if value_text.startswith(prefix):
                    value_text = value_text[len(prefix):].strip()
                    update_session_default = prefix.startswith("/config")
                    break
            try:
                if value_text.startswith("+"):
                    target = event.current_limit + int(value_text[1:])
                else:
                    target = int(value_text)
            except ValueError:
                self._print_warning("请输入 A、C、S、+N 或目标轮数")
                continue

            if not event.current_limit < target <= event.hard_limit:
                self._print_warning(
                    f"目标轮数必须大于 {event.current_limit} 且不超过 "
                    f"{event.hard_limit}"
                )
                continue
            if update_session_default:
                try:
                    self.set_max_rounds(target)
                except ValueError as exc:
                    self._print_warning(str(exc))
                    continue
            return RoundLimitDecision(
                RoundLimitDecisionAction.EXTEND,
                target,
            )

    async def _prompt_for_stalled_task(
        self, event: TaskStalledEvent,
    ) -> TaskStalledDecision:
        prompt = (
            f"停滞 [R更换策略/C继续{event.continue_rounds}轮/S停止，"
            "也可输入 +N] › "
        )
        while True:
            try:
                answer = (
                    await self._read_control_input(
                        [("class:warning", prompt)]
                    )
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return TaskStalledDecision(TaskStalledDecisionAction.STOP)

            if answer == "r":
                return TaskStalledDecision(TaskStalledDecisionAction.STRATEGY)
            if answer == "c":
                return TaskStalledDecision(
                    TaskStalledDecisionAction.CONTINUE,
                    event.continue_rounds,
                )
            if answer == "s":
                return TaskStalledDecision(TaskStalledDecisionAction.STOP)

            value_text = answer
            if value_text.startswith("/rounds "):
                value_text = value_text[len("/rounds "):].strip()
            if value_text.startswith("+"):
                value_text = value_text[1:]
            try:
                rounds = int(value_text)
            except ValueError:
                self._print_warning("请输入 R、C、S 或 +N")
                continue
            remaining = event.hard_limit - event.round_number
            if not 1 <= rounds <= remaining:
                self._print_warning(f"继续轮数必须是 1 到 {remaining}")
                continue
            return TaskStalledDecision(
                TaskStalledDecisionAction.CONTINUE,
                rounds,
            )

    def _resolve_hitl(self, decision: HITLDecision) -> None:
        if self._runtime.resolve_approval(decision):
            labels = {
                HITLDecision.ALLOW_ONCE: "允许（本次）",
                HITLDecision.ALLOW_SESSION: "允许（本会话）",
                HITLDecision.ALLOW_PERMANENT: "允许（永久）",
                HITLDecision.DENY: "拒绝",
            }
            self._print_info(f"安全确认：{labels[decision]}")

    # -- output ------------------------------------------------------------

    def _print_user(self, text: str) -> None:
        self._console.print()
        line = Text("> ", style="bold green")
        line.append(text)
        self._console.print(line, highlight=False)

    def _print_ai_prefix(self) -> None:
        self._console.print()
        self._console.print("TinyCode: ", style="bold blue", end="", highlight=False)

    def _print_stream(self, text: str) -> None:
        if text:
            self._console.print(
                text, end="", markup=False, highlight=False, soft_wrap=True
            )

    def _print_info(self, text: str) -> None:
        self._console.print(text, style="dim", markup=False, highlight=False)

    def _print_warning(self, text: str) -> None:
        self._console.print(f"⚠ {text}", style="bold yellow", markup=False, highlight=False)

    def _print_error(self, text: str) -> None:
        self._console.print(f"Error: {text}", style="bold red", markup=False, highlight=False)

    def _print_turn_metrics(
        self,
        *,
        metrics: TurnMetrics,
        model_requests: int,
    ) -> None:
        usage = getattr(self._agent_loop, "turn_usage", None)
        if usage is not None and getattr(usage, "available", False):
            token_text = f"{getattr(usage, 'total_tokens', 0):,}"
        else:
            token_text = "不可用"

        self._console.print(
            "本轮统计 · "
            f"Turn: {metrics.turns} · "
            f"模型请求: {max(0, model_requests)} 次 · "
            f"消耗 Token: {token_text} · "
            f"耗时: {metrics.elapsed_seconds:.2f} 秒 · "
            f"工具调用: {metrics.tool_calls} 次 · "
            f"成功率: {metrics.success_rate_text}",
            style="dim",
            highlight=False,
        )

    async def _print_workspace_changes(
        self,
        snapshot: WorkspaceSnapshot | None,
    ) -> None:
        if snapshot is None:
            return
        changes = await asyncio.to_thread(snapshot.compare)
        if not changes.any:
            return
        self._render_workspace_changes(changes)

    def _render_workspace_changes(self, changes: WorkspaceChanges) -> None:
        counts: list[str] = []
        if changes.added:
            counts.append(f"新增 {len(changes.added)}")
        if changes.modified:
            counts.append(f"修改 {len(changes.modified)}")
        if changes.deleted:
            counts.append(f"删除 {len(changes.deleted)}")
        self._console.print()
        self._console.print(
            "文件变更 · " + " · ".join(counts),
            style="dim",
            highlight=False,
        )
        for paths, marker, style in (
            (changes.added, "+", "green"),
            (changes.modified, "~", "yellow"),
            (changes.deleted, "-", "red"),
        ):
            for path in paths:
                self._console.print(
                    f"  {marker} {path}",
                    style=style,
                    markup=False,
                    highlight=False,
                )
        self._console.print()

    def _print_context_snapshot(self) -> None:
        # The last provider request already includes system instructions, tool
        # schemas and conversation history, so its real usage is the best
        # snapshot. Retained history can be larger after an intentionally
        # isolated direct-answer turn, so keep the conservative maximum. Fall
        # back to the history estimator when a service omits streaming usage.
        last_usage = TokenUsage.from_raw(
            getattr(self._agent_loop.provider, "last_usage", None)
        )
        history_estimate = self._history.estimated_token_count()
        used = (
            max(last_usage.total_tokens, history_estimate)
            if last_usage.available
            else history_estimate
        )
        used = max(0, used)
        window = max(1, int(self._compressor.context_window))
        remaining = max(0, window - used)
        percentage = used / window * 100
        if percentage >= 90:
            style = "bold red"
        elif percentage >= 70:
            style = "bold yellow"
        else:
            style = "dim"

        self._console.print(
            "上下文   · "
            f"≈{self._format_compact_tokens(used)} / "
            f"{self._format_compact_tokens(window)}（{percentage:.0f}%）· "
            f"剩余 ≈{self._format_compact_tokens(remaining)}",
            style=style,
            highlight=False,
        )

    @staticmethod
    def _format_compact_tokens(value: int) -> str:
        value = max(0, int(value))
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}m"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

    def _close_stream_line(self, line_open: bool) -> bool:
        if line_open:
            self._console.print()
        return False

    def _start_progress(self, text: str) -> None:
        self._progress_text = text
        self._status_text = f"运行中 · {text}"
        if not self._console.is_terminal:
            return
        if self._progress is None:
            self._progress = self._console.status(
                text, spinner="dots", spinner_style="cyan", refresh_per_second=10
            )
            self._progress.start()
        else:
            self._progress.update(text)

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self._progress_text = None

    # -- background lifecycle ---------------------------------------------

    async def _maybe_update_notes(self) -> None:
        try:
            await asyncio.sleep(1.0)
            count = await self._agent_loop.update_notes_if_needed()
            if count > 0:
                requests = getattr(self._note_manager, "last_update_model_requests", 0)
                tokens = getattr(self._note_manager, "last_update_tokens", 0)
                self._print_info(
                    f"📝 已更新 {count} 个笔记文件 · 模型请求 {requests} 次"
                    + (f" · Token {tokens:,}" if tokens else "")
                )
            errors = getattr(self._note_manager, "last_errors", [])
            if errors:
                self._print_warning("部分自动笔记更新失败: " + "; ".join(errors))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._print_warning(f"自动笔记更新失败: {type(exc).__name__}: {exc}")

    def _schedule_note_update(self) -> None:
        if self._note_manager is None:
            return
        should_update = getattr(self._note_manager, "should_update", None)
        if callable(should_update) and not should_update():
            return
        if self._note_update_task and not self._note_update_task.done():
            return
        task = self._spawn(self._maybe_update_notes())
        self._note_update_task = task
        task.add_done_callback(self._finish_note_update)

    def _finish_note_update(self, task: asyncio.Task) -> None:
        if self._note_update_task is task:
            self._note_update_task = None

    def _cancel_note_update(self) -> asyncio.Task | None:
        task = self._note_update_task
        if task and not task.done():
            task.cancel()
        self._note_update_task = None
        return task

    @staticmethod
    async def _join_note_update(task: asyncio.Task | None) -> None:
        """Wait until a cancelled note stream has released provider resources."""
        if task is None or task is asyncio.current_task():
            return
        await asyncio.gather(task, return_exceptions=True)

    def _spawn(self, awaitable) -> asyncio.Task:
        task = asyncio.ensure_future(awaitable)
        self._background_tasks.add(task)
        task.add_done_callback(self._finish_background_task)
        return task

    def _finish_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._print_warning(f"后台任务失败: {type(exc).__name__}: {exc}")

    async def _exit_notes(self) -> None:
        if self._note_manager:
            await self._note_manager.update_on_exit()

    async def shutdown(self) -> None:
        """Cancel and join UI-owned tasks before the event loop closes."""
        if self._closed:
            return
        self._closed = True
        self._runtime.cancel()
        self._stop_progress()
        self._cancel_note_update()
        current = asyncio.current_task()
        tasks = [task for task in self._background_tasks if task is not current]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._start_progress("退出前保存笔记")
        try:
            await asyncio.wait_for(self._exit_notes(), timeout=30.0)
        except asyncio.TimeoutError:
            self._print_warning("退出时笔记更新超时，已停止等待")
        except Exception as exc:
            self._print_warning(f"退出时笔记更新失败: {type(exc).__name__}: {exc}")
        finally:
            self._stop_progress()

    def _do_save(self) -> None:
        self._session_store.save(
            self._history, provider_name=self._provider_name, model=self._model
        )

    def _save_checkpoint(self) -> None:
        try:
            self._do_save()
        except Exception as exc:
            self._print_warning(f"会话检查点保存失败: {type(exc).__name__}: {exc}")

    # -- run ---------------------------------------------------------------

    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        mcp = f" · MCP {self._mcp_server_count}" if self._mcp_server_count else ""
        self._console.print(
            f"TinyCode · {self._provider_name}/{self._model}{mcp} · 就绪",
            style="dim",
            highlight=False,
        )
        output_context = patch_stdout(raw=True) if self._uses_prompt_toolkit else nullcontext()
        self._input_loop_active = True
        with output_context:
            while not self._exit_requested:
                try:
                    text = (
                        await self._prompt_session.prompt_async(
                            self._input_prompt
                        )
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    self._runtime.cancel()
                    break
                if not text:
                    continue

                if self._runtime.active and self._is_cancel_command(text):
                    await self._handle_cancel_input(text)
                elif (
                    self._pending_control_input is not None
                    and not self._pending_control_input.future.done()
                ):
                    self._pending_control_input.future.set_result(text)
                elif self._runtime.active:
                    await self._handle_active_input(text)
                elif self._cmd_dispatcher.is_command(text):
                    await self._handle_command(text)
                else:
                    # Both prompt_toolkit and plain input already echo submitted
                    # input. Do not print the same user message a second time.
                    self._start_user_input(text, display_user=False)
                # Let the foreground task publish approval/stall input requests
                # before the next prompt is rendered. Real terminals naturally
                # yield here; this also makes scripted input deterministic.
                await asyncio.sleep(0)
        self._input_loop_active = False
        if self._runtime.active:
            self._runtime.cancel()
        await self._wait_for_foreground()
        self._stop_progress()
        try:
            self._do_save()
        except Exception as exc:
            self._print_warning(f"退出前会话保存失败: {type(exc).__name__}: {exc}")
        # Finish optional note persistence before announcing that the process
        # has safely exited. CleanupStack calls shutdown() again, which is
        # intentionally idempotent.
        await self.shutdown()
        self._print_info("Goodbye!")

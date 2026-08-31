"""Component-based fullscreen chat UI for TinyCode.

The agent runtime remains owned by :class:`TinyCodeTUI`. Textual only owns
presentation: durable turn widgets, Markdown, process disclosure, scrolling,
and the fixed composer.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from rich.console import Console
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible, Input, Markdown, Static

from tinyCode.tui.app import TinyCodeTUI
from tinyCode.tui.metrics import TurnMetrics
from tinyCode.tui.workspace_changes import WorkspaceChanges


class _FullscreenStreamSink:
    """Forward model deltas to the active answer widget."""

    def __init__(self, append: Callable[[str], None]) -> None:
        self._append = append
        self._line_open = False

    def write(self, text: str) -> bool:
        if text:
            self._append(text)
            self._line_open = not text.endswith("\n")
        return self._line_open

    def close_line(self) -> bool:
        if self._line_open:
            self._append("\n")
        self._line_open = False
        return False


@dataclass
class _SystemNotice:
    kind: str
    title: str
    text: str


@dataclass
class _ConversationTurn:
    user_text: str
    process_lines: list[str] = field(default_factory=list)
    answer: str = ""
    workspace_summary: str = ""
    metrics_summary: str = ""
    process_collapsed: bool = False
    finished: bool = False
    answer_is_markdown: bool = True
    answer_kind: str = "assistant"
    notices: list[_SystemNotice] = field(default_factory=list)


class _TurnView(Vertical):
    """A single user request and its matching agent response."""

    def __init__(self, turn: _ConversationTurn) -> None:
        super().__init__(classes="turn")
        self.turn = turn
        self.user = Static(turn.user_text, classes="user-bubble", markup=False)
        self.user_row = Horizontal(self.user, classes="user-row")
        self.process_text = Static("", classes="process-text", markup=False)
        self.process = Collapsible(
            self.process_text,
            title="执行过程",
            collapsed=turn.process_collapsed,
            classes="process-block",
        )
        self.agent_label = Static("TinyCode", classes="agent-label")
        self.answer = Markdown(turn.answer, classes="agent-answer")
        self.plain_answer = Static(
            turn.answer, classes="agent-plain-answer", markup=False,
        )
        self.plain_answer.add_class(f"{turn.answer_kind}-answer")
        notice_widgets = [self._notice_widget(notice) for notice in turn.notices]
        self.notice_list = Vertical(*notice_widgets, classes="notice-list")
        self._rendered_notice_count = len(notice_widgets)
        self.workspace = Static("", classes="workspace-card", markup=False)
        self.metrics = Static("", classes="turn-metrics", markup=False)

    def compose(self) -> ComposeResult:
        yield self.user_row
        yield self.process
        yield self.notice_list
        yield self.agent_label
        yield self.answer
        yield self.plain_answer
        yield self.workspace
        yield self.metrics

    def on_mount(self) -> None:
        self.sync()

    def sync(self) -> None:
        turn = self.turn
        self.user_row.display = bool(turn.user_text)
        self.process_text.update("\n\n".join(turn.process_lines))
        self.process.title = f"执行过程 · {len(turn.process_lines)} 个事件"
        self.process.collapsed = turn.process_collapsed
        self.process.display = bool(turn.process_lines)
        self.agent_label.display = bool(turn.answer)
        self.answer.display = bool(turn.answer) and turn.answer_is_markdown
        self.plain_answer.display = bool(turn.answer) and not turn.answer_is_markdown
        if turn.answer and turn.answer_is_markdown:
            self.answer.update(turn.answer)
        elif turn.answer:
            self.plain_answer.update(turn.answer)
        if self.notice_list.is_mounted:
            for notice in turn.notices[self._rendered_notice_count:]:
                self.notice_list.mount(self._notice_widget(notice))
                self._rendered_notice_count += 1
        self.workspace.display = bool(turn.workspace_summary)
        workspace = Text()
        for index, line in enumerate(turn.workspace_summary.splitlines()):
            if index:
                workspace.append("\n")
            style = ""
            if line.startswith("+ "):
                style = "green"
            elif line.startswith("- "):
                style = "red"
            elif line.startswith("~ "):
                style = "yellow"
            elif index == 0:
                style = "bold"
            workspace.append(line, style=style)
        self.workspace.update(workspace)
        self.metrics.display = bool(turn.metrics_summary)
        self.metrics.update(turn.metrics_summary)

    @staticmethod
    def _notice_widget(notice: _SystemNotice) -> Static:
        content = Text()
        content.append(notice.title, style="bold")
        content.append("\n" + notice.text)
        return Static(
            content,
            classes=f"system-card {notice.kind}-card",
            markup=False,
        )


class _ChatScroll(VerticalScroll):
    """Scrollable history that reports whether the user is at the live tail."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.call_after_refresh(self._sync_follow_tail)

    def _sync_follow_tail(self) -> None:
        app = self.app
        if isinstance(app, _TinyCodeFullscreenApp):
            app.follow_tail = self.max_scroll_y <= 0 or self.scroll_y >= self.max_scroll_y - 1
            if app.follow_tail:
                app._hide_new_output()


class _TinyCodeFullscreenApp(App[None]):
    """Textual shell; all task decisions stay in the owner TUI."""

    TITLE = "TinyCode"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "cancel_or_exit", "取消/退出", priority=True),
        Binding("ctrl+e", "toggle_process", "执行过程", priority=True),
        Binding("pageup", "history_up", "历史上翻", priority=True),
        Binding("pagedown", "history_down", "回到最新", priority=True),
        Binding("tab", "complete_command", show=False, priority=True),
        Binding("up", "previous_command", show=False, priority=True),
        Binding("down", "next_command", show=False, priority=True),
        Binding("escape", "dismiss_commands", show=False, priority=True),
    ]

    CSS = """
    Screen { layout: vertical; background: $background; }
    #topbar {
        height: 1; padding: 0 1; color: $text-muted; background: $surface;
    }
    #chat-scroll {
        height: 1fr; padding: 1 2; scrollbar-size-vertical: 1;
        scrollbar-color: $primary 30%; scrollbar-background: $surface;
    }
    #turn-list { height: auto; }
    .turn { height: auto; margin: 0 0 1 0; }
    .user-row { height: auto; align-horizontal: right; margin: 0 0 1 0; }
    .user-bubble {
        width: auto; max-width: 70%; height: auto; padding: 1 2;
        color: $text; background: $boost; border: round $panel-lighten-2;
    }
    .process-block {
        width: 100%; height: auto; margin: 0 0 1 0; color: $text-muted;
        border: none; background: transparent;
    }
    .process-text {
        width: 100%; height: auto; padding: 0 1;
        color: $text-muted; background: $surface;
    }
    .agent-label {
        width: auto; height: 1; margin: 0 0 1 0;
        color: $primary; text-style: bold;
    }
    .agent-answer {
        width: 100%; height: auto; margin: 0 0 1 0;
        color: $text; background: transparent;
    }
    .agent-plain-answer {
        width: 100%; height: auto; margin: 0 0 1 0;
        color: $text; background: transparent;
    }
    .command-answer { color: $text-muted; }
    .warning-answer { color: $warning; }
    .error-answer { color: $error; }
    .notice-list { width: 100%; height: auto; }
    .system-card {
        width: 100%; height: auto; margin: 0 0 1 0; padding: 0 1;
        background: $surface;
    }
    .warning-card { color: $warning; border-left: thick $warning; }
    .error-card { color: $error; border-left: thick $error; }
    .approval-card { color: $warning; border-left: thick $warning; }
    .agent-answer MarkdownH1, .agent-answer MarkdownH2, .agent-answer MarkdownH3 {
        width: 100%; height: auto; padding: 0; margin: 1 0;
        color: $text; background: transparent; text-align: left;
        text-style: bold; border: none;
    }
    .agent-answer MarkdownFence {
        width: 100%; background: $surface; border: none;
    }
    .workspace-card {
        width: 100%; height: auto; margin: 1 0 0 0; padding: 1 2;
        color: $success; background: $surface; border: round $panel-lighten-2;
    }
    .turn-metrics {
        width: 100%; height: auto; margin: 1 0 0 0; color: $text-muted;
    }
    #composer-shell {
        height: auto; min-height: 3; padding: 0 1 1 1; background: $background;
    }
    #command-menu {
        display: none; width: 100%; height: auto; max-height: 10;
        margin: 0 0 1 0; padding: 0 1;
        color: $text-muted; background: $surface;
        border: round $panel-lighten-2;
    }
    #new-output {
        display: none; width: 100%; height: 1; margin: 0 0 1 0;
        color: $primary; text-align: center; text-style: bold;
        background: $surface;
    }
    #composer { width: 100%; border: round $panel-lighten-2; }
    #helpbar { height: 1; padding: 0 2; color: $text-disabled; }
    """

    def __init__(self, owner: FullscreenTinyCodeTUI) -> None:
        super().__init__()
        self.owner = owner
        self.follow_tail = True
        self.command_candidates: list[str] = []
        self.command_index = 0

    def compose(self) -> ComposeResult:
        yield Static(self.owner._header_text(), id="topbar", markup=False)
        with _ChatScroll(id="chat-scroll"):
            yield Vertical(id="turn-list")
        with Vertical(id="composer-shell"):
            yield Static("", id="command-menu", markup=False)
            yield Static(
                "↓ 有新的模型输出 · PgDn 回到底部",
                id="new-output",
                markup=False,
            )
            yield Input(placeholder=self.owner._input_placeholder(), id="composer")
        yield Static(
            "PgUp/PgDn 历史 · Ctrl-E 过程 · Ctrl-C 取消/退出",
            id="helpbar",
            markup=False,
        )

    def on_mount(self) -> None:
        self.owner._textual_ready(self)
        self.query_one("#composer", Input).focus()

    @on(Input.Submitted, "#composer")
    def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        self._hide_command_menu()
        if text:
            self.run_worker(self.owner._submit_input(text), exclusive=False)

    @on(Input.Changed, "#composer")
    def _on_input_changed(self, event: Input.Changed) -> None:
        self._update_command_menu(event.value)

    @on(events.Click, "#new-output")
    def _on_new_output_clicked(self) -> None:
        self.action_history_down()

    def action_cancel_or_exit(self) -> None:
        if self.owner._runtime.active:
            self.owner.cancel_active_turn()
        else:
            self.exit()

    def action_toggle_process(self) -> None:
        self.owner._toggle_latest_process()

    def action_history_up(self) -> None:
        self.follow_tail = False
        self.query_one("#chat-scroll", VerticalScroll).scroll_page_up(animate=False)

    def action_history_down(self) -> None:
        self.follow_tail = True
        self._hide_new_output()
        self.scroll_to_latest(force=True)

    def action_complete_command(self) -> None:
        if not self.command_candidates:
            return
        command = self.command_candidates[self.command_index]
        meta = self.owner._cmd_registry.lookup(command.removeprefix("/"))
        composer = self.query_one("#composer", Input)
        composer.value = command + (" " if meta is not None and meta.params else "")
        composer.cursor_position = len(composer.value)
        self._hide_command_menu()

    def action_previous_command(self) -> None:
        if self.command_candidates:
            self.command_index = (self.command_index - 1) % len(self.command_candidates)
            self._render_command_menu()

    def action_next_command(self) -> None:
        if self.command_candidates:
            self.command_index = (self.command_index + 1) % len(self.command_candidates)
            self._render_command_menu()

    def action_dismiss_commands(self) -> None:
        self._hide_command_menu()

    def scroll_to_latest(self, *, force: bool = False) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        if not self.follow_tail and not force:
            self._show_new_output()
            return
        self._hide_new_output()
        self.call_after_refresh(
            scroll.scroll_end, animate=False, immediate=True,
        )

    def refresh_header(self) -> None:
        self.query_one("#topbar", Static).update(self.owner._header_text())

    def refresh_composer(self) -> None:
        self.query_one("#composer", Input).placeholder = self.owner._input_placeholder()

    def _update_command_menu(self, value: str) -> None:
        if not value.startswith("/") or any(char.isspace() for char in value):
            self._hide_command_menu()
            return
        prefix = value[1:]
        candidates = self.owner._cmd_registry.get_completions(prefix)
        if self.owner._runtime.active:
            candidates = [item for item in candidates if item == "/cancel"]
        self.command_candidates = candidates
        self.command_index = min(self.command_index, max(0, len(candidates) - 1))
        self._render_command_menu()

    def _render_command_menu(self) -> None:
        menu = self.query_one("#command-menu", Static)
        if not self.command_candidates:
            menu.display = False
            return
        window_size = 7
        start = max(0, min(
            self.command_index - window_size // 2,
            len(self.command_candidates) - window_size,
        ))
        visible = self.command_candidates[start:start + window_size]
        content = Text()
        for offset, command in enumerate(visible):
            index = start + offset
            selected = index == self.command_index
            meta = self.owner._cmd_registry.lookup(command.removeprefix("/"))
            description = meta.description if meta is not None else ""
            marker = "›" if selected else " "
            style = "bold reverse" if selected else ""
            content.append(f"{marker} {command:<18}", style=style)
            content.append(description, style="dim")
            if offset < len(visible) - 1:
                content.append("\n")
        menu.update(content)
        menu.display = True

    def _hide_command_menu(self) -> None:
        self.command_candidates = []
        self.command_index = 0
        if self.is_mounted:
            self.query_one("#command-menu", Static).display = False

    def _show_new_output(self) -> None:
        self.query_one("#new-output", Static).display = True

    def _hide_new_output(self) -> None:
        self.query_one("#new-output", Static).display = False


class FullscreenTinyCodeTUI(TinyCodeTUI):
    """Codex-like fullscreen conversation UI backed by Textual widgets."""

    _MAX_PROCESS_CHARS = 120_000

    @classmethod
    def supported(cls) -> bool:
        return bool(
            getattr(sys.stdin, "isatty", lambda: False)()
            and getattr(sys.stdout, "isatty", lambda: False)()
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("application_input", None)
        kwargs.pop("application_output", None)
        kwargs["console"] = Console(file=io.StringIO(), force_terminal=False)
        super().__init__(*args, **kwargs)
        self._turns: list[_ConversationTurn] = []
        self._active_turn: _ConversationTurn | None = None
        self._active_view: _TurnView | None = None
        self._app: _TinyCodeFullscreenApp | None = None
        self._process_lines: list[str] = []
        self._assistant_draft = ""
        self._process_collapsed = False
        self._metric_summary = ""
        self._workspace_summary = ""
        self._last_process_progress = ""

    def _print_user(self, text: str) -> None:
        if self._active_turn is not None:
            self._active_turn.finished = True
        turn = _ConversationTurn(user_text=text)
        self._turns.append(turn)
        self._active_turn = turn
        self._active_view = None
        self._process_lines = turn.process_lines
        self._assistant_draft = ""
        self._process_collapsed = False
        self._metric_summary = ""
        self._workspace_summary = ""
        self._last_process_progress = ""
        if self._app is not None:
            self._app.call_later(self._mount_turn, turn)

    def _print_ai_prefix(self) -> None:
        self._sync_active_view()

    def _create_stream_renderer(self) -> _FullscreenStreamSink:
        return _FullscreenStreamSink(self._append_assistant_text)

    def _before_tool_call(self) -> None:
        if self._assistant_draft:
            self._append_process("模型前置说明：\n" + self._assistant_draft)
            self._assistant_draft = ""
            if self._active_turn is not None:
                self._active_turn.answer = ""
                self._active_turn.process_collapsed = False
            self._process_collapsed = False
            self._sync_active_view()

    def _finalize_response(self, response: str) -> None:
        if not self._assistant_draft and response:
            self._assistant_draft = response
            if self._active_turn is not None:
                self._active_turn.answer = response
            self._append_process("模型未形成独立最终段，已回填最后生成文本")
        self._sync_active_view()

    def _print_success(self) -> None:
        self._append_process("✓ 本轮已正常完成")
        self._set_process_collapsed(True)
        if self._active_turn is not None:
            self._active_turn.finished = True
        self._refresh_chrome()

    def _print_info(self, text: str) -> None:
        if self._command_active and not self._runtime.active:
            self._append_system_answer(text, kind="command")
            return
        if text.startswith("安全确认："):
            self._add_notice("approval", "安全确认结果", text.removeprefix("安全确认："))
            return
        self._append_process(text)

    def _print_warning(self, text: str) -> None:
        if self._command_active and not self._runtime.active:
            self._append_system_answer("⚠ " + text, kind="warning")
            return
        self._add_notice("warning", "提示", text)

    def _print_approval(self, text: str) -> None:
        self._add_notice("approval", "需要安全确认", text)

    def _print_error(self, text: str) -> None:
        if self._command_active and not self._runtime.active:
            self._append_system_answer("错误：" + text, kind="error")
            return
        self._add_notice("error", "错误", text)
        self._set_process_collapsed(True)
        self._refresh_chrome()

    def _clear_display(self) -> None:
        self._turns.clear()
        self._active_turn = None
        self._active_view = None
        self._process_lines = []
        self._assistant_draft = ""
        self._metric_summary = ""
        self._workspace_summary = ""
        self._last_process_progress = ""
        self._process_collapsed = False
        if self._app is not None:
            turn_list = self._app.query_one("#turn-list", Vertical)
            self._app.call_later(turn_list.remove_children)

    def _render_workspace_changes(self, changes: WorkspaceChanges) -> None:
        counts: list[str] = []
        if changes.added:
            counts.append(f"新增 {len(changes.added)}")
        if changes.modified:
            counts.append(f"修改 {len(changes.modified)}")
        if changes.deleted:
            counts.append(f"删除 {len(changes.deleted)}")
        total = len(changes.added) + len(changes.modified) + len(changes.deleted)
        entries = [f"已编辑 {total} 个文件 · " + " · ".join(counts)]
        entries.extend(f"+ {path}" for path in changes.added)
        entries.extend(f"~ {path}" for path in changes.modified)
        entries.extend(f"- {path}" for path in changes.deleted)
        self._workspace_summary = "\n".join(entries)
        if self._active_turn is not None:
            self._active_turn.workspace_summary = self._workspace_summary
        self._sync_active_view()

    def _print_turn_metrics(self, *, metrics: TurnMetrics, model_requests: int) -> None:
        usage = getattr(self._agent_loop, "turn_usage", None)
        token_text = (
            f"{getattr(usage, 'total_tokens', 0):,}"
            if usage is not None and getattr(usage, "available", False)
            else "不可用"
        )
        resources = [f"Token {token_text}"]
        cache = self._format_cache_summary()
        if cache:
            resources.append(cache)
        metrics_prefix = "本轮统计 · "
        metrics_indent = self._terminal_indent(metrics_prefix)
        self._metric_summary = (
            f"{metrics_prefix}Turn {metrics.turns} · 请求 {max(0, model_requests)} · "
            f"{metrics.elapsed_seconds:.2f} 秒 · 工具 {metrics.tool_calls}"
            f"（{metrics.success_rate_text}）\n"
            + metrics_indent
            + " · ".join(resources)
            + "\n"
            + self._build_context_snapshot().plain
        )
        if self._active_turn is not None:
            self._active_turn.metrics_summary = self._metric_summary
            self._active_turn.finished = True
        self._sync_active_view()
        self._refresh_chrome()

    def _start_progress(self, text: str) -> None:
        self._progress_text = text
        self._status_text = f"运行中 · {text}"
        if (
            not (self._command_active and not self._runtime.active)
            and text != self._last_process_progress
        ):
            self._last_process_progress = text
            self._append_process("· " + text)
        self._refresh_chrome()

    def _stop_progress(self) -> None:
        self._progress_text = None
        if self._command_active and not self._runtime.active:
            self._status_text = "就绪 · 可输入任务"
        self._refresh_chrome()

    async def request_tool_input(self, question: str, options: list[str]) -> str | None:
        self._stop_progress()
        lines = [f"需要你确认：{question}"]
        lines.extend(f"  {index}. {option}" for index, option in enumerate(options, 1))
        self._add_notice("approval", "需要你的选择", "\n".join(lines))
        prompt = "请输入选项序号或答案 › " if options else "请回答 › "
        while True:
            try:
                answer = (await self._read_control_input([("class:prompt", prompt)])).strip()
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

    async def _submit_input(self, text: str) -> None:
        if self._runtime.active and self._is_cancel_command(text):
            await self._handle_cancel_input(text)
        elif self._pending_control_input is not None and not self._pending_control_input.future.done():
            self._pending_control_input.future.set_result(text)
        elif self._runtime.active:
            await self._handle_active_input(text)
        elif self._cmd_dispatcher.is_command(text):
            await self._handle_command(text)
        else:
            self._start_user_input(text, display_user=True)
        if self._exit_requested and self._app is not None:
            self._app.exit()
        self._invalidate_input_prompt()

    def _textual_ready(self, app: _TinyCodeFullscreenApp) -> None:
        self._app = app
        for turn in self._turns:
            app.call_later(self._mount_turn, turn)

    def _mount_active_turn(self) -> None:
        if self._active_turn is not None:
            self._mount_turn(self._active_turn)

    def _mount_turn(self, turn: _ConversationTurn) -> None:
        if self._app is None or not self._app.is_running:
            return
        view = _TurnView(turn)
        self._app.query_one("#turn-list", Vertical).mount(view)
        if turn is self._active_turn:
            self._active_view = view
        self._app.scroll_to_latest()

    def _sync_active_view(self) -> None:
        if self._active_view is not None and self._active_view.is_mounted:
            self._active_view.sync()
        if self._app is not None and self._app.is_running:
            self._app.scroll_to_latest()

    def _append_assistant_text(self, text: str) -> None:
        self._assistant_draft += text
        if self._active_turn is not None:
            self._active_turn.answer = self._assistant_draft
        self._sync_active_view()

    def _append_system_answer(self, text: str, *, kind: str) -> None:
        if self._active_turn is not None:
            self._active_turn.finished = True
        turn = _ConversationTurn(
            user_text="", answer=text, finished=True, answer_is_markdown=False,
            answer_kind=kind,
        )
        self._turns.append(turn)
        self._active_turn = turn
        self._active_view = None
        self._process_lines = turn.process_lines
        self._assistant_draft = text
        self._process_collapsed = True
        self._metric_summary = ""
        self._workspace_summary = ""
        if self._app is not None:
            self._app.call_later(self._mount_turn, turn)

    def _add_notice(self, kind: str, title: str, text: str) -> None:
        if self._active_turn is None:
            turn = _ConversationTurn(user_text="")
            self._turns.append(turn)
            self._active_turn = turn
            self._process_lines = turn.process_lines
            if self._app is not None:
                self._app.call_later(self._mount_turn, turn)
        self._active_turn.notices.append(_SystemNotice(kind, title, text))
        self._sync_active_view()

    def _append_process(self, text: str) -> None:
        if self._active_turn is None:
            turn = _ConversationTurn(user_text="", process_collapsed=False)
            self._turns.append(turn)
            self._active_turn = turn
            self._process_lines = turn.process_lines
            if self._app is not None:
                self._app.call_later(self._mount_turn, turn)
        self._process_lines.append(text)
        while sum(len(line) for line in self._process_lines) > self._MAX_PROCESS_CHARS:
            self._process_lines.pop(0)
        self._sync_active_view()

    def _set_process_collapsed(self, collapsed: bool) -> None:
        self._process_collapsed = collapsed
        if self._active_turn is not None:
            self._active_turn.process_collapsed = collapsed
        self._sync_active_view()

    def _toggle_latest_process(self) -> None:
        if self._active_turn is not None and self._active_turn.process_lines:
            self._set_process_collapsed(not self._active_turn.process_collapsed)

    def _header_text(self) -> str:
        mcp = f" · MCP {self._mcp_server_count}" if self._mcp_server_count else ""
        return f"TinyCode · {self._provider_name}/{self._model}{mcp} · {self._status_text}"

    def _input_placeholder(self) -> str:
        return "".join(text for _style, text in self._input_prompt())

    def _refresh_chrome(self) -> None:
        if self._app is not None and self._app.is_running:
            self._app.refresh_header()

    def _invalidate_input_prompt(self) -> None:
        if self._app is not None and self._app.is_running:
            self._app.refresh_composer()

    async def run_async(self) -> None:
        if not self.supported():
            self._console = Console()
            await super().run_async()
            return

        app = _TinyCodeFullscreenApp(self)
        self._app = app
        self._input_loop_active = True
        try:
            await app.run_async(mouse=True)
        finally:
            self._input_loop_active = False
            if self._runtime.active:
                self._runtime.cancel()
            await self._wait_for_foreground()
            self._stop_progress()
            try:
                self._do_save()
            except Exception as exc:
                self._print_warning(f"退出前会话保存失败: {type(exc).__name__}: {exc}")
            await self.shutdown()

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from prompt_toolkit.document import Document
from rich.console import Console
from textual import events
from textual.containers import VerticalScroll
from textual.selection import SELECT_ALL
from textual.widgets import Button, Static, TextArea

from tinyCode.agent.events import (
    AgentDoneEvent,
    BackgroundResultsAppliedEvent,
    HITLRequestEvent,
    RoundLimitDecisionAction,
    RoundLimitExtendedEvent,
    RoundLimitReachedEvent,
    RoundStartEvent,
    TaskStalledDecisionAction,
    TaskStalledEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tinyCode.conversation.compression import CompressionResult
from tinyCode.conversation.history import ConversationHistory
from tinyCode.security.models import HITLDecision
from tinyCode.security.models import SecurityLevel
from tinyCode.providers.base import CacheUsage, TokenUsage, ToolCall
from tinyCode.tools.base import ToolResult
from tinyCode.tui.app import TinyCodeTUI, _StreamingMarkdownRenderer
from tinyCode.tui.factory import create_tui
from tinyCode.tui.fullscreen_textual import (
    FullscreenTinyCodeTUI,
    _Composer,
    _TinyCodeFullscreenApp,
    _TurnView,
)
from tinyCode.tui.workspace_changes import WorkspaceChanges
from tinyCode.config.models import TracingConfig
from tinyCode.tracing.recorder import TraceRecorder
from tinyCode.teams.auto import TeamRunResult


class FakeHistory:
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.deferred_messages: list[str] = []
        self.steering_messages: list[str] = []
        self.estimated_tokens = 0

    def add_user_message(self, content: str) -> None:
        self.user_messages.append(content)

    def flush_deferred(self) -> int:
        pending = self.deferred_messages + self.steering_messages
        self.deferred_messages = []
        self.steering_messages = []
        self.user_messages.extend(pending)
        return len(pending)

    def defer_user_message(self, content: str) -> None:
        if content:
            self.deferred_messages.append(content)

    def queue_steering_message(self, content: str) -> None:
        if content:
            self.steering_messages.append(content)

    @property
    def steering_count(self) -> int:
        return len(self.steering_messages)

    def discard_steering_messages(self) -> int:
        count = len(self.steering_messages)
        self.steering_messages = []
        return count

    def flush_steering(self) -> int:
        pending = self.steering_messages
        self.steering_messages = []
        self.user_messages.extend(pending)
        return len(pending)

    @property
    def deferred_count(self) -> int:
        return len(self.deferred_messages)

    def estimated_token_count(self) -> int:
        return self.estimated_tokens


class FakeCompressor:
    warning_threshold = 100
    context_window = 200

    def __init__(self) -> None:
        self.force_values: list[bool] = []

    def reset_circuit(self) -> None:
        pass

    def reset_warning(self) -> None:
        pass

    @property
    def circuit_open(self) -> bool:
        return False

    async def check_and_compress(
        self, history, provider, *, force: bool = False,
    ) -> CompressionResult:
        self.force_values.append(force)
        return CompressionResult(
            was_compressed=force,
            estimated_tokens_before=20,
            estimated_tokens_after=10,
        )


class FakeSessionStore:
    def save(self, history, provider_name: str, model: str) -> None:
        pass


class FakePromptSession:
    def __init__(self, answers=()) -> None:
        self.answers = list(answers)
        self.prompts: list[object] = []

    async def prompt_async(self, message=None):
        if callable(message):
            message = message()
        self.prompts.append(message)
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeAgentLoop:
    provider = object()
    plan_only = False
    cache_hit = False
    turn_model_requests = 1

    def __init__(self, response: str = "answer") -> None:
        self.response = response
        self.recorded_rounds: list[tuple[str, str]] = []
        self.max_rounds = 30
        self.round_extension = 10
        self.hard_max_rounds = 100
        self.round_limit_action = "ask"

    async def run(self, history):
        yield TextDeltaEvent(self.response)
        yield AgentDoneEvent("no_tool_call")

    def record_round(self, user_msg: str, assistant_msg: str) -> None:
        self.recorded_rounds.append((user_msg, assistant_msg))

    async def update_notes_if_needed(self) -> int:
        return 0

    def toggle_plan_only(self) -> bool:
        return self.plan_only

    def set_security_level(self, level) -> None:
        pass

    def set_max_rounds(self, value: int) -> int:
        self.max_rounds = value
        return value

    def set_round_extension(self, value: int) -> int:
        self.round_extension = value
        return value

    def set_hard_max_rounds(self, value: int) -> int:
        self.hard_max_rounds = value
        return value

    def set_round_limit_action(self, value: str) -> str:
        self.round_limit_action = value
        return value

    def get_system_prompt(self, section: str = "all") -> str:
        return f"system prompt:{section}"

    def cancel(self) -> None:
        pass


class FailingOnceAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__("recovered")
        self.calls = 0

    async def run(self, history):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider connection dropped")
        yield TextDeltaEvent(self.response)
        yield AgentDoneEvent("no_tool_call")


class SequentialAgentLoop(FakeAgentLoop):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)

    async def run(self, history):
        yield TextDeltaEvent(self.responses.pop(0))
        yield AgentDoneEvent("no_tool_call")


class MetricsAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__("done")
        self.turn_usage = TokenUsage(
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
            available=True,
        )
        self.turn_cache_usage = CacheUsage(
            read_tokens=80,
            write_tokens=20,
            miss_tokens=20,
            available=True,
        )
        self.turn_model_requests = 2

    async def run(self, history):
        yield RoundStartEvent(round_number=1, max_rounds=30)
        for index, success in enumerate((True, True, False), start=1):
            name = "shared_tool"
            yield ToolCallEvent(ToolCall(f"call_{index}", name, {}))
            yield ToolResultEvent(
                tool_name=name,
                call_id=f"call_{index}",
                result=ToolResult(
                    success=success,
                    content="ok" if success else "",
                    error="failed" if not success else "",
                ),
            )
        yield TextDeltaEvent("done")
        yield AgentDoneEvent("no_tool_call")


class WorkspaceChangingAgentLoop(FakeAgentLoop):
    def __init__(self, root: Path) -> None:
        super().__init__("done")
        self.root = root

    async def run(self, history):
        yield ToolCallEvent(ToolCall("call_1", "run_command", {"command": "generator"}))
        (self.root / "created.py").write_text("created\n", encoding="utf-8")
        (self.root / "changed.py").write_text("after\n", encoding="utf-8")
        (self.root / "deleted.py").unlink()
        yield ToolResultEvent(
            tool_name="run_command",
            call_id="call_1",
            result=ToolResult(success=True, content="ok"),
        )
        yield TextDeltaEvent("done")
        yield AgentDoneEvent("no_tool_call")


class ReadOnlyAgentLoop(FakeAgentLoop):
    def tool_may_modify_workspace(self, _tool_name: str, _params=None) -> bool:
        return False

    async def run(self, history):
        yield ToolCallEvent(ToolCall("call_1", "read_file", {"path": "README.md"}))
        yield ToolResultEvent(
            tool_name="read_file",
            call_id="call_1",
            result=ToolResult(success=True, content="read"),
        )
        yield TextDeltaEvent("done")
        yield AgentDoneEvent("no_tool_call")


class RoundReportingAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.round_started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, history):
        yield RoundStartEvent(round_number=2, max_rounds=30)
        self.round_started.set()
        await self.release.wait()
        yield TextDeltaEvent(self.response)
        yield AgentDoneEvent("no_tool_call")


class PausingAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_sent = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, history):
        yield TextDeltaEvent("x" * 160)
        self.chunk_sent.set()
        await self.release.wait()
        yield AgentDoneEvent("no_tool_call")


class CancellableAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancel_called = False

    async def run(self, history):
        self.started.set()
        await asyncio.Event().wait()
        yield AgentDoneEvent("no_tool_call")

    def cancel(self) -> None:
        self.cancel_called = True


class LineStreamingAgentLoop(FakeAgentLoop):
    async def run(self, history):
        for chunk in (
            "```python\n",
            "def merge_sort(values):\n",
            "    return values\n",
            "```",
        ):
            yield TextDeltaEvent(chunk)
            await asyncio.sleep(0)
        yield AgentDoneEvent("no_tool_call")


class AskingAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.input_handler = None

    async def run(self, history):
        assert self.input_handler is not None
        answer = await self.input_handler("选择实现语言", ["Python", "Rust"])
        yield TextDeltaEvent(f"selected:{answer}")
        yield AgentDoneEvent("no_tool_call")


class ApprovalAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.future: asyncio.Future | None = None

    async def run(self, history):
        self.future = asyncio.get_running_loop().create_future()
        yield HITLRequestEvent(
            tool_name="run_command",
            params={"command": "ls -la"},
            prompt="confirm",
            future=self.future,
        )
        await self.future
        yield TextDeltaEvent("approved")
        yield AgentDoneEvent("no_tool_call")


class RoundBudgetAgentLoop(FakeAgentLoop):
    def __init__(self, *, stalled: bool = False) -> None:
        super().__init__()
        self.future: asyncio.Future | None = None
        self.stalled = stalled

    async def run(self, history):
        self.future = asyncio.get_running_loop().create_future()
        yield RoundStartEvent(30, 30)
        yield RoundLimitReachedEvent(
            round_number=30,
            current_limit=30,
            extension=10,
            hard_limit=100,
            stalled=self.stalled,
            future=self.future,
        )
        decision = await self.future
        if decision.action == RoundLimitDecisionAction.STOP:
            yield AgentDoneEvent("round_budget_stopped")
            return
        new_limit = decision.requested_limit or 40
        yield RoundLimitExtendedEvent(
            previous_limit=30,
            new_limit=new_limit,
            hard_limit=100,
            automatic=decision.action == RoundLimitDecisionAction.AUTO,
        )
        yield TextDeltaEvent("finished")
        yield AgentDoneEvent("no_tool_call")


class HardLimitAgentLoop(FakeAgentLoop):
    async def run(self, history):
        yield RoundStartEvent(100, 100)
        yield AgentDoneEvent("hard_max_rounds")


class BackgroundResultReportingLoop(FakeAgentLoop):
    def __init__(self, continued: bool) -> None:
        super().__init__()
        self.continued = continued

    async def run(self, history):
        yield BackgroundResultsAppliedEvent(
            result_count=2,
            continued=self.continued,
        )
        yield AgentDoneEvent("no_tool_call" if self.continued else "hard_max_rounds")


class StalledAgentLoop(FakeAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.future: asyncio.Future | None = None

    async def run(self, history):
        self.future = asyncio.get_running_loop().create_future()
        yield TaskStalledEvent(
            state="oscillating",
            reasons=("最近 4 轮状态呈 A-B-A-B 往返震荡",),
            recovery_prompt="try another strategy",
            round_number=4,
            continue_rounds=5,
            hard_limit=20,
            future=self.future,
        )
        decision = await self.future
        if decision.action == TaskStalledDecisionAction.STOP:
            yield AgentDoneEvent("stalled")
            return
        yield TextDeltaEvent("recovered")
        yield AgentDoneEvent("no_tool_call")


class TuiNotesTests(unittest.IsolatedAsyncioTestCase):
    def _make_tui(self, agent_loop=None, answers=(), trace_recorder=None):
        output = io.StringIO()
        tui = TinyCodeTUI(
            agent_loop=agent_loop or FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
            trace_recorder=trace_recorder,
            console=Console(
                file=output,
                force_terminal=False,
                color_system=None,
                width=120,
            ),
            prompt_session=FakePromptSession(answers),
        )
        return tui, output

    async def test_manual_compress_command_bypasses_threshold(self):
        tui, _ = self._make_tui()

        result = await tui.trigger_compress()

        self.assertEqual([True], tui._compressor.force_values)
        self.assertIn("上下文已压缩", result)

    async def test_image_turn_keeps_structured_content_in_history(self):
        class VisionProvider:
            def supports_images(self):
                return True

        loop = FakeAgentLoop("看到了截图")
        loop.provider = VisionProvider()
        tui, output = self._make_tui(loop)
        content = [
            {"type": "text", "text": "分析截图"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/screen.png"},
            },
        ]

        scheduled = tui.send_image_to_conversation(
            content, "[图片: screen.png]\n分析截图",
        )
        await tui._wait_for_foreground()

        self.assertTrue(scheduled)
        self.assertEqual(content, tui._history.user_messages[0])
        self.assertIn("screen.png", output.getvalue())
        self.assertIn("看到了截图", output.getvalue())

    async def test_selected_local_image_is_persisted_then_sent(self):
        class VisionProvider:
            def supports_images(self):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "screen.png"
            source.write_bytes(b"\x89PNG\r\n\x1a\nimage")
            loop = FakeAgentLoop("图片分析完成")
            loop.provider = VisionProvider()
            tui, _output = self._make_tui(loop)
            tui._worktree_manager = SimpleNamespace(repo_root=root)

            scheduled = await tui.submit_image_source(
                str(source), "分析这个截图",
            )
            await tui._wait_for_foreground()

            self.assertTrue(scheduled)
            content = tui._history.user_messages[0]
            self.assertEqual("image_file", content[1]["type"])
            stored = Path(content[1]["image_file"]["path"])
            self.assertTrue(stored.is_file())
            self.assertEqual(
                (root / ".tinyCode" / "attachments").resolve(),
                stored.parent.resolve(),
            )

    async def test_prompt_command_prints_snapshot_without_starting_turn(self):
        tui, output = self._make_tui(answers=["/prompt base", "/exit"])

        await tui.run_async()

        self.assertIn("system prompt:base", output.getvalue())
        self.assertEqual([], tui._history.user_messages)

    async def test_request_tool_input_accepts_option_number(self):
        tui, output = self._make_tui(answers=["2"])

        answer = await tui.request_tool_input(
            "选择数据库", ["PostgreSQL", "SQLite"],
        )

        self.assertEqual("SQLite", answer)
        self.assertIn("需要你确认：选择数据库", output.getvalue())
        self.assertIn("2. SQLite", output.getvalue())

    async def test_request_tool_input_can_be_cancelled(self):
        tui, _ = self._make_tui(answers=[EOFError()])

        answer = await tui.request_tool_input("需要值", [])

        self.assertIsNone(answer)

    async def test_foreground_tool_input_uses_single_input_loop(self):
        loop = AskingAgentLoop()
        tui, output = self._make_tui(loop, answers=["开始", "2"])
        loop.input_handler = tui.request_tool_input

        await tui.run_async()

        self.assertIn("需要你确认：选择实现语言", output.getvalue())
        self.assertIn("TinyCode: selected:Rust", output.getvalue())
        self.assertEqual(["开始"], tui._history.user_messages)

    async def test_repeated_prompt_command_prints_every_snapshot_in_terminal(self):
        tui, _ = self._make_tui(
            answers=["/prompt base", "/prompt base", "/exit"]
        )
        output = io.StringIO()
        tui._console = Console(
            file=output,
            force_terminal=True,
            color_system="standard",
            width=80,
        )

        await tui.run_async()

        self.assertEqual(2, output.getvalue().count("system prompt:base"))
        self.assertEqual([], tui._history.user_messages)

    async def test_exit_command_leaves_input_loop_without_model_turn(self):
        tui, output = self._make_tui(answers=["/exit"])

        await tui.run_async()

        self.assertTrue(tui._exit_requested)
        self.assertEqual([], tui._history.user_messages)
        self.assertIn("正在保存当前会话并安全退出", output.getvalue())
        self.assertIn("Goodbye!", output.getvalue())

    async def test_configured_security_level_is_visible_in_tui(self):
        output = io.StringIO()
        tui = TinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
            security_level=SecurityLevel.STRICT,
            console=Console(file=output, force_terminal=False, color_system=None),
            prompt_session=FakePromptSession(),
        )

        self.assertEqual("strict", tui.get_security_level())

    async def test_team_runner_is_wired_into_command_registry(self):
        calls = []

        async def runner(team_name: str, goal: str) -> str:
            calls.append((team_name, goal))
            return "team ok"

        output = io.StringIO()
        tui = TinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
            team_runner=runner,
            console=Console(file=output, force_terminal=False, color_system=None),
            prompt_session=FakePromptSession(["a"]),
        )

        was_command, result = await tui._cmd_dispatcher.dispatch("/team run Alpha do it")

        self.assertTrue(was_command)
        self.assertEqual("team ok", result)
        self.assertEqual([("Alpha", "do it")], calls)

    async def test_automatic_team_proposal_runs_and_preserves_review_on_decline(self):
        class Proposal:
            def render(self):
                return "Team proposal"

        class Service:
            config = SimpleNamespace(require_plan_approval=True)

            def propose(self, text):
                return Proposal()

            async def preflight(self):
                return True, ""

            async def run(self, proposal):
                return TeamRunResult(
                    "team answer", run_id="abcdef123456", review_ready=True,
                )

            async def apply(self, run_id):
                raise AssertionError("review should remain pending")

        output = io.StringIO()
        history = ConversationHistory()
        tui = TinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=history,
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
            auto_team_service=Service(),
            console=Console(file=output, force_terminal=False, color_system=None),
            prompt_session=FakePromptSession(["y", "n"]),
        )

        await tui._on_user_input("complex task")

        self.assertEqual(
            ["user", "assistant"],
            [message["role"] for message in history.get_messages()],
        )
        self.assertIn("team answer", output.getvalue())
        self.assertIn("/team review apply abcdef123456", output.getvalue())

    async def test_completed_round_is_recorded_before_note_update_is_scheduled(self):
        agent_loop = FakeAgentLoop()
        tui, _output = self._make_tui(agent_loop)
        tui._note_manager = object()
        recorded_at_schedule: list[bool] = []

        def fake_ensure_future(coro):
            recorded_at_schedule.append(bool(agent_loop.recorded_rounds))
            coro.close()
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

        with patch("tinyCode.tui.app.asyncio.ensure_future", fake_ensure_future):
            await tui._on_user_input("hello")

        self.assertEqual([True], recorded_at_schedule)
        self.assertEqual([("hello", "answer")], agent_loop.recorded_rounds)

    async def test_next_turn_waits_for_cancelled_note_stream_cleanup(self):
        cleanup_done = asyncio.Event()

        class CleanupAwareLoop(FakeAgentLoop):
            async def run(self, history):
                if not cleanup_done.is_set():
                    raise RuntimeError("note provider stream still active")
                yield TextDeltaEvent("safe")
                yield AgentDoneEvent("no_tool_call")

        async def lingering_note_update():
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                cleanup_done.set()

        tui, output = self._make_tui(CleanupAwareLoop())
        note_task = tui._spawn(lingering_note_update())
        tui._note_update_task = note_task
        await asyncio.sleep(0)

        self.assertTrue(tui._start_user_input("next"))
        await tui._wait_for_foreground()

        self.assertTrue(cleanup_done.is_set())
        self.assertIn("TinyCode: safe", output.getvalue())

    async def test_normal_completion_is_explicit(self):
        tui, output = self._make_tui(FakeAgentLoop("done"))

        await tui._on_user_input("work")

        rendered = output.getvalue()
        self.assertIn("TinyCode: done", rendered)
        self.assertIn("✓ 本轮已正常完成", rendered)
        self.assertEqual("就绪 · 上一轮已正常完成", tui._status_text)

    async def test_turn_lifecycle_is_persisted_to_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            tui, _output = self._make_tui(
                FakeAgentLoop("done"),
                trace_recorder=recorder,
            )

            await tui._on_user_input("trace this task")

            path = recorder.latest_path()
            self.assertIsNotNone(path)
            assert path is not None
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("task_start", rows[0]["event"])
            self.assertEqual("task_end", rows[-1]["event"])
            self.assertEqual("no_tool_call", rows[-1]["status"])
            self.assertTrue(any(row["event"] == "context_snapshot" for row in rows))

    async def test_tool_result_trace_keeps_call_ids_for_same_named_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            tui, _output = self._make_tui(
                MetricsAgentLoop(),
                trace_recorder=recorder,
            )

            await tui._on_user_input("run concurrent tools")

            path = recorder.latest_path()
            assert path is not None
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            results = [row for row in rows if row["event"] == "tool_result"]
            self.assertEqual(3, len(results))
            self.assertEqual(
                ["call_1", "call_2", "call_3"],
                [row["attributes"]["call_id"] for row in results],
            )
            self.assertEqual(
                {"shared_tool"},
                {row["attributes"]["tool"] for row in results},
            )

    async def test_completion_shows_turn_token_time_and_tool_metrics(self):
        tui, output = self._make_tui(MetricsAgentLoop())

        with patch("tinyCode.tui.metrics.monotonic", side_effect=[10.0, 12.345]):
            await tui._on_user_input("work")

        rendered = output.getvalue()
        completion_index = rendered.index("✓ 本轮已正常完成")
        metrics_index = rendered.index("本轮统计 · Turn 1")
        self.assertLess(completion_index, metrics_index)
        self.assertIn("本轮统计 · Turn 1", rendered)
        self.assertIn("请求 2", rendered)
        self.assertIn("Token 125", rendered)
        self.assertIn("2.35 秒", rendered)
        self.assertIn("工具 3（66.7%）", rendered)
        self.assertIn(
            "           Token 125 · Cache 命中 80（80%） · 写入 20",
            rendered,
        )

    async def test_completion_lists_all_workspace_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "changed.py").write_text("before\n", encoding="utf-8")
            (root / "deleted.py").write_text("delete me\n", encoding="utf-8")
            tui, output = self._make_tui(WorkspaceChangingAgentLoop(root))

            with patch("tinyCode.tui.app.Path.cwd", return_value=root):
                await tui._on_user_input("change workspace")

        rendered = output.getvalue()
        self.assertIn("文件变更 · 新增 1 · 修改 1 · 删除 1", rendered)
        self.assertIn("+ created.py", rendered)
        self.assertIn("~ changed.py", rendered)
        self.assertIn("- deleted.py", rendered)
        self.assertIn("- deleted.py", rendered)
        self.assertLess(rendered.index("TinyCode: done"), rendered.index("文件变更"))
        self.assertLess(rendered.index("文件变更"), rendered.index("✓ 本轮已正常完成"))
        self.assertLess(rendered.index("✓ 本轮已正常完成"), rendered.index("Turn "))
        self.assertLess(rendered.index("Turn "), rendered.index("上下文   ·"))

    async def test_read_only_tool_skips_workspace_scan(self):
        tui, _output = self._make_tui(ReadOnlyAgentLoop())

        with patch("tinyCode.tui.app.WorkspaceSnapshot.capture") as capture:
            await tui._on_user_input("read only")

        capture.assert_not_called()

    async def test_context_snapshot_is_shown_after_turn_metrics(self):
        tui, output = self._make_tui(FakeAgentLoop("done"))
        tui._history.estimated_tokens = 58

        await tui._on_user_input("work")

        rendered = output.getvalue()
        self.assertLess(rendered.index("Turn "), rendered.index("上下文   ·"))
        self.assertIn("[███░░░░░░░░░] 29% · 已用 ≈58 / 总计 200", rendered)
        self.assertNotIn("剩余", rendered)

    async def test_context_snapshot_prefers_last_provider_usage(self):
        loop = FakeAgentLoop("done")
        loop.provider = SimpleNamespace(
            last_usage={
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
            }
        )
        tui, output = self._make_tui(loop)
        tui._history.estimated_tokens = 10

        await tui._on_user_input("work")

        rendered = output.getvalue()
        self.assertIn("[██████░░░░░░] 50% · 已用 ≈100 / 总计 200", rendered)

    async def test_context_snapshot_keeps_small_nonzero_percentage(self):
        tui, output = self._make_tui(FakeAgentLoop("done"))
        tui._history.estimated_tokens = 4_100
        tui._compressor.context_window = 2_600_000

        await tui._on_user_input("work")

        self.assertIn("0.2% · 已用 ≈4.1k / 总计 2.6m", output.getvalue())

    async def test_context_snapshot_does_not_underreport_retained_history(self):
        loop = FakeAgentLoop("done")
        loop.provider = SimpleNamespace(
            last_usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            }
        )
        tui, output = self._make_tui(loop)
        tui._history.estimated_tokens = 120

        await tui._on_user_input("work")

        self.assertIn("[███████░░░░░] 60% · 已用 ≈120 / 总计 200", output.getvalue())

    async def test_stream_is_continuous_and_each_chunk_is_printed_once(self):
        tui, output = self._make_tui(LineStreamingAgentLoop())

        await tui._on_user_input("输出归并排序")

        rendered = output.getvalue()
        code = "╭─ python\n│ def merge_sort(values):\n│     return values\n╰─"
        self.assertIn(code, rendered)
        self.assertEqual(1, rendered.count("def merge_sort(values):"))
        self.assertNotIn("模型正在输出", rendered)
        self.assertNotIn("等待模型", rendered)

    def test_streaming_markdown_renderer_preserves_text_and_adds_styles(self):
        output = io.StringIO()
        console = Console(
            file=output,
            record=True,
            force_terminal=True,
            color_system="standard",
            width=120,
        )
        renderer = _StreamingMarkdownRenderer(console)

        self.assertTrue(renderer.write("普通正文"))
        for chunk in (
            "\n# 标题\n",
            "- 条目\n1. 步骤\n",
            "> 引用\n```python\n",
            "print('ok')\n",
            "```",
        ):
            renderer.write(chunk)
        renderer.close_line()

        styled = console.export_text(styles=True, clear=False)
        plain = console.export_text()
        expected = (
            "普通正文\n# 标题\n- 条目\n1. 步骤\n> 引用\n"
            "╭─ python\n│ print('ok')\n╰─\n"
        )
        self.assertEqual(expected, plain)
        self.assertIn("\x1b[", styled)

    def test_streaming_markdown_renderer_styles_inline_markdown(self):
        source = "**加粗**、*斜体*、~~删除~~、`代码`、[链接](https://example.com)"

        styled = _StreamingMarkdownRenderer._style_inline(source)

        self.assertEqual(source, styled.plain)
        styles = {span.style for span in styled.spans}
        self.assertTrue(
            {"bold", "italic", "strike", "bold cyan", "underline blue"}
            .issubset(styles)
        )

    def test_fullscreen_reclassifies_pre_tool_draft_and_keeps_final_text(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        tui._print_user("完成任务")
        sink = tui._create_stream_renderer()
        sink.write("我先检查项目。")
        tui._before_tool_call()
        sink.write("最终答案。")
        tui._print_success()

        self.assertIn("模型前置说明：\n我先检查项目。", tui._process_lines)
        self.assertNotIn("我先检查项目。", tui._assistant_draft)
        self.assertEqual("最终答案。", tui._assistant_draft)
        self.assertTrue(tui._process_collapsed)

    async def test_fullscreen_chat_feed_uses_independent_message_widgets(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            tui._print_user("修复问题")
            tui._append_process("· 正在检查文件")
            tui._append_assistant_text("# 修复完成\n说明文本")
            tui._set_process_collapsed(True)
            tui._render_workspace_changes(WorkspaceChanges(
                added=("new.py",), modified=("app.py",),
            ))
            await pilot.pause()

            views = list(app.query(_TurnView))
            self.assertEqual(1, len(views))
            view = views[0]
            self.assertEqual("修复问题", view.user.content)
            self.assertTrue(view.process.collapsed)
            self.assertIn("# 修复完成", view.turn.answer)
            self.assertIn("已编辑 2 个文件", view.turn.workspace_summary)

    async def test_fullscreen_attach_button_opens_picker_and_stages_image(self):
        loop = FakeAgentLoop()
        loop.provider = SimpleNamespace(supports_images=lambda: True)
        tui = FullscreenTinyCodeTUI(
            agent_loop=loop,
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="deepseek",
            model="deepseek-flash",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            with patch(
                "tinyCode.tui.fullscreen_textual.select_local_image",
                new=AsyncMock(return_value="/tmp/screen.png"),
            ):
                await pilot.click("#attach-button")
                await pilot.pause()

            composer = app.query_one("#composer", _Composer)
            self.assertEqual("", composer.text)
            self.assertEqual("/tmp/screen.png", app.pending_image_source)
            self.assertTrue(app.query_one("#attachment-row").display)
            self.assertIn(
                "screen.png",
                str(app.query_one("#attachment-label", Static).content),
            )
            self.assertFalse(app.query_one("#attach-button", Button).disabled)

    async def test_fullscreen_reasoning_indicator_animates_and_stops(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            tui._print_user("执行复杂任务")
            tui._start_progress("Reasoning")
            await pilot.pause()

            view = app.query_one(_TurnView)
            self.assertTrue(tui._active_turn.activity_active)
            first_frame = str(view.process_text.content)
            self.assertRegex(first_frame, r"[◐◓◑◒] Reasoning")
            view._advance_process_spinner()
            self.assertNotEqual(first_frame, str(view.process_text.content))

            tui._stop_progress()
            await pilot.pause()
            self.assertFalse(tui._active_turn.activity_active)
            self.assertIn("· Reasoning", view.process_text.content)
            self.assertNotIn("◐ Reasoning", view.process_text.content)

    async def test_fullscreen_mouse_selection_copies_without_mode_switch(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            tui._print_user("可直接复制")
            await pilot.pause()
            view = app.query_one(_TurnView)
            app.screen.selections = {view.user: SELECT_ALL}
            await pilot.pause()

            with patch.object(app, "copy_to_clipboard") as copy:
                app.post_message(events.TextSelected())
                await pilot.pause()
                copy.assert_called_once_with("可直接复制")

                copy.reset_mock()
                await pilot.press("super+c")
                copy.assert_called_once_with("可直接复制")

            app.screen.clear_selection()
            composer = app.query_one("#composer", _Composer)
            with patch.object(app, "copy_to_clipboard") as copy:
                composer.text = "输入框文本"
                composer.select_all()
                await pilot.pause()
                copy.assert_called_with("输入框文本")

                copy.reset_mock()
                await pilot.press("super+c")
                copy.assert_called_once_with("输入框文本")

    async def test_fullscreen_keeps_each_completed_turn_in_chat_history(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            tui._print_user("第一件事")
            tui._append_assistant_text("第一份回答")
            tui._print_success()
            tui._print_user("第二件事")
            tui._append_assistant_text("第二份回答")
            await pilot.pause()

            views = list(app.query(_TurnView))
            self.assertEqual(2, len(views))
            self.assertEqual("第一件事", views[0].turn.user_text)
            self.assertEqual("第一份回答", views[0].turn.answer)
            self.assertEqual("第二件事", views[1].turn.user_text)
            self.assertEqual("第二份回答", views[1].turn.answer)

    async def test_fullscreen_history_scroll_pauses_and_resumes_tail_follow(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(80, 18)) as pilot:
            for index in range(8):
                tui._print_user(f"任务 {index}")
                tui._append_assistant_text((f"回答 {index}\n") * 5)
                tui._print_success()
            await pilot.pause()

            scroll = app.query_one("#chat-scroll", VerticalScroll)
            self.assertGreater(scroll.max_scroll_y, 0)
            app.action_history_up()
            await pilot.pause()
            self.assertFalse(app.follow_tail)
            app.action_history_down()
            await pilot.pause()
            self.assertTrue(app.follow_tail)

            scroll.scroll_to(y=0, animate=False, force=True)
            await pilot.pause()
            tui._append_assistant_text("新的流式内容")
            await pilot.pause()
            self.assertFalse(app.follow_tail)
            self.assertLess(scroll.scroll_y, scroll.max_scroll_y)
            self.assertTrue(app.query_one("#new-output").display)
            await pilot.click("#new-output")
            await pilot.pause()
            self.assertTrue(app.follow_tail)
            self.assertFalse(app.query_one("#new-output").display)

    def test_tui_factory_keeps_stream_default_and_selects_fullscreen(self):
        kwargs = {
            "agent_loop": FakeAgentLoop(),
            "history": FakeHistory(),
            "compressor": FakeCompressor(),
            "session_store": FakeSessionStore(),
            "note_manager": None,
            "provider_name": "fake",
            "model": "fake",
        }
        self.assertIsInstance(create_tui(ui_mode="stream", **kwargs), TinyCodeTUI)
        with patch("tinyCode.tui.factory.FullscreenTinyCodeTUI.supported", return_value=True):
            self.assertIsInstance(
                create_tui(ui_mode="fullscreen", **kwargs), FullscreenTinyCodeTUI,
            )

    async def test_fullscreen_moves_pre_tool_text_to_process_after_real_events(self):
        class DraftThenToolLoop(FakeAgentLoop):
            def tool_may_modify_workspace(self, _tool_name: str, _params=None) -> bool:
                return False

            async def run(self, history):
                yield TextDeltaEvent("我先读取配置。")
                yield ToolCallEvent(ToolCall("call_1", "read_file", {"path": "x"}))
                yield ToolResultEvent(
                    tool_name="read_file",
                    call_id="call_1",
                    result=ToolResult(success=True, content="ok"),
                )
                yield TextDeltaEvent("最终结果。")
                yield AgentDoneEvent("no_tool_call")

        tui = FullscreenTinyCodeTUI(
            agent_loop=DraftThenToolLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )

        await tui._on_user_input("执行任务")

        self.assertIn("最终结果。", tui._assistant_draft)
        self.assertNotIn("我先读取配置。", tui._assistant_draft)
        self.assertTrue(any("我先读取配置。" in item for item in tui._process_lines))
        self.assertTrue(tui._process_collapsed)

    async def test_fullscreen_enter_submits_input_and_starts_a_turn(self):
        loop = FakeAgentLoop("已收到")
        tui = FullscreenTinyCodeTUI(
            agent_loop=loop,
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            app.query_one("#composer", TextArea).focus()
            await pilot.press(*"测试输入", "enter")
            await tui._wait_for_foreground()
            await pilot.pause()

        self.assertEqual(["测试输入"], tui._history.user_messages)
        self.assertIn("已收到", tui._assistant_draft)

    async def test_fullscreen_shift_enter_inserts_newline_before_submit(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop("已收到多行输入"),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            composer = app.query_one("#composer", _Composer)
            composer.focus()
            await pilot.press(*"第一行", "shift+enter", *"第二行")
            await pilot.pause()

            self.assertEqual("第一行\n第二行", composer.text)
            self.assertEqual([], tui._history.user_messages)

            await pilot.press("enter")
            await tui._wait_for_foreground()
            await pilot.pause()

        self.assertEqual(["第一行\n第二行"], tui._history.user_messages)
        self.assertIn("已收到多行输入", tui._assistant_draft)

    async def test_fullscreen_stop_button_tracks_and_cancels_active_task(self):
        loop = CancellableAgentLoop()
        tui = FullscreenTinyCodeTUI(
            agent_loop=loop,
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)

        async with app.run_test(size=(100, 36)) as pilot:
            stop_button = app.query_one("#stop-button", Button)
            self.assertFalse(stop_button.display)

            await pilot.press(*"执行长任务", "enter")
            await loop.started.wait()
            await pilot.pause()
            self.assertTrue(stop_button.display)
            self.assertFalse(stop_button.disabled)

            await pilot.click("#stop-button")
            await tui._wait_for_foreground()
            await pilot.pause()

            self.assertTrue(loop.cancel_called)
            self.assertFalse(stop_button.display)
            self.assertEqual("cancelled", tui._runtime.snapshot().last_outcome.value)

    async def test_fullscreen_application_shows_streamed_final_answer(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop("全屏 `回答`"),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)
        async with app.run_test(size=(100, 36)) as pilot:
            await pilot.press(*"测试全屏", "enter")
            await tui._wait_for_foreground()
            await app.workers.wait_for_complete()
            await pilot.pause()
            views = list(app.query(_TurnView))
            self.assertEqual(1, len(views))
            self.assertTrue(views[0].answer.display)
            self.assertFalse(views[0].plain_answer.display)

        self.assertIn("全屏 `回答`", tui._assistant_draft)

    async def test_fullscreen_exit_does_not_mount_command_card_during_unmount(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop("使用 `inline code` 完成。"),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)

        async with app.run_test(size=(100, 36)):
            await tui._submit_input("生成带行内代码的回答")
            await tui._wait_for_foreground()
            await tui._submit_input("/exit")

        self.assertTrue(tui._shutting_down)
        self.assertTrue(tui._exit_requested)

    async def test_fullscreen_token_line_aligns_with_turn(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=MetricsAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )

        await tui._on_user_input("统计任务")

        lines = tui._metric_summary.splitlines()
        self.assertTrue(any(
            "工具 shared_tool 失败：failed" in line
            for line in tui._process_lines
        ))
        self.assertTrue(lines[0].startswith("本轮统计 · Turn"))
        self.assertTrue(
            lines[1].startswith(tui._terminal_indent("本轮统计 · ") + "Token")
        )

    async def test_fullscreen_slash_menu_filters_and_tab_completes(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)

        async with app.run_test(size=(100, 36)) as pilot:
            composer = app.query_one("#composer", TextArea)
            composer.focus()
            await pilot.press("/", "p", "r", "o")
            await pilot.pause()

            self.assertIn("/prompt", app.command_candidates)
            self.assertTrue(app.query_one("#command-menu").display)
            await pilot.press("tab")
            await pilot.pause()

            self.assertEqual("/prompt ", composer.text)
            self.assertFalse(app.query_one("#command-menu").display)

    async def test_fullscreen_only_suggests_cancel_while_task_is_active(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        self.assertTrue(tui._runtime.reserve())
        app = _TinyCodeFullscreenApp(tui)

        async with app.run_test(size=(100, 36)) as pilot:
            app.query_one("#composer", TextArea).focus()
            await pilot.press("/")
            await pilot.pause()

            self.assertEqual(["/cancel"], app.command_candidates)

    async def test_fullscreen_slash_command_result_is_a_visible_message(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)

        async with app.run_test(size=(100, 36)) as pilot:
            await tui._submit_input("/help")
            await pilot.pause()

            views = list(app.query(_TurnView))
            self.assertEqual(1, len(views))
            self.assertTrue(views[0].turn.answer)
            self.assertFalse(views[0].turn.user_text)
            self.assertFalse(views[0].turn.answer_is_markdown)
            self.assertEqual("command", views[0].turn.answer_kind)
            self.assertFalse(views[0].answer.display)
            self.assertTrue(views[0].plain_answer.display)
            self.assertIn("\n", views[0].turn.answer)

    async def test_fullscreen_classifies_warning_error_and_approval_cards(self):
        tui = FullscreenTinyCodeTUI(
            agent_loop=FakeAgentLoop(),
            history=FakeHistory(),
            compressor=FakeCompressor(),
            session_store=FakeSessionStore(),
            note_manager=None,
            provider_name="fake",
            model="fake",
        )
        app = _TinyCodeFullscreenApp(tui)

        async with app.run_test(size=(100, 36)) as pilot:
            tui._print_user("执行危险操作")
            tui._print_warning("上下文接近上限")
            tui._print_approval("即将执行写操作")
            tui._print_error("工具执行失败")
            await pilot.pause()

            self.assertEqual(
                ["warning", "approval", "error"],
                [notice.kind for notice in tui._active_turn.notices],
            )
            view = list(app.query(_TurnView))[0]
            self.assertEqual(3, len(list(view.notice_list.children)))

    async def test_escaped_newlines_are_normalized_before_output_and_recording(self):
        loop = FakeAgentLoop("```java\\nclass QuickSort {}\\n```")
        tui, output = self._make_tui(loop)

        await tui._on_user_input("show code")

        rendered = output.getvalue()
        self.assertIn("╭─ java\n│ class QuickSort {}\n╰─", rendered)
        self.assertNotIn("\\nclass", rendered)
        self.assertEqual(
            [("show code", "```java\nclass QuickSort {}\n```")],
            loop.recorded_rounds,
        )

    async def test_real_code_escape_sequences_are_not_changed(self):
        code = '```python\nprint("\\\\n")\n```'
        loop = FakeAgentLoop(code)
        tui, output = self._make_tui(loop)

        await tui._on_user_input("show code")

        self.assertIn('print("\\\\n")', output.getvalue())
        self.assertEqual([("show code", code)], loop.recorded_rounds)

    async def test_goodbye_is_printed_after_note_shutdown_finishes(self):
        class ExitNoteManager:
            async def update_on_exit(self):
                output.write("notes-finished\n")
                return {}

        tui, output = self._make_tui(answers=["/exit"])
        tui._note_manager = ExitNoteManager()

        await tui.run_async()

        rendered = output.getvalue()
        self.assertLess(rendered.index("notes-finished"), rendered.index("Goodbye!"))

    async def test_hitl_uses_prompt_session_and_resolves_uppercase_choice(self):
        loop = ApprovalAgentLoop()
        tui, output = self._make_tui(loop, answers=["A"])

        await tui._on_user_input("run ls")

        self.assertEqual(HITLDecision.ALLOW_ONCE, loop.future.result())
        self.assertFalse(tui._runtime.waiting_for_approval)
        self.assertIn("安全确认：允许（本次）", output.getvalue())
        self.assertIn("TinyCode: approved", output.getvalue())

    async def test_invalid_hitl_choice_is_reprompted(self):
        loop = ApprovalAgentLoop()
        prompt = FakePromptSession(["x", "d"])
        tui, output = self._make_tui(loop)
        tui._prompt_session = prompt

        await tui._on_user_input("run ls")

        self.assertEqual(HITLDecision.DENY, loop.future.result())
        self.assertEqual(2, len(prompt.prompts))
        self.assertIn("请输入 A、S、P 或 D", output.getvalue())

    async def test_round_budget_can_extend_without_starting_a_new_turn(self):
        loop = RoundBudgetAgentLoop()
        tui, output = self._make_tui(loop, answers=["A"])

        await tui._on_user_input("long task")

        self.assertEqual(RoundLimitDecisionAction.EXTEND, loop.future.result().action)
        self.assertEqual(40, loop.future.result().requested_limit)
        rendered = output.getvalue()
        self.assertIn("任务尚未完成，已用完轮次预算 30/30", rendered)
        self.assertIn("轮次预算已扩展：30 → 40", rendered)
        self.assertIn("✓ 本轮已正常完成", rendered)

    async def test_round_budget_accepts_custom_target_and_stall_warning(self):
        loop = RoundBudgetAgentLoop(stalled=True)
        tui, output = self._make_tui(loop, answers=["/rounds 55"])

        await tui._on_user_input("long task")

        self.assertEqual(55, loop.future.result().requested_limit)
        self.assertIn("最近 3 轮重复了相同工具调用", output.getvalue())
        self.assertIn("轮次预算已扩展：30 → 55", output.getvalue())

    async def test_round_budget_config_syntax_also_updates_session_default(self):
        loop = RoundBudgetAgentLoop()
        tui, _ = self._make_tui(loop, answers=["/config max-rounds 55"])

        await tui._on_user_input("long task")

        self.assertEqual(55, loop.future.result().requested_limit)
        self.assertEqual(55, loop.max_rounds)

    async def test_stopping_at_soft_budget_is_not_rendered_as_completion(self):
        loop = RoundBudgetAgentLoop()
        tui, output = self._make_tui(loop, answers=["S"])

        await tui._on_user_input("long task")

        rendered = output.getvalue()
        self.assertIn("任务尚未完成，已暂停并保留当前进度", rendered)
        self.assertNotIn("✓ 本轮已正常完成", rendered)
        self.assertEqual("就绪 · 任务因轮次预算暂停", tui._status_text)

    async def test_hard_limit_is_distinct_from_normal_completion(self):
        tui, output = self._make_tui(HardLimitAgentLoop())

        await tui._on_user_input("long task")

        rendered = output.getvalue()
        self.assertIn("已达到轮次硬上限", rendered)
        self.assertNotIn("✓ 本轮已正常完成", rendered)
        self.assertEqual("就绪 · 任务达到轮次硬上限", tui._status_text)

    async def test_background_result_event_uses_its_own_count_in_both_states(self):
        continued_tui, continued_output = self._make_tui(
            BackgroundResultReportingLoop(True),
        )
        stopped_tui, stopped_output = self._make_tui(
            BackgroundResultReportingLoop(False),
        )

        await continued_tui._on_user_input("continue with worker result")
        await stopped_tui._on_user_input("worker result at hard limit")

        self.assertIn("已接收 2 个后台 Subagent 结果", continued_output.getvalue())
        self.assertIn("已保存 2 个后台 Subagent 结果", stopped_output.getvalue())

    async def test_stalled_task_can_request_strategy_change(self):
        loop = StalledAgentLoop()
        tui, output = self._make_tui(loop, answers=["R"])

        await tui._on_user_input("complex task")

        self.assertEqual(
            TaskStalledDecisionAction.STRATEGY,
            loop.future.result().action,
        )
        rendered = output.getvalue()
        self.assertIn("检测到任务方案往返震荡", rendered)
        self.assertIn("A-B-A-B", rendered)
        self.assertIn("TinyCode: recovered", rendered)
        self.assertIn("✓ 本轮已正常完成", rendered)

    async def test_stalled_task_stop_is_not_normal_completion(self):
        tui, output = self._make_tui(StalledAgentLoop(), answers=["S"])

        await tui._on_user_input("complex task")

        rendered = output.getvalue()
        self.assertIn("任务因持续无进展已暂停", rendered)
        self.assertNotIn("✓ 本轮已正常完成", rendered)
        self.assertEqual("就绪 · 任务因无进展暂停", tui._status_text)

    async def test_normal_prompt_is_restored_after_hitl_confirmation(self):
        loop = ApprovalAgentLoop()
        tui, output = self._make_tui(loop, answers=["run ls", "A"])
        prompt = tui._prompt_session

        await tui.run_async()

        prompt_texts = [
            "".join(fragment[1] for fragment in message)
            for message in prompt.prompts
        ]
        self.assertEqual(
            [
                "› ",
                "确认 [A本次/S会话/P永久/D拒绝] › ",
                "› ",
            ],
            prompt_texts,
        )
        self.assertEqual(["run ls"], tui._history.user_messages)
        self.assertEqual(1, output.getvalue().count("TinyCode: approved"))

    async def test_provider_exception_does_not_lock_later_turns(self):
        loop = FailingOnceAgentLoop()
        tui, output = self._make_tui(loop)
        tui._history = ConversationHistory()

        await tui._on_user_input("first")
        self.assertFalse(tui._runtime.active)
        self.assertIn("provider connection dropped", output.getvalue())

        await tui._on_user_input("second")
        self.assertFalse(tui._runtime.active)
        self.assertEqual(2, loop.calls)
        self.assertIn("TinyCode: recovered", output.getvalue())

    async def test_round_progress_is_transient_not_conversation_output(self):
        loop = RoundReportingAgentLoop()
        tui, output = self._make_tui(loop)

        task = asyncio.create_task(tui._on_user_input("summarize"))
        await loop.round_started.wait()

        self.assertEqual("第 2/30 轮 · 等待模型", tui._progress_text)
        self.assertNotIn("等待模型", output.getvalue())

        loop.release.set()
        await task
        self.assertIsNone(tui._progress_text)

    async def test_long_stream_chunk_is_visible_before_stream_finishes(self):
        loop = PausingAgentLoop()
        tui, output = self._make_tui(loop)

        task = asyncio.create_task(tui._on_user_input("stream"))
        await loop.chunk_sent.wait()
        await asyncio.sleep(0)

        self.assertIn("x" * 160, output.getvalue())

        loop.release.set()
        await task

    def test_slash_completion_keeps_the_slash(self):
        tui, _output = self._make_tui()

        completions = list(tui._completer.get_completions(Document("/he"), None))

        self.assertEqual(["/help"], [item.text for item in completions])
        self.assertEqual([-3], [item.start_position for item in completions])

    def test_empty_slash_completion_never_adds_two_slashes(self):
        tui, _output = self._make_tui()

        completions = list(tui._completer.get_completions(Document("/"), None))

        self.assertTrue(completions)
        self.assertTrue(all(item.text.startswith("/") for item in completions))
        self.assertFalse(any(item.text.startswith("//") for item in completions))

    async def test_second_prompt_does_not_replay_the_previous_answer(self):
        loop = SequentialAgentLoop(["quick-sort answer", "merge-sort answer"])
        tui, output = self._make_tui(loop)
        tui._prompt_session = FakePromptSession(["输出快速排序", "输出归并排序"])

        await tui.run_async()

        rendered = output.getvalue()
        self.assertEqual(1, rendered.count("quick-sort answer"))
        self.assertEqual(1, rendered.count("merge-sort answer"))
        self.assertLess(rendered.index("quick-sort answer"), rendered.index("merge-sort answer"))
        self.assertEqual(2, rendered.count("✓ 本轮已正常完成"))
        self.assertEqual(["输出快速排序", "输出归并排序"], tui._history.user_messages)

    async def test_active_turn_accepts_steering_and_cancel_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = PausingAgentLoop()
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            tui, output = self._make_tui(
                loop,
                answers=[
                    "开始长任务",
                    "不要修改测试，先检查根因",
                    "/cancel",
                    "/exit",
                ],
                trace_recorder=recorder,
            )

            await tui.run_async()

            self.assertEqual([], tui._history.steering_messages)
            rendered = output.getvalue()
            self.assertIn("已排队追加指令", rendered)
            self.assertIn("已丢弃 1 条尚未注入的追加指令", rendered)
            self.assertIn("正在取消当前任务", rendered)
            self.assertIn("Goodbye!", rendered)
            path = recorder.latest_path()
            assert path is not None
            events = {
                json.loads(line)["event"]
                for line in path.read_text(encoding="utf-8").splitlines()
            }
            self.assertIn("steering_queued", events)
            self.assertIn("cancellation_requested", events)

    def test_cancel_command_is_available_in_completion(self):
        tui, _output = self._make_tui()

        completions = list(
            tui._completer.get_completions(Document("/can"), None)
        )

        self.assertEqual(["/cancel"], [item.text for item in completions])

    async def test_non_tty_uses_plain_input_without_duplicate_user_render(self):
        output = io.StringIO()
        history = FakeHistory()
        with patch("tinyCode.tui.app.sys.stdin", io.StringIO()), patch(
            "builtins.input", side_effect=["你好", EOFError]
        ):
            tui = TinyCodeTUI(
                agent_loop=FakeAgentLoop("你好！有什么我可以帮你的吗？"),
                history=history,
                compressor=FakeCompressor(),
                session_store=FakeSessionStore(),
                note_manager=None,
                provider_name="fake",
                model="fake",
                console=Console(
                    file=output,
                    force_terminal=False,
                    color_system=None,
                ),
            )

            await tui.run_async()

        rendered = output.getvalue()
        self.assertFalse(tui._uses_prompt_toolkit)
        self.assertEqual(["你好"], history.user_messages)
        self.assertNotIn("> 你好", rendered)
        self.assertEqual(1, rendered.count("你好！有什么我可以帮你的吗？"))


if __name__ == "__main__":
    unittest.main()

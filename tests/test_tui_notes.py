import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.document import Document
from rich.console import Console

from tinyCode.agent.events import (
    AgentDoneEvent,
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
from tinyCode.providers.base import TokenUsage, ToolCall
from tinyCode.tools.base import ToolResult
from tinyCode.tui.app import TinyCodeTUI
from tinyCode.config.models import TracingConfig
from tinyCode.tracing.recorder import TraceRecorder


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

    async def check_and_compress(self, history, provider) -> CompressionResult:
        return CompressionResult()


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
        metrics_index = rendered.index("本轮统计")
        self.assertLess(completion_index, metrics_index)
        self.assertIn("Turn: 1", rendered)
        self.assertIn("模型请求: 2 次", rendered)
        self.assertIn("消耗 Token: 125", rendered)
        self.assertIn("耗时: 2.35 秒", rendered)
        self.assertIn("工具调用: 3 次", rendered)
        self.assertIn("成功率: 66.7%", rendered)

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
        self.assertIn("\n\n文件变更", rendered)
        self.assertIn("- deleted.py\n\n✓ 本轮已正常完成", rendered)
        self.assertLess(rendered.index("TinyCode: done"), rendered.index("文件变更"))
        self.assertLess(rendered.index("文件变更"), rendered.index("✓ 本轮已正常完成"))
        self.assertLess(rendered.index("✓ 本轮已正常完成"), rendered.index("本轮统计"))
        self.assertLess(rendered.index("本轮统计"), rendered.index("上下文   ·"))

    async def test_context_snapshot_is_shown_after_turn_metrics(self):
        tui, output = self._make_tui(FakeAgentLoop("done"))
        tui._history.estimated_tokens = 58

        await tui._on_user_input("work")

        rendered = output.getvalue()
        self.assertLess(rendered.index("本轮统计"), rendered.index("上下文   ·"))
        self.assertIn("≈58 / 200（29%）", rendered)
        self.assertIn("剩余 ≈142", rendered)

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
        self.assertIn("≈100 / 200（50%）", rendered)
        self.assertIn("剩余 ≈100", rendered)

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

        self.assertIn("≈120 / 200（60%）", output.getvalue())

    async def test_stream_is_continuous_and_each_chunk_is_printed_once(self):
        tui, output = self._make_tui(LineStreamingAgentLoop())

        await tui._on_user_input("输出归并排序")

        rendered = output.getvalue()
        code = "```python\ndef merge_sort(values):\n    return values\n```"
        self.assertIn(code, rendered)
        self.assertEqual(1, rendered.count("def merge_sort(values):"))
        self.assertNotIn("模型正在输出", rendered)
        self.assertNotIn("等待模型", rendered)

    async def test_escaped_newlines_are_normalized_before_output_and_recording(self):
        loop = FakeAgentLoop("```java\\nclass QuickSort {}\\n```")
        tui, output = self._make_tui(loop)

        await tui._on_user_input("show code")

        rendered = output.getvalue()
        self.assertIn("```java\nclass QuickSort {}\n```", rendered)
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

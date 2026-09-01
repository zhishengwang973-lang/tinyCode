import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tinyCode.agent.loop import AgentLoop
from tinyCode.agent.events import AgentDoneEvent, RoundStartEvent, TextDeltaEvent
from tinyCode.conversation.history import ConversationHistory
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.providers.base import ToolCall
from tinyCode.storage.recovery import TaskRecoveryStore
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tui.app import TinyCodeTUI
from rich.console import Console


class _WriteProbeTool(BaseTool):
    def __init__(self) -> None:
        self.executions = 0

    @property
    def name(self) -> str:
        return "write_probe"

    @property
    def description(self) -> str:
        return "Test-only write probe."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WRITE

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter("path", "string", "Target path")]

    async def execute(self, **kwargs) -> ToolResult:
        self.executions += 1
        return ToolResult(success=True, content="written")


class _BrokenRecoveryStore:
    def record_tool_intent(self, *args, **kwargs):
        raise OSError("disk full")

    def record_tool_result(self, *args, **kwargs):
        raise AssertionError("blocked writes have no result record")


class _SessionProbe:
    current_id = "session-ui"

    def __init__(self) -> None:
        self.saves = 0

    def save(self, history, provider_name, model) -> None:
        self.saves += 1


class _RecoveryUILoop:
    max_rounds = 5
    hard_max_rounds = 10
    turn_model_requests = 1
    turn_usage = SimpleNamespace(total_tokens=3)
    turn_cache_usage = SimpleNamespace(
        available=False, read_tokens=0, input_tokens=0,
    )
    cache_hit = False
    provider = SimpleNamespace(last_usage={})

    def __init__(self) -> None:
        self.recovery_task_ids: list[str | None] = []

    def set_recovery_task(self, task_id):
        self.recovery_task_ids.append(task_id)

    async def run(self, history):
        yield RoundStartEvent(1, 5)
        yield TextDeltaEvent("done")
        history.add_assistant_message("done")
        yield AgentDoneEvent("no_tool_call")

    def cancel(self):
        return None

    def record_round(self, user_msg, assistant_msg):
        return None


def _loop(tool: BaseTool, recovery_store) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(tool)
    provider = SimpleNamespace(config=SimpleNamespace(protocol="openai"))
    return AgentLoop(
        provider=provider,
        tool_registry=registry,
        tool_executor=ToolExecutor(default_timeout=2),
        prompt_builder=PromptBuilder(),
        prompt_injector=PromptInjector(),
        recovery_store=recovery_store,
    )


class TaskRecoveryStoreTests(unittest.TestCase):
    def test_unfinished_tool_intent_is_reported_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            with patch("tinyCode.storage.recovery.os.getpid", return_value=987654321):
                task = store.start_task(
                    session_id="session-1",
                    user_task="update config",
                    workspace=root,
                    turn_id=3,
                    max_rounds=30,
                    hard_max_rounds=100,
                )
                store.record_tool_intent(
                    task["task_id"], call_id="call-1", tool_name="write_file",
                    arguments={"path": "config.yaml", "content": "x"},
                    round_number=4, may_modify=True,
                )

            interrupted = store.find_latest_interrupted(root)

            self.assertIsNotNone(interrupted)
            pending = interrupted["inflight_tools"]
            self.assertEqual("write_file", pending["call-1"]["tool"])
            self.assertEqual(["config.yaml"], pending["call-1"]["targets"])
            prompt = store.build_recovery_prompt(interrupted)
            self.assertIn("不得盲目重试", prompt)
            self.assertIn("call_id=call-1", prompt)

    def test_completed_tool_result_resolves_wal_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            task = store.start_task(
                session_id="s", user_task="write", workspace=root, turn_id=1,
                max_rounds=5, hard_max_rounds=10,
            )
            store.record_tool_intent(
                task["task_id"], call_id="c", tool_name="write_file",
                arguments={"file": "a.txt"}, round_number=1, may_modify=True,
            )
            store.record_tool_result(
                task["task_id"], call_id="c", tool_name="write_file",
                success=True, content="ok",
            )

            self.assertEqual({}, store.unresolved_tools(task["task_id"]))
            rows = [
                json.loads(line)
                for line in (root / "recovery" / f"{task['task_id']}.tools.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["intent", "result"], [row["event"] for row in rows])
            self.assertNotIn("content", rows[1])

    def test_partial_model_draft_is_visible_but_not_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            with patch("tinyCode.storage.recovery.os.getpid", return_value=987654321):
                task = store.start_task(
                    session_id="s", user_task="answer", workspace=root, turn_id=1,
                    max_rounds=5, hard_max_rounds=10,
                )
                store.checkpoint_draft(task["task_id"], "unfinished answer", force=True)

            interrupted = store.find_latest_interrupted(root)
            self.assertTrue(interrupted["has_draft"])
            self.assertIn("仅供审计", store.describe(interrupted))
            self.assertNotIn("unfinished answer", store.build_recovery_prompt(interrupted))

    def test_pending_steering_can_be_restored_without_aliasing(self):
        history = ConversationHistory()
        history.restore_steering_messages(["first", "", "second"])
        pending = history.pending_steering_messages()
        pending.append("third")

        self.assertEqual(["first", "second"], history.pending_steering_messages())
        self.assertEqual(2, history.flush_steering())
        self.assertIn("[追加指令 1]", history.get_messages()[-1]["content"])

    def test_unresolved_tool_keeps_failed_task_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            with patch("tinyCode.storage.recovery.os.getpid", return_value=987654321):
                task = store.start_task(
                    session_id="s", user_task="write", workspace=root, turn_id=1,
                    max_rounds=5, hard_max_rounds=10,
                )
                store.record_tool_intent(
                    task["task_id"], call_id="c", tool_name="write_file",
                    arguments={"path": "a"}, round_number=1, may_modify=True,
                )
                store.finish(task["task_id"], "failed", error="hook crashed")

            interrupted = store.find_latest_interrupted(root)
            self.assertIsNotNone(interrupted)
            self.assertEqual("interrupted", interrupted["state"])

    def test_later_turn_supersedes_paused_task_in_same_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            first = store.start_task(
                session_id="s", user_task="first", workspace=root, turn_id=1,
                max_rounds=5, hard_max_rounds=10,
            )
            store.finish(first["task_id"], "paused")

            store.start_task(
                session_id="s", user_task="continue", workspace=root, turn_id=2,
                max_rounds=5, hard_max_rounds=10,
            )

            self.assertEqual("superseded", store.get(first["task_id"])["state"])

    def test_recovery_preserves_wait_reason_but_invalidates_old_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            task = store.start_task(
                session_id="s", user_task="run command", workspace=root, turn_id=1,
                max_rounds=5, hard_max_rounds=10,
            )
            store.update(
                task["task_id"], state="waiting_approval",
                waiting={"kind": "security_approval", "tool": "run_command"},
            )

            prepared = store.prepare_recovery(task["task_id"])

            self.assertEqual({}, prepared["waiting"])
            self.assertEqual(
                "security_approval", prepared["interrupted_waiting"]["kind"],
            )
            prompt = store.build_recovery_prompt(prepared)
            self.assertIn("旧审批都已失效", prompt)


class ToolWALTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_execution_writes_intent_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            task = store.start_task(
                session_id="s", user_task="write", workspace=root, turn_id=1,
                max_rounds=5, hard_max_rounds=10,
            )
            tool = _WriteProbeTool()
            loop = _loop(tool, store)
            loop.set_recovery_task(task["task_id"])
            loop._active_round = 2

            result = await loop._execute_tool_with_hooks(
                ToolCall("call-7", "write_probe", {"path": "out.txt"})
            )

            self.assertTrue(result.success)
            self.assertEqual(1, tool.executions)
            self.assertEqual({}, store.unresolved_tools(task["task_id"]))

    async def test_write_is_blocked_when_intent_cannot_be_persisted(self):
        tool = _WriteProbeTool()
        loop = _loop(tool, _BrokenRecoveryStore())
        loop.set_recovery_task("task")

        result = await loop._execute_tool_with_hooks(
            ToolCall("call", "write_probe", {"path": "out.txt"})
        )

        self.assertFalse(result.success)
        self.assertIn("已阻止", result.error)
        self.assertEqual(0, tool.executions)

    async def test_tui_finishes_task_manifest_after_session_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskRecoveryStore(root / "recovery")
            loop = _RecoveryUILoop()
            session = _SessionProbe()
            tui = TinyCodeTUI(
                agent_loop=loop,
                history=ConversationHistory(),
                compressor=SimpleNamespace(context_window=1000),
                session_store=session,
                note_manager=None,
                provider_name="test",
                model="model",
                recovery_store=store,
                console=Console(file=io.StringIO(), force_terminal=False),
            )

            await tui._on_user_input("finish safely")

            manifests = list((root / "recovery").glob("*.task.json"))
            self.assertEqual(1, len(manifests))
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["state"])
            self.assertGreaterEqual(session.saves, 2)
            self.assertEqual([manifest["task_id"], None], loop.recovery_task_ids)


if __name__ == "__main__":
    unittest.main()

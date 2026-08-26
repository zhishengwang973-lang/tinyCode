import asyncio
import json
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from tinyCode.conversation.history import ConversationHistory
from tinyCode.config.models import ProviderConfig, TracingConfig
from tinyCode.providers.base import BaseProvider, Message, ToolCall
from tinyCode.subagent.manager import BackgroundTaskManager
from tinyCode.subagent.models import SubAgentRole, SubAgentTask, TaskStatus
from tinyCode.subagent.runner import SubAgentRunner
from tinyCode.subagent.tool import SubAgentTool
from tinyCode.subagent.filter import ToolFilter
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tracing.recorder import TraceRecorder
from tinyCode.tracing.render import render_text


class ToolOnlyProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderConfig(
                name="test",
                protocol="openai",
                model="test-model",
                api_key="test-key",
            )
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id="tool-1", name="missing_tool", input={})

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = ""
    ) -> Message:
        return {
            "role": "assistant",
            "content": text_prefix or None,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": "{}"}}
                for tc in tool_calls
            ],
        }

    def make_tool_result_message(
        self, tool_call_id: str, tool_name: str, result_text: str
    ) -> Message:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result_text,
        }


class BlockingRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, task, history):
        self.started.set()
        await asyncio.Event().wait()


class FailingProvider(ToolOnlyProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        yield "partial"
        raise RuntimeError("connection dropped")


class FinalTextProvider(ToolOnlyProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        yield "sub-agent finished"


class CapturingFinalTextProvider(FinalTextProvider):
    def __init__(self) -> None:
        super().__init__()
        self.received_messages: list[list[Message]] = []

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.received_messages.append(messages)
        yield "sub-agent finished"


class SubAgentRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_background_filter_uses_registry_read_categories(self):
        role = SubAgentRole(name="reader", tools_allow=None)
        allowed = ToolFilter(
            role,
            background=True,
            read_tools={"read_file", "mcp_docs_resource"},
        ).filter(["read_file", "write_file", "mcp_docs_resource"])

        self.assertEqual(["mcp_docs_resource", "read_file"], allowed)

    def test_sub_agents_cannot_prompt_the_foreground_user(self):
        allowed = ToolFilter(
            SubAgentRole(name="worker", tools_allow=None),
        ).filter(["read_file", "request_user_input", "web_search"])

        self.assertEqual(["read_file", "web_search"], allowed)

    def test_background_result_is_context_not_orphan_tool_message(self):
        task_manager = BackgroundTaskManager()
        history = ConversationHistory()
        task = task_manager.create(None, "inspect", background=True)
        task.start()
        task.complete("finished", rounds=1)

        task_manager.inject_result(task, history)

        self.assertEqual([], history.get_messages())
        self.assertEqual(1, history.flush_deferred())
        messages = history.get_messages()
        self.assertEqual(1, len(messages))
        self.assertEqual("user", messages[0]["role"])
        self.assertIn("finished", messages[0]["content"])

    async def test_run_fails_when_sub_agent_produces_no_final_text(self):
        task = SubAgentTask(task="try a missing tool")
        task.start()
        runner = SubAgentRunner(
            provider=ToolOnlyProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={},
        )

        with self.assertRaisesRegex(RuntimeError, "未返回结果"):
            await runner.run(task, ConversationHistory())

        self.assertEqual(TaskStatus.FAILED, task.status)
        self.assertIn("未返回结果", task.result)

    async def test_fork_omits_in_flight_parent_tool_calls(self):
        provider = CapturingFinalTextProvider()
        runner = SubAgentRunner(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={},
        )
        parent_history = ConversationHistory()
        parent_history.add_user_message("审查当前项目")
        parent_history.add_raw_message({
            "role": "assistant",
            "content": "我会委派这个审查。",
            "tool_calls": [{
                "id": "call_subagent_in_flight",
                "type": "function",
                "function": {"name": "sub_agent", "arguments": "{}"},
            }],
        })
        task = SubAgentTask(task="只审查缓存相关逻辑")
        task.start()

        result = await runner.run(task, parent_history)

        self.assertEqual("sub-agent finished", result)
        self.assertEqual(1, len(provider.received_messages))
        copied = provider.received_messages[0]
        self.assertFalse(any("tool_calls" in message for message in copied))
        self.assertTrue(any(
            "我会委派这个审查。" in str(message.get("content"))
            for message in copied
        ))

    async def test_fork_preserves_completed_tool_call_pairs(self):
        provider = CapturingFinalTextProvider()
        runner = SubAgentRunner(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={},
        )
        parent_history = ConversationHistory()
        parent_history.add_raw_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_finished",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        })
        parent_history.add_raw_message({
            "role": "tool",
            "tool_call_id": "call_finished",
            "name": "read_file",
            "content": "README 内容",
        })
        task = SubAgentTask(task="总结已有上下文")
        task.start()

        await runner.run(task, parent_history)

        copied = provider.received_messages[0]
        assistant = next(message for message in copied if message.get("role") == "assistant")
        self.assertEqual("call_finished", assistant["tool_calls"][0]["id"])
        self.assertTrue(any(
            message.get("tool_call_id") == "call_finished" for message in copied
        ))

    async def test_sub_agent_tool_returns_failure_when_runner_has_no_result(self):
        task_manager = BackgroundTaskManager()
        tool = SubAgentTool(
            runner=SubAgentRunner(
                provider=ToolOnlyProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
                roles={},
            ),
            task_manager=task_manager,
            roles={"worker": SubAgentRole(name="worker", max_rounds=1)},
            history=ConversationHistory(),
        )

        result = await tool.execute(task="try a missing tool", role="worker")
        tasks = task_manager.list_tasks()

        self.assertFalse(result.success)
        self.assertIn("未返回结果", result.error)
        self.assertEqual(1, len(tasks))
        self.assertEqual(TaskStatus.FAILED, tasks[0].status)

    async def test_cancel_background_task_cancels_actual_coroutine(self):
        task_manager = BackgroundTaskManager()
        runner = BlockingRunner()
        history = ConversationHistory()
        tool = SubAgentTool(
            runner=runner,
            task_manager=task_manager,
            roles={"worker": SubAgentRole(name="worker")},
            history=history,
        )

        result = await tool.execute(
            task="wait forever",
            role="worker",
            background=True,
        )
        await runner.started.wait()
        task = task_manager.list_tasks()[0]

        self.assertTrue(result.success)
        self.assertTrue(task_manager.cancel(task.id))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(TaskStatus.CANCELLED, task.status)
        self.assertEqual(1, history.flush_deferred())
        self.assertIn("已取消", history.get_messages()[0]["content"])

    async def test_partial_text_does_not_hide_sub_agent_failure(self):
        task = SubAgentTask(task="inspect")
        task.start()
        runner = SubAgentRunner(
            provider=FailingProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={},
        )

        with self.assertRaisesRegex(RuntimeError, "connection dropped"):
            await runner.run(task, ConversationHistory())

        self.assertEqual(TaskStatus.FAILED, task.status)

    async def test_sub_agent_model_span_is_child_of_parent_tool_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            handle = recorder.begin_task("delegate")
            runner = SubAgentRunner(
                provider=FinalTextProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
                roles={},
                trace_recorder=recorder,
            )
            task = SubAgentTask(task="inspect")
            task.start()

            recorder.record("round_start", attributes={"round": 5})
            with recorder.span(
                "sub_agent", "tool", {"round": 5},
            ) as parent_span:
                result = await runner.run(task, ConversationHistory())
                parent_span.finish("ok")
            recorder.record("round_end", status="completed", attributes={"round": 5})
            recorder.finish_task(handle, status="no_tool_call")

            self.assertEqual("sub-agent finished", result)
            assert handle is not None
            rows = [
                json.loads(line)
                for line in handle.path.read_text(encoding="utf-8").splitlines()
            ]
            model_start = next(
                row for row in rows
                if row["event"] == "span_start" and row["kind"] == "model_request"
            )
            self.assertEqual(parent_span.span_id, model_start["parent_span_id"])
            rendered = render_text(handle.path)
            self.assertIn("Turn 5", rendered)
            self.assertIn("工具 sub_agent", rendered)
            self.assertIn("模型 request #1", rendered)

    async def test_background_concurrency_is_bounded_and_shutdown_joins_tasks(self):
        task_manager = BackgroundTaskManager(max_concurrent=1)
        runner = BlockingRunner()
        history = ConversationHistory()
        tool = SubAgentTool(
            runner=runner,
            task_manager=task_manager,
            roles={"worker": SubAgentRole(name="worker")},
            history=history,
        )

        first = await tool.execute("one", role="worker", background=True)
        await runner.started.wait()
        second = await tool.execute("two", role="worker", background=True)

        self.assertTrue(first.success)
        self.assertFalse(second.success)
        self.assertIn("并发上限", second.error)
        await task_manager.shutdown()
        self.assertEqual({}, task_manager._running)

    async def test_sub_agent_tool_rejects_invalid_model_arguments(self):
        task_manager = BackgroundTaskManager()
        tool = SubAgentTool(
            runner=BlockingRunner(),
            task_manager=task_manager,
            roles={"worker": SubAgentRole(name="worker")},
            history=ConversationHistory(),
        )

        bad_task = await tool.execute(task=[], role="worker")
        bad_role = await tool.execute(task="inspect", role=42)
        bad_background = await tool.execute(
            task="inspect", role="worker", background="yes",
        )

        self.assertFalse(bad_task.success)
        self.assertFalse(bad_role.success)
        self.assertFalse(bad_background.success)
        self.assertEqual([], task_manager.list_tasks())


if __name__ == "__main__":
    unittest.main()

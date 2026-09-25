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
from tinyCode.subagent.tool import SubAgentTool, SubAgentWaitTool
from tinyCode.subagent.filter import ToolFilter
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.context import use_workspace
from tinyCode.tools.read_file import ReadFileTool
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tools.write_file import WriteFileTool
from tinyCode.tracing.recorder import TraceRecorder
from tinyCode.tracing.render import render_text


async def _text_stream(text: str):
    yield text


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


class CapturingToolOnlyProvider(ToolOnlyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.received_messages: list[list[Message]] = []

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.received_messages.append(messages)
        yield ToolCall(id=f"missing-{len(self.received_messages)}", name="missing_tool", input={})


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


class IntermediateThenFinalProvider(ToolOnlyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        if self.calls == 1:
            yield "intermediate narration"
            yield ToolCall(id="missing-1", name="missing_tool", input={})
        else:
            yield "final report"


class SlowProvider(ToolOnlyProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        await asyncio.sleep(1)
        yield "too late"


class CapturingFinalTextProvider(FinalTextProvider):
    def __init__(self) -> None:
        super().__init__()
        self.received_messages: list[list[Message]] = []

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.received_messages.append(messages)
        yield "sub-agent finished"


class SubAgentRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_sub_agent_schema_explains_when_to_delegate(self):
        tool = SubAgentTool(
            runner=BlockingRunner(),
            task_manager=BackgroundTaskManager(),
            roles={
                "explorer": SubAgentRole(
                    name="explorer", description="只读探索",
                ),
            },
            history=ConversationHistory(),
        )

        self.assertIn("边界明确、可独立交付", tool.description)
        self.assertIn("简单问答", tool.description)
        self.assertIn("不要让多个可写任务修改重叠文件", tool.description)
        parameters = {parameter.name: parameter for parameter in tool.parameters}
        self.assertIn("目标、范围、约束和预期交付物", parameters["task"].description)
        self.assertIn("依赖当前对话上下文", parameters["role"].description)
        self.assertIn("可并行的只读后台任务", parameters["background"].description)

    def test_sub_agent_approval_discloses_effective_capabilities(self):
        roles = {
            "worker": SubAgentRole(
                name="worker",
                tools_allow=["read_file", "write_file"],
                max_rounds=7,
                permission="normal",
                timeout_seconds=90,
            ),
        }
        runner = SubAgentRunner(
            provider=FinalTextProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles=roles,
        )
        tool = SubAgentTool(
            runner=runner,
            task_manager=BackgroundTaskManager(),
            roles=roles,
            history=ConversationHistory(),
        )

        visible = tool.approval_parameters({
            "task": "inspect",
            "role": "worker",
            "background": True,
        })

        manifest = visible["capabilities"]
        self.assertEqual("worker", manifest["role"])
        self.assertEqual("normal", manifest["permission"])
        self.assertEqual(7, manifest["max_rounds"])
        self.assertEqual(90, manifest["timeout_seconds"])
        self.assertTrue(manifest["background"])
        self.assertEqual([], manifest["allowed_tools"])

    def test_sub_agent_side_effect_is_decided_per_call(self):
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        roles = {
            "reader": SubAgentRole(
                name="reader", tools_allow=["read_file"],
            ),
            "writer": SubAgentRole(
                name="writer", tools_allow=["write_file"],
            ),
        }
        runner = SubAgentRunner(
            provider=FinalTextProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            roles=roles,
        )
        tool = SubAgentTool(
            runner=runner,
            task_manager=BackgroundTaskManager(),
            roles=roles,
            history=ConversationHistory(),
        )

        self.assertTrue(tool.available_in_inspect)
        self.assertFalse(tool.may_modify({"task": "read", "role": "reader"}))
        self.assertTrue(tool.may_modify({"task": "write", "role": "writer"}))
        self.assertFalse(tool.may_modify({
            "task": "background review",
            "role": "writer",
            "background": True,
        }))
        reader_scope = tool.security_parameters({
            "task": "first review", "role": "reader",
        })["command"]
        same_scope = tool.security_parameters({
            "task": "another review", "role": "reader",
        })["command"]
        writer_scope = tool.security_parameters({
            "task": "write", "role": "writer",
        })["command"]
        self.assertEqual(reader_scope, same_scope)
        self.assertNotEqual(reader_scope, writer_scope)

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

        messages = history.get_messages()
        self.assertEqual(1, len(messages))
        self.assertEqual("system", messages[0]["role"])
        self.assertIn("不可信数据", messages[0]["content"])
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

        with self.assertRaisesRegex(RuntimeError, "未正常完成: hard_max_rounds"):
            await runner.run(task, ConversationHistory())

        self.assertEqual(TaskStatus.FAILED, task.status)
        self.assertIn("未正常完成", task.result)

    async def test_only_terminal_round_text_becomes_sub_agent_result(self):
        provider = IntermediateThenFinalProvider()
        runner = SubAgentRunner(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={"worker": SubAgentRole(name="worker", max_rounds=2)},
        )
        task = SubAgentTask(role="worker", task="inspect the current project")
        task.start()

        result = await runner.run(task, ConversationHistory())

        self.assertEqual("final report", result)
        self.assertNotIn("intermediate narration", result)

    async def test_sub_agent_inherits_parent_project_instructions(self):
        provider = CapturingFinalTextProvider()
        runner = SubAgentRunner(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={"worker": SubAgentRole(name="worker")},
            instructions_text="PROJECT RULE: always preserve public APIs",
        )
        task = SubAgentTask(role="worker", task="explain the task")
        task.start()

        await runner.run(task, ConversationHistory())

        self.assertTrue(any(
            "PROJECT RULE" in str(message.get("content"))
            for message in provider.received_messages[0]
        ))

    async def test_sub_agent_result_is_bounded_and_persisted(self):
        provider = FinalTextProvider()
        provider.chat_stream = lambda messages, tools=None, system_blocks=None: _text_stream(
            "x" * 20_000
        )
        role = SubAgentRole(name="worker")
        runner = SubAgentRunner(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={"worker": role},
        )
        task = SubAgentTask(role="worker", task="produce a report")
        task.start()

        with tempfile.TemporaryDirectory() as tmp, use_workspace(Path(tmp)):
            result = await runner.run(task, ConversationHistory())
            stored = Path(tmp) / task.result_path
            self.assertTrue(stored.is_file())
            self.assertEqual(
                20_000, len(stored.read_text(encoding="utf-8").rstrip("\n")),
            )

        self.assertLess(len(result), 17_000)
        self.assertIn("结果已截断", result)

    async def test_sub_agent_has_an_overall_wall_clock_timeout(self):
        role = SubAgentRole(name="worker", timeout_seconds=0.01)
        runner = SubAgentRunner(
            provider=SlowProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={"worker": role},
        )
        task = SubAgentTask(role="worker", task="wait")
        task.start()

        with self.assertRaisesRegex(RuntimeError, "超过 0.01 秒"):
            await runner.run(task, ConversationHistory())

        self.assertEqual(TaskStatus.FAILED, task.status)

    async def test_sub_agent_auto_extends_soft_budget_and_reserves_final_round(self):
        provider = CapturingToolOnlyProvider()
        role = SubAgentRole(
            name="worker",
            max_rounds=2,
            initial_rounds=1,
            round_extension=1,
            finalization_rounds=1,
        )
        runner = SubAgentRunner(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            roles={"worker": role},
        )
        task = SubAgentTask(role="worker", task="inspect")
        task.start()

        with self.assertRaisesRegex(RuntimeError, "hard_max_rounds"):
            await runner.run(task, ConversationHistory())

        self.assertEqual(2, len(provider.received_messages))
        second_request = str(provider.received_messages[1])
        self.assertIn("轮次预算已扩展", second_request)
        self.assertIn("最后一个可用轮次", second_request)

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

    def test_long_fork_history_keeps_a_recent_protocol_safe_suffix(self):
        messages = []
        for index in range(20):
            messages.extend([
                {"role": "user", "content": f"task {index}"},
                {"role": "assistant", "content": f"result {index}"},
            ])

        compacted = SubAgentRunner._compact_fork_messages(messages)

        self.assertLessEqual(len(compacted), 24)
        self.assertEqual("user", compacted[0]["role"])
        self.assertEqual("task 19", compacted[-2]["content"])
        self.assertEqual("result 19", compacted[-1]["content"])

    def test_fork_history_uses_the_character_budget_when_safe(self):
        messages = [
            {"role": "user", "content": f"task {index}: " + "x" * 3_000}
            for index in range(20)
        ]

        compacted = SubAgentRunner._compact_fork_messages(messages)

        self.assertLess(SubAgentRunner._fork_message_chars(compacted), 40_000)
        self.assertEqual(messages[-1]["content"], compacted[-1]["content"])

    def test_single_oversized_tool_pair_is_bounded_without_breaking_protocol(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "huge-call",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "huge-call",
                "name": "read_file",
                "content": "x" * 60_000,
            },
        ]

        compacted = SubAgentRunner._compact_fork_messages(messages)

        self.assertLessEqual(
            SubAgentRunner._fork_message_chars(compacted), 40_000,
        )
        self.assertEqual("huge-call", compacted[0]["tool_calls"][0]["id"])
        self.assertEqual("huge-call", compacted[1]["tool_call_id"])

    async def test_sub_agent_tool_returns_failure_when_runner_has_no_result(self):
        task_manager = BackgroundTaskManager()
        tool = SubAgentTool(
            runner=SubAgentRunner(
                provider=ToolOnlyProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
                roles={"worker": SubAgentRole(name="worker", max_rounds=1)},
            ),
            task_manager=task_manager,
            roles={"worker": SubAgentRole(name="worker", max_rounds=1)},
            history=ConversationHistory(),
        )

        result = await tool.execute(task="try a missing tool", role="worker")
        tasks = task_manager.list_tasks()

        self.assertFalse(result.success)
        self.assertTrue(result.error)
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
        notices = task_manager.drain_context_messages()
        self.assertEqual(1, len(notices))
        self.assertIn("已取消", notices[0])

    async def test_wait_tool_joins_and_consumes_background_notification(self):
        task_manager = BackgroundTaskManager()
        task = task_manager.create("worker", "inspect", background=True)
        task.start()

        async def finish():
            await asyncio.sleep(0)
            task.complete("worker result", tokens=7, rounds=2)
            task_manager.publish(task)

        running = asyncio.create_task(finish())
        task_manager.attach(task.id, running)
        tool = SubAgentWaitTool(task_manager)

        result = await tool.execute(task.id, timeout_seconds=1)

        self.assertTrue(result.success)
        self.assertEqual("worker result", result.content)
        self.assertEqual([], task_manager.drain_context_messages())

    async def test_wait_tool_owns_its_timeout_instead_of_generic_executor(self):
        task_manager = BackgroundTaskManager()
        task = task_manager.create("worker", "slow", background=True)
        task.start()

        async def finish():
            await asyncio.sleep(0.03)
            task.complete("finished")
            task_manager.publish(task)

        running = asyncio.create_task(finish())
        task_manager.attach(task.id, running)
        tool = SubAgentWaitTool(task_manager)

        result = await ToolExecutor(default_timeout=0.01).execute(
            tool,
            {"task_id": task.id, "timeout_seconds": 0.1},
        )

        self.assertTrue(tool.timeout_exempt)
        self.assertTrue(result.success)
        self.assertEqual("finished", result.content)

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

    async def test_background_sub_agent_writes_a_linked_detached_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = TraceRecorder(TracingConfig(), root)
            parent = recorder.begin_task("delegate", model="test-model")
            assert parent is not None
            runner = SubAgentRunner(
                provider=FinalTextProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
                roles={"worker": SubAgentRole(name="worker")},
                trace_recorder=recorder,
            )
            task = SubAgentTask(
                role="worker", task="inspect", background=True,
            )
            task.start()

            result = await runner.run(task, ConversationHistory())
            recorder.finish_task(parent, status="no_tool_call")

            traces = sorted((root / ".tinyCode" / "traces").glob("*.jsonl"))
            child = next(
                path for path in traces
                if path.resolve() != parent.path.resolve()
            )
            parent_rows = [
                json.loads(line) for line in parent.path.read_text().splitlines()
            ]
            child_rows = [
                json.loads(line) for line in child.read_text().splitlines()
            ]
            page = recorder.render_last_html()
            assert page is not None
            page_text = page.read_text(encoding="utf-8")
            child_html_exists = child.with_suffix(".html").is_file()
            reopened_latest = TraceRecorder(
                TracingConfig(), root,
            ).latest_path()

        self.assertEqual("sub-agent finished", result)
        self.assertEqual(2, len(traces))
        link = next(
            row for row in parent_rows
            if row["event"] == "subagent_trace_started"
        )
        self.assertEqual(task.id, link["attributes"]["task_id"])
        self.assertEqual(
            child.resolve(), Path(link["attributes"]["trace_path"]).resolve(),
        )
        self.assertEqual(parent.trace_id, child_rows[0]["attributes"]["parent_trace_id"])
        self.assertTrue(any(
            row.get("kind") == "model_request" for row in child_rows
        ))
        self.assertEqual("no_tool_call", child_rows[-1]["status"])
        self.assertIsNotNone(page)
        self.assertTrue(child_html_exists)
        self.assertIn("Subagent Trace", page_text)
        self.assertIsNotNone(reopened_latest)
        assert reopened_latest is not None
        self.assertEqual(parent.path.resolve(), reopened_latest.resolve())

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

    async def test_task_state_survives_restart_and_running_task_is_reconciled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = BackgroundTaskManager(project_root=root)
            task = first.create("worker", "inspect", background=True)
            first.mark_started(task)

            second = BackgroundTaskManager(project_root=root)
            restored = second.get(task.id)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(TaskStatus.FAILED, restored.status)
        self.assertIn("进程", restored.result)

    def test_undelivered_completion_survives_restart_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = BackgroundTaskManager(project_root=root)
            task = first.create("worker", "inspect", background=True)
            first.mark_started(task)
            task.complete("durable result", tokens=9, rounds=2)
            first.publish(task)

            second = BackgroundTaskManager(project_root=root)
            notices = second.drain_context_messages()
            third = BackgroundTaskManager(project_root=root)

        self.assertEqual(1, len(notices))
        self.assertIn("durable result", notices[0])
        self.assertEqual([], third.drain_context_messages())

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

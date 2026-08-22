import asyncio
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from tinyCode.agent.events import ContextCompressionEvent, ErrorEvent, RoundStartEvent, ToolBlockedEvent, ToolCallEvent, ToolResultEvent
from tinyCode.agent.loop import AgentLoop
from tinyCode.conversation.compression import CompressionResult
from tinyCode.conversation.history import ConversationHistory
from tinyCode.conversation.truncator import ToolResultTruncator, TruncateConfig
from tinyCode.config.models import ProviderConfig
from tinyCode.hooks.models import HookEvent
from tinyCode.providers.base import BaseProvider, Message, ProviderHTTPError, ToolCall
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tools.tool_result_read import ToolResultReadTool
from tinyCode.tools.tool_result_search import ToolResultSearchTool


class UnknownToolProvider(BaseProvider):
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


class ReadToolProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id="tool-1", name="read_fixture", input={"path": "README.md"})


class WriteToolProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id="tool-1", name="write_fixture", input={"path": "README.md"})


class BadInputProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id="tool-1", name="write_fixture", input=[])


class InvalidEventProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield {"unexpected": "event"}


class InvalidToolCallIdProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id=123, name="write_fixture", input={})


class InvalidToolCallNameProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id="tool-1", name=123, input={})


class DuplicateToolCallIdProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(id="duplicate", name="read_fixture", input={"path": "a"})
        yield ToolCall(id="duplicate", name="read_fixture", input={"path": "b"})


class TooManyToolCallsProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        for index in range(129):
            yield ToolCall(id=f"tool-{index}", name="missing_tool", input={})


class OversizedToolArgumentsProvider(UnknownToolProvider):
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        yield ToolCall(
            id="tool-large",
            name="missing_tool",
            input={"value": "x" * 100},
        )


class EmptyProvider(UnknownToolProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        if False:
            yield ""


class ExplodingProvider(UnknownToolProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        raise RuntimeError("stream disconnected")
        yield ""


class TimeoutThenSuccessProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0.05)
        yield "recovered"


class StalledAfterOutputProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        yield "partial"
        await asyncio.sleep(0.05)
        yield "late"


class OversizedResponseProvider(UnknownToolProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        yield "1234"
        yield "5678"


class RateLimitedThenSuccessProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        if self.calls == 1:
            raise ProviderHTTPError(429, "rate limited")
        yield "recovered"


class ToolCaptureProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.received_tools: list[list[dict] | None] = []
        self.received_messages: list[list[Message]] = []

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.received_tools.append(tools)
        self.received_messages.append(messages)
        yield "direct answer"


class LargeToolResultProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.received_tools: list[list[dict] | None] = []

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        self.received_tools.append(tools)
        if self.calls == 1:
            yield ToolCall(
                id="tool-large-result",
                name="read_fixture",
                input={"path": "large.txt"},
            )
        else:
            yield "done"


class PrematureToolResultProvider(UnknownToolProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        yield ToolCall(
            id="tool-result-too-early",
            name="tool_result_read",
            input={"file_path": "missing.txt"},
        )


class MultiRoundUsageProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        self.last_usage = {
            "prompt_tokens": 10 * self.calls,
            "completion_tokens": 5,
            "total_tokens": 10 * self.calls + 5,
        }
        if self.calls == 1:
            yield ToolCall(id="tool-usage", name="missing_tool", input={})
        else:
            yield "done"


class ToolThenExplodeProvider(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        if self.calls == 1:
            yield ToolCall(id="tool-durable", name="write_fixture", input={"path": "x"})
            return
        raise RuntimeError("follow-up disconnected")
        yield ""


class AnthropicProviderStub(UnknownToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.config.protocol = "anthropic"


class ReadFixtureTool(BaseTool):
    def __init__(self) -> None:
        self.executed = False

    @property
    def name(self) -> str:
        return "read_fixture"

    @property
    def description(self) -> str:
        return "read fixture"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter("path", "string", "path")]

    async def execute(self, **kwargs) -> ToolResult:
        self.executed = True
        return ToolResult(success=True, content="read ok")


class LargeReadFixtureTool(ReadFixtureTool):
    async def execute(self, **kwargs) -> ToolResult:
        self.executed = True
        return ToolResult(success=True, content="x" * 100)


class WriteFixtureTool(ReadFixtureTool):
    @property
    def name(self) -> str:
        return "write_fixture"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WRITE


class InterceptingHookEngine:
    def __init__(self) -> None:
        self.events: list[HookEvent] = []

    async def fire(self, event: HookEvent, context=None):
        self.events.append(event)
        if event == HookEvent.TOOL_PRE_EXEC:
            return "blocked by hook"
        return None


class RecordingHookEngine:
    def __init__(self) -> None:
        self.calls = []

    async def fire(self, event: HookEvent, context=None):
        self.calls.append((event, dict(context or {})))
        return None


class MidTaskCompressor:
    def __init__(self):
        self.calls = 0

    async def check_and_compress(self, history, provider, *, extra_tokens=0):
        self.calls += 1
        if self.calls == 2:
            history.add_context_message("mid-task summary")
            return CompressionResult(
                was_compressed=True,
                estimated_tokens_before=1000,
                estimated_tokens_after=200,
            )
        return CompressionResult(
            estimated_tokens_before=100,
            estimated_tokens_after=100,
        )


class FailedOverflowCompressor:
    context_window = 100

    async def check_and_compress(self, history, provider, *, extra_tokens=0):
        return CompressionResult(
            estimated_tokens_before=101,
            estimated_tokens_after=101,
            error="summary unavailable",
        )


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def _make_loop(self, provider, **kwargs):
        return AgentLoop(
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=kwargs.pop("max_rounds", 1),
            **kwargs,
        )

    async def test_unknown_write_tool_returns_structured_failure_event(self):
        loop = AgentLoop(
            provider=UnknownToolProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("call a missing tool")

        events = [event async for event in loop.run(history)]
        tool_results = [event for event in events if isinstance(event, ToolResultEvent)]

        self.assertEqual(1, len(tool_results))
        self.assertFalse(tool_results[0].result.success)
        self.assertIn("未知工具", tool_results[0].result.error)

    async def test_direct_answer_turn_cannot_execute_unadvertised_tool_call(self):
        tool = ReadFixtureTool()
        registry = ToolRegistry()
        registry.register(tool)
        loop = AgentLoop(
            provider=ReadToolProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("写一个快速排序")

        events = [event async for event in loop.run(history)]

        self.assertFalse(tool.executed)
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual("unadvertised_tool_call", errors[0].code)

    async def test_new_direct_algorithm_request_does_not_send_previous_answer(self):
        provider = ToolCaptureProvider()
        loop = self._make_loop(provider)
        history = ConversationHistory()
        history.add_user_message("输出一个快速排序算法")
        history.add_assistant_message("previous quicksort answer")
        history.add_user_message("输出一个归并排序算法")

        _events = [event async for event in loop.run(history)]

        user_messages = [
            message["content"]
            for message in provider.received_messages[0]
            if message.get("role") == "user"
        ]
        self.assertEqual(["输出一个归并排序算法"], user_messages)
        self.assertNotIn("previous quicksort answer", str(provider.received_messages[0]))

    async def test_duplicate_tool_call_ids_fail_before_any_tool_executes(self):
        tool = ReadFixtureTool()
        registry = ToolRegistry()
        registry.register(tool)
        loop = AgentLoop(
            provider=DuplicateToolCallIdProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("检查当前项目")

        events = [event async for event in loop.run(history)]

        self.assertFalse(tool.executed)
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual("duplicate_tool_call_id", errors[0].code)

    async def test_tool_call_count_is_bounded_before_execution(self):
        loop = self._make_loop(TooManyToolCallsProvider())
        history = ConversationHistory()
        history.add_user_message("检查当前项目")

        events = [event async for event in loop.run(history)]
        errors = [event for event in events if isinstance(event, ErrorEvent)]

        self.assertEqual("too_many_tool_calls", errors[0].code)
        self.assertFalse(any(isinstance(event, ToolResultEvent) for event in events))

    async def test_tool_argument_bytes_share_the_response_size_limit(self):
        loop = self._make_loop(
            OversizedToolArgumentsProvider(),
            max_response_chars=20,
        )
        history = ConversationHistory()
        history.add_user_message("检查当前项目")

        events = [event async for event in loop.run(history)]
        errors = [event for event in events if isinstance(event, ErrorEvent)]

        self.assertEqual("response_too_large", errors[0].code)

    async def test_failed_compression_never_sends_overflowing_provider_request(self):
        provider = ToolCaptureProvider()
        loop = self._make_loop(
            provider,
            compressor=FailedOverflowCompressor(),
        )
        history = ConversationHistory()
        history.add_user_message("large request")

        events = [event async for event in loop.run(history)]

        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual("context_compression_failed", errors[0].code)
        self.assertEqual([], provider.received_tools)

    def test_system_prompt_snapshot_includes_effective_context(self):
        loop = AgentLoop(
            provider=UnknownToolProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            instructions_text="project instruction",
            environment_text="cwd: /workspace",
            max_rounds=1,
        )

        snapshot = loop.get_system_prompt()

        self.assertIn("protocol=openai", snapshot)
        self.assertIn("--- Base System Prompt ---", snapshot)
        self.assertIn("project instruction", snapshot)
        self.assertIn("cwd: /workspace", snapshot)

    def test_prompt_snapshot_does_not_consume_pending_injection(self):
        injector = PromptInjector()
        injector.queue_injection("one shot")
        loop = AgentLoop(
            provider=UnknownToolProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=injector,
            max_rounds=1,
        )

        preview = loop.get_system_prompt("injection")
        actual = injector.build_injection(1)

        self.assertIn("one shot", preview)
        self.assertIn("one shot", actual)

    async def test_token_usage_is_aggregated_across_react_rounds(self):
        provider = MultiRoundUsageProvider()
        loop = self._make_loop(provider, max_rounds=2)
        history = ConversationHistory()
        history.add_user_message("use a tool and finish")

        events = [event async for event in loop.run(history)]

        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))
        self.assertTrue(loop.turn_usage.available)
        self.assertEqual(40, loop.turn_usage.total_tokens)
        self.assertEqual(30, loop.turn_usage.input_tokens)
        self.assertEqual(10, loop.turn_usage.output_tokens)
        self.assertEqual(2, loop.turn_model_requests)

    async def test_context_is_rechecked_and_compressed_between_react_rounds(self):
        compressor = MidTaskCompressor()
        loop = self._make_loop(
            MultiRoundUsageProvider(), max_rounds=2, compressor=compressor,
        )
        history = ConversationHistory()
        history.add_user_message("use a tool and continue")

        events = [event async for event in loop.run(history)]

        self.assertEqual(2, compressor.calls)
        compression_events = [
            event for event in events if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(1, len(compression_events))
        self.assertTrue(compression_events[0].was_compressed)
        self.assertTrue(any(
            message.get("content") == "mid-task summary"
            for message in history.get_messages()
        ))

    async def test_completed_tool_history_survives_follow_up_provider_failure(self):
        registry = ToolRegistry()
        tool = WriteFixtureTool()
        registry.register(tool)
        loop = AgentLoop(
            provider=ToolThenExplodeProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=2,
        )
        history = ConversationHistory()
        history.add_user_message("write then explain")

        events = [event async for event in loop.run(history)]
        messages = history.get_messages()

        self.assertTrue(tool.executed)
        self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
        self.assertEqual(["user", "assistant", "tool"], [m["role"] for m in messages])
        self.assertEqual("tool-durable", messages[-1]["tool_call_id"])
        self.assertIn("read ok", messages[-1]["content"])

    async def test_direct_algorithm_request_does_not_offer_workspace_tools(self):
        provider = ToolCaptureProvider()
        registry = ToolRegistry()
        registry.register(ReadFixtureTool())
        loop = AgentLoop(
            provider=provider,
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("写一个快速排序")

        events = [event async for event in loop.run(history)]

        self.assertEqual([None], provider.received_tools)
        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))

    async def test_project_request_still_offers_workspace_tools(self):
        provider = ToolCaptureProvider()
        registry = ToolRegistry()
        registry.register(ReadFixtureTool())
        loop = AgentLoop(
            provider=provider,
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("读取 README.md 并总结这个项目")

        events = [event async for event in loop.run(history)]

        self.assertEqual(1, len(provider.received_tools))
        self.assertTrue(provider.received_tools[0])
        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))

    async def test_tool_result_helpers_are_exposed_only_after_result_is_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            provider = LargeToolResultProvider()
            registry = ToolRegistry()
            registry.register(LargeReadFixtureTool())
            registry.register(ToolResultSearchTool(storage_dir))
            registry.register(ToolResultReadTool(storage_dir))
            truncator = ToolResultTruncator(TruncateConfig(
                per_result_threshold=10,
                total_round_threshold=1_000,
                preview_length=5,
                storage_dir=storage_dir,
            ))
            loop = AgentLoop(
                provider=provider,
                tool_registry=registry,
                tool_executor=ToolExecutor(),
                prompt_builder=PromptBuilder(),
                prompt_injector=PromptInjector(),
                truncator=truncator,
                max_rounds=2,
            )
            history = ConversationHistory()
            history.add_user_message("读取项目里的大文件")

            events = [event async for event in loop.run(history)]

        first_names = self._openai_tool_names(provider.received_tools[0])
        second_names = self._openai_tool_names(provider.received_tools[1])
        self.assertNotIn("tool_result_search", first_names)
        self.assertNotIn("tool_result_read", first_names)
        self.assertIn("tool_result_search", second_names)
        self.assertIn("tool_result_read", second_names)
        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))

    async def test_hidden_tool_result_helper_cannot_be_called_prematurely(self):
        registry = ToolRegistry()
        registry.register(ToolResultReadTool())
        loop = AgentLoop(
            provider=PrematureToolResultProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("检查当前项目")

        events = [event async for event in loop.run(history)]

        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual(1, len(errors))
        self.assertEqual("unadvertised_tool_call", errors[0].code)
        self.assertIn("当前不可用", errors[0].message)

    def test_deferred_tool_filter_supports_anthropic_schemas(self):
        registry = ToolRegistry()
        registry.register(ToolResultReadTool())
        loop = AgentLoop(
            provider=AnthropicProviderStub(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
        )

        hidden = loop._build_tool_defs(include_tool_result_tools=False)
        visible = loop._build_tool_defs(include_tool_result_tools=True)

        self.assertEqual(set(), loop._tool_definition_names(hidden))
        self.assertEqual({"tool_result_read"}, loop._tool_definition_names(visible))

    def test_workspace_switch_moves_tool_result_cache_root(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            initial_storage = Path(first) / ".tinyCode" / "tool_results"
            truncator = ToolResultTruncator(TruncateConfig(
                storage_dir=initial_storage,
            ))
            registry = ToolRegistry()
            search_tool = ToolResultSearchTool(truncator.storage_dir)
            read_tool = ToolResultReadTool(truncator.storage_dir)
            registry.register(search_tool)
            registry.register(read_tool)
            loop = AgentLoop(
                provider=UnknownToolProvider(),
                tool_registry=registry,
                tool_executor=ToolExecutor(),
                prompt_builder=PromptBuilder(),
                prompt_injector=PromptInjector(),
                truncator=truncator,
            )

            loop.set_workspace(Path(second))

            expected = Path(second).resolve() / ".tinyCode" / "tool_results"
            self.assertEqual(expected, truncator.storage_dir)
            self.assertEqual(expected, search_tool.storage_dir)
            self.assertEqual(expected, read_tool.storage_dir)
            self.assertTrue(expected.is_dir())

    @staticmethod
    def _openai_tool_names(definitions):
        return {
            definition["function"]["name"]
            for definition in definitions or []
        }

    async def test_read_tool_pre_exec_hook_can_intercept_execution(self):
        registry = ToolRegistry()
        read_tool = ReadFixtureTool()
        registry.register(read_tool)
        hook_engine = InterceptingHookEngine()
        loop = AgentLoop(
            provider=ReadToolProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            hook_engine=hook_engine,
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("read something")

        events = [event async for event in loop.run(history)]
        tool_results = [event for event in events if isinstance(event, ToolResultEvent)]

        self.assertEqual(1, len(tool_results))
        self.assertFalse(tool_results[0].result.success)
        self.assertEqual("blocked by hook", tool_results[0].result.error)
        self.assertFalse(read_tool.executed)
        self.assertIn(HookEvent.TOOL_POST_EXEC, hook_engine.events)

    async def test_plan_only_blocked_write_tool_is_returned_to_history(self):
        registry = ToolRegistry()
        write_tool = WriteFixtureTool()
        registry.register(write_tool)
        loop = AgentLoop(
            provider=WriteToolProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        loop.toggle_plan_only()
        history = ConversationHistory()
        history.add_user_message("try to write")

        events = [event async for event in loop.run(history)]
        blocked_events = [event for event in events if isinstance(event, ToolBlockedEvent)]
        tool_messages = [msg for msg in history.get_messages() if msg.get("role") == "tool"]

        self.assertEqual(1, len(blocked_events))
        self.assertFalse(write_tool.executed)
        self.assertEqual(1, len(tool_messages))
        self.assertIn("Plan-only 模式已开启", tool_messages[0]["content"])

    async def test_malformed_tool_input_is_rejected_before_hooks(self):
        registry = ToolRegistry()
        write_tool = WriteFixtureTool()
        registry.register(write_tool)
        hook_engine = InterceptingHookEngine()
        loop = AgentLoop(
            provider=BadInputProvider(),
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            hook_engine=hook_engine,
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("try malformed params")

        events = [event async for event in loop.run(history)]
        tool_results = [event for event in events if isinstance(event, ToolResultEvent)]

        self.assertEqual(1, len(tool_results))
        self.assertFalse(tool_results[0].result.success)
        self.assertIn("工具调用参数必须是对象", tool_results[0].result.error)
        self.assertFalse(write_tool.executed)
        self.assertNotIn(HookEvent.TOOL_PRE_EXEC, hook_engine.events)

    async def test_invalid_provider_event_returns_error_event(self):
        loop = AgentLoop(
            provider=InvalidEventProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("provider emits malformed event")

        events = [event async for event in loop.run(history)]
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]

        self.assertEqual(1, len(errors))
        self.assertIn("Provider 输出事件必须是字符串或 ToolCall", errors[0].message)
        self.assertEqual([], tool_calls)

    async def test_invalid_tool_call_id_returns_error_event(self):
        loop = AgentLoop(
            provider=InvalidToolCallIdProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("provider emits malformed tool id")

        events = [event async for event in loop.run(history)]
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]

        self.assertEqual(1, len(errors))
        self.assertIn("ToolCall.id 必须是非空字符串", errors[0].message)
        self.assertEqual([], tool_calls)

    async def test_invalid_tool_call_name_returns_error_event(self):
        loop = AgentLoop(
            provider=InvalidToolCallNameProvider(),
            tool_registry=ToolRegistry(),
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            max_rounds=1,
        )
        history = ConversationHistory()
        history.add_user_message("provider emits malformed tool name")

        events = [event async for event in loop.run(history)]
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]

        self.assertEqual(1, len(errors))
        self.assertIn("ToolCall.name 必须是非空字符串", errors[0].message)
        self.assertEqual([], tool_calls)

    async def test_empty_provider_response_returns_error_event(self):
        hooks = RecordingHookEngine()
        loop = self._make_loop(EmptyProvider(), hook_engine=hooks)
        history = ConversationHistory()
        history.add_user_message("hello")

        events = [event async for event in loop.run(history)]

        self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
        self.assertIn(HookEvent.SYSTEM_ERROR, [event for event, _ in hooks.calls])
        round_end = [ctx for event, ctx in hooks.calls if event == HookEvent.ROUND_END]
        self.assertEqual("error", round_end[-1]["outcome"])

        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual(1, len(errors))
        self.assertIn("空响应", errors[0].message)

    async def test_provider_exception_is_converted_to_error_and_rolls_back_partial_history(self):
        loop = self._make_loop(ExplodingProvider())
        history = ConversationHistory()
        history.add_user_message("hello")

        events = [event async for event in loop.run(history)]

        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual(1, len(errors))
        self.assertIn("stream disconnected", errors[0].message)
        self.assertEqual([{"role": "user", "content": "hello"}], history.get_messages())

    async def test_oversized_response_is_stopped_and_history_is_rolled_back(self):
        loop = self._make_loop(OversizedResponseProvider(), max_response_chars=6)
        history = ConversationHistory()
        history.add_user_message("hello")

        events = [event async for event in loop.run(history)]

        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual(1, len(errors))
        self.assertIn("超过", errors[0].message)
        self.assertEqual([{"role": "user", "content": "hello"}], history.get_messages())

    async def test_deferred_background_context_survives_provider_rollback(self):
        loop = self._make_loop(ExplodingProvider())
        history = ConversationHistory()
        history.add_user_message("hello")
        history.defer_user_message("background result")

        events = [event async for event in loop.run(history)]

        self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
        self.assertEqual(
            ["hello", "background result"],
            [message["content"] for message in history.get_messages()],
        )

    async def test_each_react_round_emits_round_start(self):
        loop = self._make_loop(UnknownToolProvider(), max_rounds=2)
        history = ConversationHistory()
        history.add_user_message("call tool")

        events = [event async for event in loop.run(history)]
        rounds = [event for event in events if isinstance(event, RoundStartEvent)]

        self.assertEqual([1, 2], [event.round_number for event in rounds])
        self.assertTrue(all(event.max_rounds == 2 for event in rounds))

    async def test_runtime_max_rounds_applies_to_subsequent_turn(self):
        loop = self._make_loop(UnknownToolProvider(), max_rounds=1)
        self.assertEqual(1, loop.max_rounds)
        self.assertEqual(3, loop.set_max_rounds(3))
        history = ConversationHistory()
        history.add_user_message("call tool")

        events = [event async for event in loop.run(history)]
        rounds = [event for event in events if isinstance(event, RoundStartEvent)]

        self.assertEqual([1, 2, 3], [event.round_number for event in rounds])
        self.assertTrue(all(event.max_rounds == 3 for event in rounds))

    def test_runtime_max_rounds_rejects_invalid_values(self):
        loop = self._make_loop(UnknownToolProvider(), max_rounds=1)

        for value in (0, 101, True, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    loop.set_max_rounds(value)
        self.assertEqual(1, loop.max_rounds)

    async def test_first_event_timeout_retries_then_recovers(self):
        provider = TimeoutThenSuccessProvider()
        loop = self._make_loop(
            provider,
            first_event_timeout=0.01,
            idle_event_timeout=0.1,
            provider_retries=1,
            retry_delay=0,
        )
        history = ConversationHistory()
        history.add_user_message("hello")

        events = [event async for event in loop.run(history)]

        self.assertEqual(2, provider.calls)
        self.assertEqual(2, loop.turn_model_requests)
        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))
        self.assertEqual("recovered", history.get_messages()[-1]["content"])

    async def test_idle_timeout_after_output_does_not_retry_or_duplicate_text(self):
        provider = StalledAfterOutputProvider()
        loop = self._make_loop(
            provider,
            first_event_timeout=0.1,
            idle_event_timeout=0.01,
            provider_retries=2,
            retry_delay=0,
        )
        history = ConversationHistory()
        history.add_user_message("hello")

        events = [event async for event in loop.run(history)]
        errors = [event for event in events if isinstance(event, ErrorEvent)]

        self.assertEqual(1, provider.calls)
        self.assertEqual(1, loop.turn_model_requests)
        self.assertEqual(1, len(errors))
        self.assertIn("流式响应超时", errors[0].message)
        self.assertEqual([{"role": "user", "content": "hello"}], history.get_messages())

    async def test_retryable_http_error_recovers_before_any_output(self):
        provider = RateLimitedThenSuccessProvider()
        loop = self._make_loop(
            provider,
            provider_retries=1,
            retry_delay=0,
        )
        history = ConversationHistory()
        history.add_user_message("hello")

        events = [event async for event in loop.run(history)]

        self.assertEqual(2, provider.calls)
        self.assertEqual(2, loop.turn_model_requests)
        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))
        self.assertEqual("recovered", history.get_messages()[-1]["content"])

    def test_anthropic_messages_do_not_contain_system_roles(self):
        loop = self._make_loop(AnthropicProviderStub())
        history = ConversationHistory()
        history.add_context_message("compressed context")
        history.add_user_message("continue")

        messages = loop._assemble_messages(history, round_num=1)

        self.assertNotIn("system", [message.get("role") for message in messages])
        self.assertTrue(any(
            "compressed context" in str(message.get("content"))
            for message in messages
        ))


if __name__ == "__main__":
    unittest.main()

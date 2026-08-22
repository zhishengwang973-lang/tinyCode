import asyncio
import io
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from rich.console import Console

from tinyCode.agent.loop import AgentLoop
from tinyCode.config.models import ProviderConfig
from tinyCode.conversation.compression import CompressionResult
from tinyCode.conversation.history import ConversationHistory
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.providers.base import BaseProvider, Message, ToolCall
from tinyCode.security import PathSandbox, SecurityGuard, SecurityPolicy, SecurityLevel
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tui.app import TinyCodeTUI


class NoopCompressor:
    warning_threshold = 100_000
    context_window = 128_000

    async def check_and_compress(self, history, provider) -> CompressionResult:
        return CompressionResult()


class MemorySessionStore:
    def __init__(self) -> None:
        self.snapshots: list[list[Message]] = []

    def save(self, history, provider_name: str, model: str) -> None:
        self.snapshots.append(history.get_messages())


class SequenceProvider(BaseProvider):
    def __init__(self, responses=None) -> None:
        super().__init__(ProviderConfig(
            name="test",
            protocol="openai",
            model="test-model",
            base_url="https://example.invalid",
            api_key="test-key",
        ))
        self.responses = list(responses or [])
        self.calls = 0

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        self.calls += 1
        if self.responses:
            for event in self.responses.pop(0):
                yield event
            return
        latest_user = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        )
        yield f"reply:{latest_user}"

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = "",
    ) -> Message:
        return {
            "role": "assistant",
            "content": text_prefix or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": "{}"},
                }
                for call in tool_calls
            ],
        }

    def make_tool_result_message(
        self, tool_call_id: str, tool_name: str, result_text: str,
    ) -> Message:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result_text,
        }


class FixtureTool(BaseTool):
    def __init__(self, name: str, category: ToolCategory) -> None:
        self._name = name
        self._category = category
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "fixture tool"

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter("path", "string", "project-relative path")]

    async def execute(self, path: str) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, content=f"ok:{path}")


class DecisionPromptSession:
    def __init__(self) -> None:
        self.answers: list[str] = []

    async def prompt_async(self, message=None) -> str:
        if not self.answers:
            raise AssertionError("test did not provide an approval decision")
        return self.answers.pop(0)


class MultiTurnFlowTests(unittest.IsolatedAsyncioTestCase):
    def _make_tui(self, loop: AgentLoop, history: ConversationHistory) -> TinyCodeTUI:
        prompt_session = DecisionPromptSession()
        tui = TinyCodeTUI(
            agent_loop=loop,
            history=history,
            compressor=NoopCompressor(),
            session_store=MemorySessionStore(),
            note_manager=None,
            provider_name="test",
            model="test-model",
            console=Console(
                file=io.StringIO(),
                force_terminal=False,
                color_system=None,
            ),
            prompt_session=prompt_session,
        )
        tui._test_prompt_session = prompt_session
        return tui

    @staticmethod
    def _make_loop(provider, registry=None, guard=None) -> AgentLoop:
        return AgentLoop(
            provider=provider,
            tool_registry=registry or ToolRegistry(),
            tool_executor=ToolExecutor(default_timeout=1.0),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
            security_guard=guard,
            max_rounds=4,
        )

    async def test_eight_plain_turns_complete_and_keep_accepting_input(self):
        provider = SequenceProvider()
        history = ConversationHistory()
        tui = self._make_tui(self._make_loop(provider), history)

        for index in range(8):
            await tui._on_user_input(f"turn-{index}")
            self.assertFalse(tui._runtime.active)

        messages = history.get_messages()
        self.assertEqual(8, provider.calls)
        self.assertEqual(16, len(messages))
        self.assertEqual(
            ["user", "assistant"] * 8,
            [message["role"] for message in messages],
        )

    async def test_read_then_confirmed_write_flow_completes(self):
        provider = SequenceProvider(responses=[
            [ToolCall("read-1", "read_file", {"path": "README.md"})],
            ["read complete"],
            [ToolCall("write-1", "write_file", {"path": "result.txt"})],
            ["write complete"],
        ])
        registry = ToolRegistry()
        read_tool = FixtureTool("read_file", ToolCategory.READ)
        write_tool = FixtureTool("write_file", ToolCategory.WRITE)
        registry.register(read_tool)
        registry.register(write_tool)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = SecurityGuard(
                policy=SecurityPolicy(SecurityLevel.NORMAL, project_root=root),
                sandbox=PathSandbox(project_root=root),
            )
            history = ConversationHistory()
            tui = self._make_tui(
                self._make_loop(provider, registry=registry, guard=guard),
                history,
            )

            await tui._on_user_input("inspect")
            tui._test_prompt_session.answers.append("A")
            await tui._on_user_input("write")

        self.assertFalse(tui._runtime.active)
        self.assertEqual(1, read_tool.calls)
        self.assertEqual(1, write_tool.calls)
        self.assertEqual(4, provider.calls)
        self.assertEqual(
            ["user", "assistant", "tool", "assistant"] * 2,
            [message["role"] for message in history.get_messages()],
        )

    async def test_complex_task_survives_read_write_deny_and_follow_up_turns(self):
        provider = SequenceProvider(responses=[
            [
                ToolCall("glob-1", "glob", {"path": "src"}),
                ToolCall("grep-1", "grep", {"path": "TODO"}),
            ],
            ["inspection complete"],
            [ToolCall("edit-1", "edit_file", {"path": "app.py"})],
            ["change complete"],
            [ToolCall("cmd-1", "run_command", {"command": "tests"})],
            ["test skipped safely"],
            ["final answer"],
        ])
        registry = ToolRegistry()
        glob_tool = FixtureTool("glob", ToolCategory.READ)
        grep_tool = FixtureTool("grep", ToolCategory.READ)
        edit_tool = FixtureTool("edit_file", ToolCategory.WRITE)
        command_tool = FixtureTool("run_command", ToolCategory.WRITE)
        for tool in (glob_tool, grep_tool, edit_tool, command_tool):
            registry.register(tool)

        with tempfile.TemporaryDirectory() as tmp:
            guard = SecurityGuard(
                policy=SecurityPolicy(
                    SecurityLevel.NORMAL,
                    project_root=Path(tmp),
                ),
                sandbox=PathSandbox(project_root=Path(tmp)),
            )
            history = ConversationHistory()
            tui = self._make_tui(
                self._make_loop(provider, registry=registry, guard=guard),
                history,
            )

            await tui._on_user_input("inspect the project")

            tui._test_prompt_session.answers.append("A")
            await tui._on_user_input("apply the fix")

            tui._test_prompt_session.answers.append("D")
            await tui._on_user_input("run tests")

            await tui._on_user_input("summarize the result")

        self.assertFalse(tui._runtime.active)
        self.assertEqual(7, provider.calls)
        self.assertEqual((1, 1, 1, 0), (
            glob_tool.calls,
            grep_tool.calls,
            edit_tool.calls,
            command_tool.calls,
        ))
        self.assertEqual(
            [
                "user", "assistant", "tool", "tool", "assistant",
                "user", "assistant", "tool", "assistant",
                "user", "assistant", "tool", "assistant",
                "user", "assistant",
            ],
            [message["role"] for message in history.get_messages()],
        )


if __name__ == "__main__":
    unittest.main()

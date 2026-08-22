import unittest

from tinyCode.agent.events import AgentDoneEvent, ErrorEvent, RoundStartEvent
from tinyCode.agent.loop import AgentLoop
from tinyCode.config.models import ProviderConfig
from tinyCode.conversation.history import ConversationHistory
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.providers.base import BaseProvider, Message, ToolCall
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.registry import ToolRegistry


class StepTool(BaseTool):
    name = property(lambda self: "step")
    description = property(lambda self: "execute one deterministic task step")
    category = property(lambda self: ToolCategory.READ)
    parameters = property(lambda self: [ToolParameter("index", "integer", "step index")])

    async def execute(self, index: int) -> ToolResult:
        return ToolResult(success=True, content=f"step {index} complete")


class LongTaskProvider(BaseProvider):
    def __init__(self, tool_rounds: int = 15) -> None:
        super().__init__(ProviderConfig(
            name="long-task",
            protocol="openai",
            model="test-model",
            api_key="test-key",
        ))
        self.tool_rounds = tool_rounds
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        if self.calls <= self.tool_rounds:
            yield ToolCall(
                id=f"call-{self.calls}",
                name="step",
                input={"index": self.calls},
            )
        else:
            yield "all steps complete"

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


class ComplexTaskFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_runtime_completes_fifteen_tool_rounds(self):
        provider = LongTaskProvider(tool_rounds=15)
        registry = ToolRegistry()
        registry.register(StepTool())
        loop = AgentLoop(
            provider=provider,
            tool_registry=registry,
            tool_executor=ToolExecutor(),
            prompt_builder=PromptBuilder(),
            prompt_injector=PromptInjector(),
        )
        history = ConversationHistory()
        history.add_user_message("complete a long task")

        events = [event async for event in loop.run(history)]

        self.assertFalse(any(isinstance(event, ErrorEvent) for event in events))
        self.assertEqual(16, sum(isinstance(event, RoundStartEvent) for event in events))
        done = [event for event in events if isinstance(event, AgentDoneEvent)]
        self.assertEqual(["no_tool_call"], [event.reason for event in done])
        self.assertEqual(16, provider.calls)
        self.assertEqual("all steps complete", history.get_messages()[-1]["content"])


if __name__ == "__main__":
    unittest.main()

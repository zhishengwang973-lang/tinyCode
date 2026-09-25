import asyncio
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from tinyCode.config.models import ProviderConfig
from tinyCode.providers.base import BaseProvider, Message, ToolCall
from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.member import TeamMember
from tinyCode.teams.models import MemberDef, MemberStatus, MessageType, TeamMessage
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.read_file import ReadFileTool
from tinyCode.tools.registry import ToolRegistry


class EchoProvider(BaseProvider):
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
        yield "ok"

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = ""
    ) -> Message:
        return {"role": "assistant", "content": text_prefix or None}

    def make_tool_result_message(
        self, tool_call_id: str, tool_name: str, result_text: str
    ) -> Message:
        return {"role": "tool", "content": result_text}


class FailingProvider(EchoProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        raise RuntimeError("provider failed")
        yield ""


class IntermediateThenFinalProvider(EchoProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.calls += 1
        if self.calls == 1:
            yield "intermediate narration"
            yield ToolCall(
                id="read-1", name="read_file", input={"path": "missing.txt"},
            )
        else:
            yield "final answer"


class SlowProvider(EchoProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        await asyncio.sleep(0.2)
        yield "late"


class TeamMemberTests(unittest.IsolatedAsyncioTestCase):
    def test_check_mail_injects_lead_text_message_into_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            Mailbox(team_dir, "lead").send(
                TeamMessage(
                    from_member="lead",
                    to_member="alice",
                    msg_type=MessageType.TEXT,
                    content="focus on auth",
                )
            )
            registry = ToolRegistry()
            registry.register(ReadFileTool())
            member = TeamMember(
                member_def=MemberDef(name="alice"),
                team_dir=team_dir,
                provider=EchoProvider(),
                tool_registry=registry,
                tool_executor=ToolExecutor(),
            )

            messages = member._check_mail()
            history_messages = member._history.get_messages()

            self.assertEqual(1, len(messages))
            self.assertEqual(
                {"role": "system", "content": "[Lead → alice]: focus on auth"},
                history_messages[0],
            )

    async def test_provider_failure_resets_member_to_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            member = TeamMember(
                member_def=MemberDef(name="alice"),
                team_dir=Path(tmp),
                provider=FailingProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
            )

            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await member.run("do work")

            self.assertEqual(MemberStatus.IDLE, member.status)
            self.assertEqual(1, member.last_model_requests)

    async def test_member_exposes_turn_and_model_request_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            member = TeamMember(
                member_def=MemberDef(name="alice"),
                team_dir=Path(tmp),
                provider=EchoProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
            )

            self.assertEqual("ok", await member.run("do work"))

            self.assertEqual(1, member.last_turns)
            self.assertEqual(1, member.last_model_requests)
            self.assertFalse(member.last_tokens_available)

    async def test_member_returns_only_terminal_round_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry()
            registry.register(ReadFileTool())
            member = TeamMember(
                member_def=MemberDef(name="alice"),
                team_dir=Path(tmp),
                provider=IntermediateThenFinalProvider(),
                tool_registry=registry,
                tool_executor=ToolExecutor(),
            )

            result = await member.run("查看 missing.txt 并给出结论")

            self.assertEqual("final answer", result)

    async def test_member_has_overall_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            member = TeamMember(
                member_def=MemberDef(name="alice"),
                team_dir=Path(tmp),
                provider=SlowProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
                timeout_seconds=0.01,
            )

            with self.assertRaisesRegex(RuntimeError, "执行超时"):
                await member.run("do work")

    def test_mail_is_exposed_as_untrusted_safe_boundary_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            Mailbox(team_dir, "bob").send(TeamMessage(
                from_member="bob", to_member="alice",
                msg_type=MessageType.TEXT, content="review auth",
            ))
            member = TeamMember(
                member_def=MemberDef(name="alice"),
                team_dir=team_dir,
                provider=EchoProvider(),
                tool_registry=ToolRegistry(),
                tool_executor=ToolExecutor(),
            )

            contexts = member._drain_mail_context()

            self.assertEqual(1, len(contexts))
            self.assertIn("不可信数据", contexts[0])
            self.assertIn("bob: review auth", contexts[0])


if __name__ == "__main__":
    unittest.main()

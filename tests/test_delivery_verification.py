import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from tinyCode.config.models import DeliveryVerificationConfig, ProviderConfig
from tinyCode.goals.models import Goal, GoalStatus
from tinyCode.providers.base import BaseProvider, Message
from tinyCode.tui.workspace_changes import WorkspaceSnapshot
from tinyCode.verification import DeliveryVerifier, ToolEvidence


class FakeVerifierProvider(BaseProvider):
    def __init__(self, response: str | list[str]) -> None:
        super().__init__(ProviderConfig(
            name="verifier", protocol="openai", model="verifier-model", api_key="key",
        ))
        self.response = response
        self.requests: list[list[Message]] = []

    async def chat_stream(
        self, messages: list[Message], tools=None, system_blocks=None,
    ) -> AsyncIterator[str]:
        del tools, system_blocks
        self.requests.append(messages)
        if isinstance(self.response, list):
            for chunk in self.response:
                yield chunk
        else:
            yield self.response


class DeliveryVerifierTests(unittest.IsolatedAsyncioTestCase):
    def _verifier(self, provider: FakeVerifierProvider) -> DeliveryVerifier:
        return DeliveryVerifier(DeliveryVerificationConfig(
            enabled=True,
            provider="verifier",
            min_goal_chars=10,
            min_tool_calls=1,
            min_changed_files=1,
            timeout_seconds=5,
        ), provider)

    @staticmethod
    def _completed_goal() -> Goal:
        goal = Goal.create("session", "实现一个复杂 healthcheck 模块并补齐测试", max_turns=4)
        goal.status = GoalStatus.COMPLETED
        return goal

    async def test_verifier_receives_bounded_observable_evidence_and_returns_verdict(self):
        provider = FakeVerifierProvider(
            '{"verdict":"pass","rationale":"测试与 diff 均支持交付",'
            '"requirements_met":["测试通过"],"missing_or_risks":[]}',
        )
        verifier = self._verifier(provider)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "health.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            snapshot = await verifier.capture_workspace(root)
            target.write_text("VALUE = 2\n", encoding="utf-8")
            changes = snapshot.fingerprints.compare()

            verdict = await verifier.verify(
                self._completed_goal(),
                final_answer="主 Agent 声称已完成",
                tool_evidence=[ToolEvidence("run_command", True, "pytest: 10 passed")],
                snapshot=snapshot,
                changes=changes,
            )

        self.assertTrue(verdict.available)
        self.assertEqual("pass", verdict.verdict)
        payload = provider.requests[0][0]["content"]
        self.assertIn("final_answer_untrusted", payload)
        self.assertIn("VALUE = 2", payload)
        self.assertIn("pytest: 10 passed", payload)

    async def test_simple_or_incomplete_goal_is_not_verified(self):
        provider = FakeVerifierProvider("{}")
        verifier = self._verifier(provider)
        simple = Goal.create("session", "总结", max_turns=1)
        self.assertFalse(verifier.should_track(simple))
        self.assertFalse(verifier.should_verify(
            self._completed_goal(), tool_calls=0, changes=None,
        ))

    async def test_invalid_judge_response_is_reported_without_raising(self):
        provider = FakeVerifierProvider("not-json")
        verifier = self._verifier(provider)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = await verifier.capture_workspace(root)
            verdict = await verifier.verify(
                self._completed_goal(), final_answer="done", tool_evidence=[],
                snapshot=snapshot, changes=WorkspaceSnapshot.capture(root).compare(),
            )

        self.assertFalse(verdict.available)
        self.assertIn("JSON", verdict.error)

    async def test_reasoning_stream_is_not_counted_as_verdict_output(self):
        provider = FakeVerifierProvider([
            "<<REASONING:" + ("逐步分析" * 8_000) + ">>",
            '{"verdict":"needs_review","rationale":"缺少完整测试证据",'
            '"requirements_met":[],"missing_or_risks":["请人工检查"]}',
        ])
        verifier = self._verifier(provider)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = await verifier.capture_workspace(root)
            verdict = await verifier.verify(
                self._completed_goal(), final_answer="done", tool_evidence=[],
                snapshot=snapshot, changes=WorkspaceSnapshot.capture(root).compare(),
            )

        self.assertTrue(verdict.available)
        self.assertEqual("needs_review", verdict.verdict)

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinyCode.agent.task_mode import (
    TaskMode,
    classify_task_mode_rule,
)
from tinyCode.agent.task_mode_router import (
    JevTaskModeClient,
    JevTaskModeDecision,
    TaskModeRouter,
)
from tinyCode.config.models import ProviderConfig, TaskModeRoutingConfig
from tinyCode.providers.base import BaseProvider, Message, TokenUsage, ToolCall


class RoutingProvider(BaseProvider):
    def __init__(self, response: str = "modify") -> None:
        super().__init__(ProviderConfig(
            name="test", protocol="openai", model="test-model",
            base_url="https://example.test", api_key="test-key",
        ))
        self.response = response
        self.calls = 0

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ):
        self.calls += 1
        self.last_usage = {
            "prompt_tokens": 12,
            "completion_tokens": 1,
            "total_tokens": 13,
        }
        yield self.response

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = "",
    ) -> Message:
        return {"role": "assistant", "content": text_prefix}

    def make_tool_result_message(
        self, tool_call_id: str, tool_name: str, result_text: str,
    ) -> Message:
        return {"role": "tool", "content": result_text}


class FakeJev:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def classify(self, state):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _config(**updates) -> TaskModeRoutingConfig:
    values = {
        "enabled": True,
        "api_key": "jev-key",
        "confidence_threshold": 0.8,
        "timeout_seconds": 1.0,
        "llm_timeout_seconds": 1.0,
        "llm_fallback": True,
    }
    values.update(updates)
    return TaskModeRoutingConfig(**values)


class TaskModeRuleConfidenceTests(unittest.TestCase):
    def test_clear_rules_are_decisive(self):
        cases = {
            "写一个快速排序": TaskMode.DIRECT,
            "检查当前项目": TaskMode.INSPECT,
            "修复当前项目": TaskMode.MODIFY,
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                decision = classify_task_mode_rule([
                    {"role": "user", "content": prompt},
                ])
                self.assertIs(expected, decision.mode)
                self.assertTrue(decision.decisive)

    def test_fallthrough_request_is_ambiguous(self):
        decision = classify_task_mode_rule([
            {"role": "user", "content": "这个能做吗"},
        ])

        self.assertFalse(decision.decisive)


class TaskModeRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_jev_client_uses_official_choice_contract(self):
        captured = {}

        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {
                    "answers": {
                        "task_mode": {
                            "type": "choice",
                            "choice": "inspect",
                            "confidence": 0.9,
                            "probabilities": {
                                "direct": 0.05,
                                "inspect": 0.9,
                                "modify": 0.05,
                            },
                        }
                    },
                    "usage": {"input_tokens": 8, "output_tokens": 3},
                }

        class Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, endpoint, **kwargs):
                captured["endpoint"] = endpoint
                captured["request"] = kwargs
                return Response()

        client = JevTaskModeClient(_config())
        with patch(
            "tinyCode.agent.task_mode_router.detect_proxy_route",
            return_value=SimpleNamespace(trust_env=False),
        ), patch(
            "tinyCode.agent.task_mode_router.httpx.AsyncClient", Client,
        ):
            decision = await client.classify({"latest_user_request": "看看这个"})

        self.assertIs(TaskMode.INSPECT, decision.mode)
        self.assertEqual(0.9, decision.confidence)
        self.assertEqual(11, decision.usage.total_tokens)
        self.assertEqual(
            "https://api.typesafe.ai/v1/systemone", captured["endpoint"],
        )
        self.assertFalse(captured["client"]["trust_env"])
        question = captured["request"]["json"]["questions"]["task_mode"]
        self.assertEqual("choice", question["type"])
        self.assertEqual(
            {"direct", "inspect", "modify"}, set(question["criteria"]),
        )

    async def test_decisive_rule_does_not_call_jev_or_llm(self):
        provider = RoutingProvider()
        jev = FakeJev(error=AssertionError("Jev should not run"))
        router = TaskModeRouter(provider, _config(), jev_client=jev)

        result = await router.route([
            {"role": "user", "content": "写一个快速排序"},
        ])

        self.assertIs(TaskMode.DIRECT, result.mode)
        self.assertEqual("rule", result.source)
        self.assertEqual(0, result.model_requests)
        self.assertEqual(0, jev.calls)
        self.assertEqual(0, provider.calls)

    async def test_high_confidence_jev_decision_is_used(self):
        provider = RoutingProvider()
        jev = FakeJev(JevTaskModeDecision(
            mode=TaskMode.INSPECT,
            confidence=0.91,
            probabilities={"direct": 0.04, "inspect": 0.91, "modify": 0.05},
            usage=TokenUsage(20, 3, 23, True),
        ))
        router = TaskModeRouter(provider, _config(), jev_client=jev)

        result = await router.route([
            {"role": "user", "content": "这个能做吗"},
        ])

        self.assertIs(TaskMode.INSPECT, result.mode)
        self.assertEqual("jev", result.source)
        self.assertEqual(0.91, result.confidence)
        self.assertEqual(1, result.model_requests)
        self.assertEqual(23, result.usage.total_tokens)
        self.assertEqual(0, provider.calls)

    async def test_low_confidence_jev_uses_generative_fallback(self):
        provider = RoutingProvider("modify")
        jev = FakeJev(JevTaskModeDecision(
            mode=TaskMode.DIRECT,
            confidence=0.55,
            probabilities={"direct": 0.55, "inspect": 0.2, "modify": 0.25},
            usage=TokenUsage(10, 2, 12, True),
        ))
        router = TaskModeRouter(provider, _config(), jev_client=jev)

        result = await router.route([
            {"role": "user", "content": "这个能做吗"},
        ])

        self.assertIs(TaskMode.MODIFY, result.mode)
        self.assertEqual("llm_fallback", result.source)
        self.assertEqual(2, result.model_requests)
        self.assertEqual(25, result.usage.total_tokens)
        self.assertIn("confidence", result.error)

    async def test_both_semantic_layers_failing_keeps_rule_baseline(self):
        provider = RoutingProvider("not-a-mode")
        jev = FakeJev(error=TimeoutError("unavailable"))
        router = TaskModeRouter(provider, _config(), jev_client=jev)

        result = await router.route([
            {"role": "user", "content": "这个能做吗"},
        ])

        self.assertIs(TaskMode.DIRECT, result.mode)
        self.assertEqual("rule_fallback", result.source)
        self.assertEqual(2, result.model_requests)
        self.assertIn("Jev TimeoutError", result.error)
        self.assertIn("LLM RuntimeError", result.error)


if __name__ == "__main__":
    unittest.main()

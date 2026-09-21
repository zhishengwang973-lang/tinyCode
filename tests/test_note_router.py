import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinyCode.config.models import NoteRoutingConfig
from tinyCode.notes.router import JevNoteRouter


def _config(**updates) -> NoteRoutingConfig:
    values = {
        "enabled": True,
        "api_key": "jev-key",
        "confidence_threshold": 0.85,
        "timeout_seconds": 1.0,
    }
    values.update(updates)
    return NoteRoutingConfig(**values)


class NoteRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_one_request_with_four_noul_questions(self):
        captured = {}

        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {
                    "answers": {
                        "user_preferences": {"type": "noul", "noul": 0.03},
                        "corrections": {"type": "noul", "noul": 0.02},
                        "project_knowledge": {"type": "noul", "noul": 0.96},
                        "references": {"type": "noul", "noul": 0.01},
                    },
                    "usage": {"input_tokens": 20, "output_tokens": 4},
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

        router = JevNoteRouter(_config())
        with patch(
            "tinyCode.notes.router.detect_proxy_route",
            return_value=SimpleNamespace(trust_env=False),
        ), patch(
            "tinyCode.notes.router.httpx.AsyncClient", Client,
        ):
            decision = await router.route("项目改用 Textual")

        self.assertEqual(frozenset({"项目知识"}), decision.categories)
        self.assertEqual(24, decision.usage.total_tokens)
        self.assertEqual(
            "https://api.typesafe.ai/v1/systemone", captured["endpoint"],
        )
        self.assertFalse(captured["client"]["trust_env"])
        request = captured["request"]
        self.assertEqual("Bearer jev-key", request["headers"]["Authorization"])
        questions = request["json"]["questions"]
        self.assertEqual(4, len(questions))
        self.assertTrue(all(item["type"] == "noul" for item in questions.values()))

    async def test_uncertain_answer_rejects_gate(self):
        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {
                    "answers": {
                        "user_preferences": {"type": "noul", "noul": 0.01},
                        "corrections": {"type": "noul", "noul": 0.5},
                        "project_knowledge": {"type": "noul", "noul": 0.99},
                        "references": {"type": "noul", "noul": 0.01},
                    }
                }

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, endpoint, **kwargs):
                return Response()

        router = JevNoteRouter(_config())
        with patch(
            "tinyCode.notes.router.detect_proxy_route",
            return_value=SimpleNamespace(trust_env=True),
        ), patch(
            "tinyCode.notes.router.httpx.AsyncClient",
            return_value=Client(),
        ):
            with self.assertRaisesRegex(RuntimeError, "置信度不足"):
                await router.route("用户纠正了项目配置")


if __name__ == "__main__":
    unittest.main()

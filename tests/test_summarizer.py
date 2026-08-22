import unittest

from tinyCode.config.models import ProviderConfig
from tinyCode.conversation.compression import ContextCompressor
from tinyCode.conversation.history import ConversationHistory
from tinyCode.conversation.summarizer import (
    MAX_SUMMARY_OUTPUT_CHARS,
    StructuredSummarizer,
    _format_for_summary,
)
from tinyCode.providers.base import BaseProvider, ToolCall


class FakeProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, response: str = "summary") -> None:
        super().__init__(config)
        self.response = response
        self.prompts: list[str] = []

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        self.prompts.append(messages[0]["content"])
        yield self.response

    def make_tool_calls_message(self, tool_calls: list[ToolCall], text_prefix: str = ""):
        return {"role": "assistant", "content": text_prefix, "tool_calls": []}

    def make_tool_result_message(self, tool_call_id: str, tool_name: str, result_text: str):
        return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result_text}


class FailingSummaryProvider(FakeProvider):
    async def chat_stream(self, messages, tools=None, system_blocks=None):
        raise RuntimeError("summary unavailable")
        yield ""  # pragma: no cover - keep this an async generator


class StructuredSummarizerTests(unittest.IsolatedAsyncioTestCase):
    def test_configured_context_window_overrides_model_table(self):
        provider = FakeProvider(
            ProviderConfig(
                name="custom", protocol="openai", model="gpt-4.1",
                api_key="test-key", context_window=32_768,
            )
        )

        summarizer = StructuredSummarizer(provider, model="gpt-4.1")

        self.assertEqual(32_768, summarizer.context_window)
        self.assertEqual(int(32_768 * 0.7), summarizer.trigger_threshold)

    def test_summary_formatter_preserves_included_user_message_verbatim(self):
        exact = "用户原话：请保留空格  and punctuation!?\n第二行"

        formatted = _format_for_summary([
            {"role": "assistant", "content": "x" * 10_000},
            {"role": "user", "content": exact},
        ], max_chars=2_000)

        self.assertIn(exact, formatted)

    def test_summary_formatter_honors_total_bound_for_many_messages(self):
        messages = [
            {"role": "assistant", "content": "x" * 100}
            for _ in range(1_000)
        ]

        formatted = _format_for_summary(messages, max_chars=2_000)

        self.assertLessEqual(len(formatted), 2_000)
        self.assertIn("消息被整体省略", formatted)

    def test_deepseek_models_use_one_million_token_context_window(self):
        provider = FakeProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-v4-flash",
                api_key="test-key",
            )
        )

        for model in (
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-chat",
            "deepseek-reasoner",
        ):
            with self.subTest(model=model):
                summarizer = StructuredSummarizer(provider, model=model)
                self.assertEqual(1_000_000, summarizer.context_window)
                self.assertEqual(700_000, summarizer.trigger_threshold)

    async def test_compression_reports_before_and_after_context_estimates(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake",
                protocol="openai",
                model="gpt-test",
                api_key="test-key",
            ),
            response="## 主要请求\n- 保留摘要",
        )
        compressor = ContextCompressor(model="gpt-3.5-turbo", provider=provider)
        history = ConversationHistory()
        for index in range(8):
            role = "user" if index % 2 == 0 else "assistant"
            history.add_raw_message({"role": role, "content": "x" * 6_000})

        result = await compressor.check_and_compress(history, provider)

        self.assertTrue(result.was_compressed)
        self.assertGreater(result.estimated_tokens_before, result.estimated_tokens_after)
        self.assertEqual(
            result.estimated_tokens_after,
            history.estimated_token_count(),
        )

    async def test_static_prompt_overhead_can_trigger_compression(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake", protocol="openai", model="small",
                api_key="test-key", context_window=1_024,
            ),
            response="## 主要请求\n保留",
        )
        compressor = ContextCompressor(model="small", provider=provider)
        history = ConversationHistory()
        for index in range(8):
            history.add_raw_message({
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"short-{index}",
            })

        result = await compressor.check_and_compress(
            history, provider, extra_tokens=900,
        )

        self.assertTrue(result.model_request_made)
        self.assertTrue(result.was_compressed)

    async def test_open_circuit_blocks_request_once_context_reaches_hard_window(self):
        provider = FailingSummaryProvider(
            ProviderConfig(
                name="fake", protocol="openai", model="small",
                api_key="test-key", context_window=1_024,
            )
        )
        compressor = ContextCompressor(model="small", provider=provider)
        history = ConversationHistory()
        for index in range(6):
            history.add_raw_message({
                "role": "user" if index % 2 == 0 else "assistant",
                "content": "x" * 600,
            })

        first = await compressor.check_and_compress(history, provider)
        second = await compressor.check_and_compress(history, provider)
        blocked = await compressor.check_and_compress(
            history, provider, extra_tokens=1_024,
        )

        self.assertIn("摘要生成失败", first.error)
        self.assertIn("摘要生成失败", second.error)
        self.assertTrue(compressor.circuit_open)
        self.assertIn("已熔断", blocked.error)
        self.assertFalse(blocked.model_request_made)

    async def test_oversized_summary_output_is_rejected(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake", protocol="openai", model="small",
                api_key="test-key",
            ),
            response="x" * (MAX_SUMMARY_OUTPUT_CHARS + 1),
        )
        summarizer = StructuredSummarizer(provider, model="gpt-3.5-turbo")
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": "old"}
            for index in range(8)
        ]

        _messages, result = await summarizer.summarize(messages)

        self.assertIn("字符上限", result.error)

    def test_needs_summary_counts_large_tool_call_arguments(self):
        summarizer = StructuredSummarizer(
            FakeProvider(
                ProviderConfig(
                    name="fake",
                    protocol="openai",
                    model="gpt-test",
                    api_key="test-key",
                )
            ),
            model="gpt-3.5-turbo",
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": "x" * 50_000,
                        },
                    }
                ],
            }
        ]

        self.assertTrue(summarizer.needs_summary(messages))

    async def test_summarize_includes_tool_calls_in_summary_prompt(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake",
                protocol="openai",
                model="gpt-test",
                api_key="test-key",
            )
        )
        summarizer = StructuredSummarizer(provider, model="gpt-3.5-turbo")
        messages = [
            {"role": "user", "content": "please write a file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": "{\"path\":\"important.py\"}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]

        await summarizer.summarize(messages)

        self.assertIn("write_file", provider.prompts[0])
        self.assertIn("important.py", provider.prompts[0])

    async def test_summarize_preserves_formal_summary_code_blocks(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake",
                protocol="openai",
                model="gpt-test",
                api_key="test-key",
            ),
            response='## 文件与代码\n```python\nprint("hi")\n```\n## 下一步\n继续',
        )
        summarizer = StructuredSummarizer(provider, model="gpt-3.5-turbo")
        messages = [
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]

        _, result = await summarizer.summarize(messages)

        self.assertIn("## 文件与代码", result.summary_text)
        self.assertIn("```python", result.summary_text)
        self.assertIn('print("hi")', result.summary_text)

    async def test_summarize_does_not_split_openai_tool_exchange(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake", protocol="openai", model="gpt-test", api_key="test-key",
            )
        )
        summarizer = StructuredSummarizer(provider, model="gpt-3.5-turbo")
        messages = [
            {"role": "user", "content": "old 1"},
            {"role": "assistant", "content": "old 2"},
            {"role": "user", "content": "run tools"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "one"},
            {"role": "tool", "tool_call_id": "call_2", "content": "two"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ]

        new_messages, result = await summarizer.summarize(messages)

        self.assertGreater(result.messages_compressed, 0)
        retained = new_messages[2:]
        self.assertEqual("user", retained[0]["role"])
        self.assertEqual("assistant", retained[1]["role"])
        self.assertEqual("call_1", retained[1]["tool_calls"][0]["id"])
        self.assertEqual(["call_1", "call_2"], [m["tool_call_id"] for m in retained[2:4]])

    async def test_summarize_does_not_split_anthropic_tool_exchange(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake", protocol="anthropic", model="claude-test", api_key="test-key",
            )
        )
        summarizer = StructuredSummarizer(provider, model="gpt-3.5-turbo")
        messages = [
            {"role": "user", "content": "old 1"},
            {"role": "assistant", "content": "old 2"},
            {"role": "user", "content": "run tool"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "answer"},
        ]

        new_messages, result = await summarizer.summarize(messages)

        self.assertGreater(result.messages_compressed, 0)
        retained = new_messages[2:]
        self.assertEqual("user", retained[0]["role"])
        self.assertEqual("assistant", retained[1]["role"])
        self.assertEqual("tool_use", retained[1]["content"][0]["type"])
        self.assertEqual("tool_result", retained[2]["content"][0]["type"])

    async def test_short_oversized_history_reports_why_it_cannot_compress(self):
        provider = FakeProvider(
            ProviderConfig(
                name="fake", protocol="openai", model="small",
                api_key="test-key", context_window=1_024,
            )
        )
        compressor = ContextCompressor(model="small", provider=provider)
        history = ConversationHistory()
        history.add_user_message("x" * 10_000)

        result = await compressor.check_and_compress(history, provider)

        self.assertFalse(result.was_compressed)
        self.assertIn("没有可安全压缩", result.error)


if __name__ == "__main__":
    unittest.main()

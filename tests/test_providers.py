import json
import asyncio
import unittest
from unittest.mock import patch

from tinyCode.config.models import ProviderConfig
from tinyCode.providers.anthropic import AnthropicProvider
from tinyCode.providers.base import (
    ProviderError,
    ProviderHTTPError,
    CacheUsage,
    TokenUsage,
    ToolCall,
    build_api_url,
)
from tinyCode.providers.deepseek import DeepSeekProvider
from tinyCode.providers.openai import OpenAIProvider
from tinyCode.providers.sse import SSEDecoder


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self._lines = lines

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeAsyncClient:
    lines: list[str] = []
    last_stream_kwargs: dict = {}
    instances = 0
    close_calls = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1

    async def aclose(self):
        type(self).close_calls += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def stream(self, *args, **kwargs):
        type(self).last_stream_kwargs = kwargs
        return _FakeStream(_FakeResponse(self.lines))


async def _collect(stream):
    return [item async for item in stream]


class ProviderUrlTests(unittest.TestCase):
    def test_versioned_base_url_is_not_duplicated(self):
        self.assertEqual(
            "http://localhost:11434/v1/chat/completions",
            build_api_url(
                "http://localhost:11434/v1",
                "/v1/chat/completions",
            ),
        )

    def test_full_endpoint_is_kept(self):
        endpoint = "https://gateway.example/v1/chat/completions"
        self.assertEqual(endpoint, build_api_url(endpoint, "/v1/chat/completions"))

    def test_oversized_sse_event_is_rejected_before_json_decode(self):
        with patch("tinyCode.providers.sse.MAX_SSE_EVENT_CHARS", 16):
            with self.assertRaises(ProviderError) as raised:
                SSEDecoder().parse('data: {"content":"too large"}')

        self.assertEqual("sse_event_too_large", raised.exception.code)

    def test_one_malformed_event_invalidates_partially_parsed_stream(self):
        decoder = SSEDecoder()
        decoder.parse('data: {"ok":true}')
        decoder.parse("data: {broken")

        with self.assertRaises(ProviderError) as raised:
            decoder.validate()

        self.assertEqual("malformed_sse", raised.exception.code)


class ProviderRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_error_body_is_streamed_with_a_memory_bound(self):
        class ErrorResponse:
            status_code = 502
            chunks_read = 0

            async def aiter_bytes(self):
                type(self).chunks_read += 1
                yield b"x" * 100_000
                raise AssertionError("bounded reader consumed the remaining body")

        class ErrorClient:
            def stream(self, *args, **kwargs):
                return _FakeStream(ErrorResponse())

        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )
        provider._client = ErrorClient()

        with self.assertRaises(ProviderHTTPError) as raised:
            await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(502, raised.exception.status_code)
        self.assertEqual(1, ErrorResponse.chunks_read)

    async def test_usage_is_isolated_between_concurrent_tasks(self):
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )
        ready = asyncio.Event()
        arrived = 0
        lock = asyncio.Lock()

        async def worker(total):
            nonlocal arrived
            provider.begin_request()
            provider.last_usage = {"total_tokens": total}
            async with lock:
                arrived += 1
                if arrived == 2:
                    ready.set()
            await ready.wait()
            return dict(provider.last_usage)

        first, second = await asyncio.gather(worker(11), worker(22))

        self.assertEqual(11, first["total_tokens"])
        self.assertEqual(22, second["total_tokens"])

    async def test_http_client_is_reused_and_closed(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        start_instances = _FakeAsyncClient.instances
        start_closes = _FakeAsyncClient.close_calls
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            await _collect(provider.chat_stream(messages=[]))
            await _collect(provider.chat_stream(messages=[]))
            await provider.close()

        self.assertEqual(1, _FakeAsyncClient.instances - start_instances)
        self.assertEqual(1, _FakeAsyncClient.close_calls - start_closes)


class AnthropicProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_is_collected_from_start_and_delta_events(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":80,"cache_read_input_tokens":20}}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":12}}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["ok"], events)
        self.assertEqual(112, TokenUsage.from_raw(provider.last_usage).total_tokens)
        self.assertEqual(20, CacheUsage.from_raw(provider.last_usage).read_tokens)

    async def test_tools_use_one_explicit_breakpoint_and_enable_auto_caching(self):
        _FakeAsyncClient.lines = ["data: [DONE]"]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic", protocol="anthropic", model="claude-test",
                base_url="https://api.anthropic.com", api_key="test-key",
            )
        )
        tools = [
            {"name": "alpha", "description": "a", "input_schema": {"type": "object"}},
            {"name": "beta", "description": "b", "input_schema": {"type": "object"}},
        ]

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            await _collect(provider.chat_stream(messages=[], tools=tools))

        body = _FakeAsyncClient.last_stream_kwargs["json"]
        self.assertEqual({"type": "ephemeral"}, body["cache_control"])
        self.assertNotIn("cache_control", body["tools"][0])
        self.assertEqual({"type": "ephemeral"}, body["tools"][1]["cache_control"])

    async def test_empty_input_tool_use_yields_tool_call(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_1","name":"list_files","input":{}}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual([ToolCall(id="toolu_1", name="list_files", input={})], events)

    async def test_fragmented_input_tool_use_yields_parsed_arguments(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_2","name":"read_file","input":{}}}',
            'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\\"path\\":"}}',
            'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\\"README.md\\"}"}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [ToolCall(id="toolu_2", name="read_file", input={"path": "README.md"})],
            events,
        )

    async def test_tool_use_arguments_must_decode_to_object(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_array","name":"read_file","input":{}}}',
            'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"[]"}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("malformed_tool_arguments", raised.exception.code)

    async def test_only_malformed_sse_is_explicit_error(self):
        _FakeAsyncClient.lines = ["data: {broken-json", "data: [DONE]"]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic", protocol="anthropic", model="claude-test",
                base_url="https://api.anthropic.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("malformed_sse", raised.exception.code)

    async def test_malformed_stream_events_are_skipped(self):
        _FakeAsyncClient.lines = [
            "data: []",
            'data: {"type":"content_block_delta","delta":"not an object"}',
            'data: {"type":"content_block_start","content_block":"not an object"}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"still works"}}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["still works"], events)

    async def test_non_string_text_deltas_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":123}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":123}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"still works"}}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["still works"], events)

    async def test_non_string_tool_use_id_is_explicit_error(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":123,"name":"list_files","input":{}}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("incomplete_tool_call", raised.exception.code)
        self.assertTrue(raised.exception.retryable)

    async def test_non_string_tool_use_name_is_explicit_error(self):
        _FakeAsyncClient.lines = [
            'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_bad","name":123,"input":{}}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_stop"}',
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                protocol="anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.anthropic.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("incomplete_tool_call", raised.exception.code)
        self.assertTrue(raised.exception.retryable)


class OpenAIProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_object_sse_payload_is_skipped(self):
        _FakeAsyncClient.lines = [
            "data: []",
            'data: {"choices":[{"delta":{"content":"still works"}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["still works"], events)

    async def test_only_malformed_sse_is_explicit_error(self):
        _FakeAsyncClient.lines = ["data: {broken-json", "data: [DONE]"]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("malformed_sse", raised.exception.code)

    async def test_stream_usage_is_requested_and_collected(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"content":"ok"}}],"usage":null}',
            'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":5,"total_tokens":45}}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["ok"], events)
        self.assertEqual(45, TokenUsage.from_raw(provider.last_usage).total_tokens)
        self.assertEqual(
            {"include_usage": True},
            _FakeAsyncClient.last_stream_kwargs["json"]["stream_options"],
        )

    async def test_nested_cached_tokens_are_preserved_for_cache_accounting(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":80}}}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            await _collect(provider.chat_stream(messages=[]))

        cache = CacheUsage.from_raw(provider.last_usage)
        self.assertEqual(80, cache.read_tokens)
        self.assertEqual(20, cache.miss_tokens)
        self.assertTrue(provider.cache_hit)

    async def test_sse_data_line_without_space_is_accepted(self):
        _FakeAsyncClient.lines = [
            'data:{"choices":[{"delta":{"content":"works"}}]}',
            "data:[DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com/v1",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["works"], events)

    async def test_fragmented_tool_call_delta_yields_text_then_tool_call(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"content":"I will read it."}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_","arguments":"{\\"path\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\\"README.md\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [
                "I will read it.",
                ToolCall(id="call_1", name="read_file", input={"path": "README.md"}),
            ],
            events,
        )

    async def test_tool_call_arguments_must_decode_to_object(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_array","function":{"name":"read_file","arguments":"[]"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("malformed_tool_arguments", raised.exception.code)

    async def test_malformed_stream_chunks_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[null]}',
            'data: {"choices":[{"delta":"not an object"}]}',
            'data: {"choices":[{"delta":{"content":"still works"}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["still works"], events)

    async def test_non_string_content_chunks_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"content":123}}]}',
            'data: {"choices":[{"delta":{"content":"still works"}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["still works"], events)

    async def test_malformed_tool_call_deltas_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":"not a list"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":["bad-call"]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":"not an object"}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [ToolCall(id="call_1", name="read_file", input={"path": "README.md"})],
            events,
        )

    async def test_string_tool_call_index_fragments_merge_with_integer_index(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":"0","id":"call_1","function":{"name":"read_"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [ToolCall(id="call_1", name="read_file", input={"path": "README.md"})],
            events,
        )

    async def test_distinct_string_tool_call_indexes_do_not_merge(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":"0","id":"call_1","function":{"name":"glob","arguments":"{}"}},{"index":"1","id":"call_2","function":{"name":"grep","arguments":"{}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai", protocol="openai", model="gpt-test",
                base_url="https://api.openai.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [
                ToolCall(id="call_1", name="glob", input={}),
                ToolCall(id="call_2", name="grep", input={}),
            ],
            events,
        )

    async def test_non_string_tool_call_id_is_explicit_error(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":123,"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("incomplete_tool_call", raised.exception.code)
        self.assertTrue(raised.exception.retryable)

    async def test_missing_tool_call_name_is_explicit_error(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_bad","function":{"arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = OpenAIProvider(
            ProviderConfig(
                name="openai",
                protocol="openai",
                model="gpt-test",
                base_url="https://api.openai.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.openai.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("incomplete_tool_call", raised.exception.code)
        self.assertTrue(raised.exception.retryable)


class DeepSeekProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_usage_is_exposed_from_deepseek_counters(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_cache_hit_tokens":75,"prompt_cache_miss_tokens":25}}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek", protocol="deepseek", model="deepseek-chat",
                base_url="https://api.deepseek.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            await _collect(provider.chat_stream(messages=[]))

        cache = CacheUsage.from_raw(provider.last_usage)
        self.assertEqual(75, cache.read_tokens)
        self.assertEqual(25, cache.miss_tokens)
        self.assertTrue(provider.cache_hit)

    async def test_non_object_sse_payload_is_skipped(self):
        _FakeAsyncClient.lines = [
            "data: []",
            'data: {"choices":[{"delta":{"content":"still works"}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek", protocol="deepseek", model="deepseek-chat",
                base_url="https://api.deepseek.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["still works"], events)

    async def test_only_malformed_sse_is_explicit_error(self):
        _FakeAsyncClient.lines = ["data: {broken-json", "data: [DONE]"]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek", protocol="deepseek", model="deepseek-test",
                base_url="https://api.deepseek.com", api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("malformed_sse", raised.exception.code)

    async def test_fragmented_dsml_content_is_converted_to_tool_call(self):
        chunks = [
            "<｜｜DS",
            "ML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"read_file\">\n",
            '<｜｜DSML｜｜parameter name="file" string="true">',
            "merge_sort.py</｜｜DSML｜｜parameter>\n",
            "</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>",
        ]
        _FakeAsyncClient.lines = [
            "data: " + json.dumps(
                {"choices": [{"delta": {"content": chunk}}]},
                ensure_ascii=False,
            )
            for chunk in chunks
        ] + ["data: [DONE]"]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }]

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[], tools=tools))

        self.assertEqual(1, len(events))
        self.assertIsInstance(events[0], ToolCall)
        self.assertTrue(events[0].id.startswith("call_dsml_"))
        self.assertEqual("read_file", events[0].name)
        self.assertEqual({"path": "merge_sort.py"}, events[0].input)

    async def test_dsml_after_text_prefix_does_not_leak_markup(self):
        content = (
            "我先读取文件。\n<｜｜DSML｜｜tool_calls>"
            '<｜｜DSML｜｜invoke name="read_file">'
            '<｜｜DSML｜｜parameter name="file" string="true">merge_sort.py'
            '</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>'
            '</｜｜DSML｜｜tool_calls>'
        )
        _FakeAsyncClient.lines = [
            "data: " + json.dumps(
                {"choices": [{"delta": {"content": chunk}}]}, ensure_ascii=False,
            )
            for chunk in (content[:15], content[15:37], content[37:])
        ] + ["data: [DONE]"]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek", protocol="deepseek", model="deepseek-test",
                base_url="https://api.deepseek.com", api_key="test-key",
            )
        )
        tools = [{"type": "function", "function": {
            "name": "read_file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}},
            },
        }}]

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[], tools=tools))

        text = "".join(event for event in events if isinstance(event, str))
        calls = [event for event in events if isinstance(event, ToolCall)]
        self.assertEqual("我先读取文件。\n", text)
        self.assertNotIn("DSML", text)
        self.assertEqual({"path": "merge_sort.py"}, calls[0].input)

    async def test_malformed_dsml_is_an_explicit_provider_error(self):
        content = "<｜｜DSML｜｜tool_calls><broken>"
        _FakeAsyncClient.lines = [
            "data: " + json.dumps(
                {"choices": [{"delta": {"content": content}}]},
                ensure_ascii=False,
            ),
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[], tools=[]))

        self.assertEqual("malformed_dsml_tool_call", raised.exception.code)

    async def test_oversized_dsml_is_an_explicit_provider_error(self):
        content = "<｜｜DSML｜｜tool_calls>" + ("x" * 50)
        _FakeAsyncClient.lines = [
            "data: " + json.dumps(
                {"choices": [{"delta": {"content": content}}]},
                ensure_ascii=False,
            ),
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek", protocol="deepseek", model="deepseek-test",
                base_url="https://api.deepseek.com", api_key="test-key",
            )
        )

        with (
            patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient),
            patch("tinyCode.providers.deepseek.MAX_TOOL_ARGUMENT_CHARS", 32),
        ):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("tool_arguments_too_large", raised.exception.code)

    async def test_stream_usage_is_requested_and_collected(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"content":"ok"}}],"usage":null}',
            'data: {"choices":[],"usage":{"prompt_tokens":90,"completion_tokens":10,"total_tokens":100}}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["ok"], events)
        self.assertEqual(100, TokenUsage.from_raw(provider.last_usage).total_tokens)
        self.assertEqual(
            {"include_usage": True},
            _FakeAsyncClient.last_stream_kwargs["json"]["stream_options"],
        )

    async def test_reasoning_and_fragmented_tool_call_delta_are_preserved(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"Need file first."}}]}',
            'data: {"choices":[{"delta":{"content":"Reading now."}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","function":{"name":"grep","arguments":"{\\"pattern\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"TODO\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [
                "<<REASONING:Need file first.>>",
                "Reading now.",
                ToolCall(id="call_2", name="grep", input={"pattern": "TODO"}),
            ],
            events,
        )

    async def test_tool_call_arguments_must_decode_to_object(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_array","function":{"name":"grep","arguments":"[]"}}]}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("malformed_tool_arguments", raised.exception.code)

    async def test_malformed_stream_chunks_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[null]}',
            'data: {"choices":[{"delta":"not an object"}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"thinking","content":"answer"}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["<<REASONING:thinking>>", "answer"], events)

    async def test_non_string_content_and_reasoning_chunks_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"reasoning_content":123,"content":456}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"thinking","content":"answer"}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(["<<REASONING:thinking>>", "answer"], events)

    async def test_malformed_tool_call_deltas_are_skipped(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":"not a list"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":["bad-call"]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","function":"not an object"}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"grep","arguments":"{\\"pattern\\":\\"TODO\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [ToolCall(id="call_2", name="grep", input={"pattern": "TODO"})],
            events,
        )

    async def test_string_tool_call_index_fragments_merge_with_integer_index(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":"0","id":"call_2","function":{"name":"gr"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"ep","arguments":"{\\"pattern\\":\\"TODO\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            events = await _collect(provider.chat_stream(messages=[]))

        self.assertEqual(
            [ToolCall(id="call_2", name="grep", input={"pattern": "TODO"})],
            events,
        )

    async def test_non_string_tool_call_id_is_explicit_error(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":123,"function":{"name":"grep","arguments":"{\\"pattern\\":\\"TODO\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("incomplete_tool_call", raised.exception.code)
        self.assertTrue(raised.exception.retryable)

    async def test_missing_tool_call_name_is_explicit_error(self):
        _FakeAsyncClient.lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_bad","function":{"arguments":"{\\"pattern\\":\\"TODO\\"}"}}]}}]}',
            "data: [DONE]",
        ]
        provider = DeepSeekProvider(
            ProviderConfig(
                name="deepseek",
                protocol="deepseek",
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_key="test-key",
            )
        )

        with patch("tinyCode.providers.deepseek.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(ProviderError) as raised:
                await _collect(provider.chat_stream(messages=[]))

        self.assertEqual("incomplete_tool_call", raised.exception.code)
        self.assertTrue(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()

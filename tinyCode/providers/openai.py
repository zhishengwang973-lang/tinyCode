"""OpenAI provider — SSE streaming with tool calling support."""

import json
from collections.abc import AsyncIterator

import httpx

from tinyCode.config.models import ProviderConfig
from tinyCode.providers.base import (
    BaseProvider,
    Message,
    ProviderError,
    ProviderHTTPError,
    ToolCall,
    MAX_TOOL_ARGUMENT_CHARS,
    build_api_url,
    normalize_usage,
    normalize_tool_call_index,
    read_error_detail,
)
from tinyCode.providers.sse import SSEDecoder


class OpenAIProvider(BaseProvider):
    """Provider for the OpenAI Chat Completions API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        return self._client

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        url = build_api_url(self.config.base_url, "/v1/chat/completions")

        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            body["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        self.last_usage = {}

        client = self._get_client()
        if client is not None:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    raise ProviderHTTPError(
                        resp.status_code,
                        (await read_error_detail(resp))[:500],
                    )

                # Accumulate tool call fragments (indexed by tool call index)
                tool_calls_acc: dict[int, dict] = {}
                decoder = SSEDecoder()

                async for line in resp.aiter_lines():
                    done, data = decoder.parse(line)
                    if done:
                        break
                    if data is None:
                        continue
                    if not isinstance(data, dict):
                        continue

                    if data.get("error"):
                        raise ProviderError(
                            f"模型服务错误: {data['error']}",
                            code="provider_error",
                        )

                    usage = data.get("usage")
                    if isinstance(usage, dict) and usage:
                        self.last_usage = normalize_usage(usage)

                    choices = data.get("choices", [])
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        continue

                    # Handle tool call deltas
                    tc_deltas = delta.get("tool_calls", [])
                    if isinstance(tc_deltas, list) and tc_deltas:
                        for tc in tc_deltas:
                            if not isinstance(tc, dict):
                                continue
                            idx = normalize_tool_call_index(tc.get("index", 0))
                            if idx is None:
                                continue
                            if idx not in tool_calls_acc:
                                tool_id = tc.get("id", "")
                                tool_calls_acc[idx] = {
                                    "id": tool_id if isinstance(tool_id, str) else "",
                                    "type": "function",
                                    "function": {
                                        "name": "",
                                        "arguments": "",
                                    },
                                }
                            acc = tool_calls_acc[idx]
                            tool_id = tc.get("id")
                            if isinstance(tool_id, str) and tool_id:
                                acc["id"] = tool_id
                            func = tc.get("function", {})
                            if not isinstance(func, dict):
                                continue
                            name_part = func.get("name")
                            if isinstance(name_part, str) and name_part:
                                acc["function"]["name"] += name_part
                            arguments_part = func.get("arguments")
                            if isinstance(arguments_part, str) and arguments_part:
                                if len(acc["function"]["arguments"]) + len(arguments_part) > MAX_TOOL_ARGUMENT_CHARS:
                                    raise ProviderError(
                                        "工具调用参数超过大小限制",
                                        code="tool_arguments_too_large",
                                    )
                                acc["function"]["arguments"] += arguments_part

                    # Handle text content
                    content = delta.get("content", "")
                    if isinstance(content, str) and content:
                        yield content

                self.last_stream_diagnostics = decoder.diagnostics
                decoder.validate()

                # After stream ends, yield completed tool calls
                for tc_data in tool_calls_acc.values():
                    tool_id = tc_data.get("id", "")
                    func = tc_data.get("function", {})
                    tool_name = func.get("name", "")
                    if not isinstance(tool_id, str) or not tool_id:
                        raise ProviderError(
                            "OpenAI 工具调用流缺少有效调用 ID",
                            code="incomplete_tool_call",
                            retryable=True,
                        )
                    if not isinstance(tool_name, str) or not tool_name:
                        raise ProviderError(
                            "OpenAI 工具调用流缺少工具名",
                            code="incomplete_tool_call",
                            retryable=True,
                        )
                    args_str = func.get("arguments", "")
                    try:
                        tool_input = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError as exc:
                        raise ProviderError(
                            f"工具 '{tool_name}' 返回了无效 JSON 参数",
                            code="malformed_tool_arguments",
                        ) from exc
                    if not isinstance(tool_input, dict):
                        raise ProviderError(
                            f"工具 '{tool_name}' 的参数必须是 JSON 对象",
                            code="malformed_tool_arguments",
                        )
                    yield ToolCall(
                        id=tool_id,
                        name=tool_name,
                        input=tool_input,
                    )

    # -- tool message formatting (OpenAI style) --------------------------------

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = ""
    ) -> Message:
        return {
            "role": "assistant",
            "content": text_prefix or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input, ensure_ascii=False),
                    },
                }
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

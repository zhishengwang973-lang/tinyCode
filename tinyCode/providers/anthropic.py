"""Anthropic provider — SSE streaming with cache_control, thinking, and tool use."""

import json
from collections.abc import AsyncIterator
from typing import Any

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

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Provider for the Anthropic Messages API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._thinking_enabled = False
        self._thinking_budget_tokens = 4096
        self.last_usage: dict[str, int] = {}
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

    # -- thinking control ----------------------------------------------------

    def enable_thinking(self, budget_tokens: int = 4096) -> None:
        self._thinking_enabled = True
        self._thinking_budget_tokens = budget_tokens

    def disable_thinking(self) -> None:
        self._thinking_enabled = False

    def supports_thinking(self) -> bool:
        return True

    @property
    def thinking_enabled(self) -> bool:
        return self._thinking_enabled

    # -- streaming -----------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        url = build_api_url(self.config.base_url, "/v1/messages")

        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "messages": messages,
            "stream": True,
        }

        # System prompt with cache_control
        if system_blocks:
            body["system"] = system_blocks

        if self._thinking_enabled:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget_tokens,
            }

        # One breakpoint on the final tool covers the stable tool prefix.
        # Anthropic accepts at most four explicit cache breakpoints, so marking
        # every tool is both unnecessary and invalid for the built-in set.
        if tools:
            cached_tools = list(tools)
            cached_tools[-1] = {
                **cached_tools[-1], "cache_control": {"type": "ephemeral"},
            }
            body["tools"] = cached_tools

        # Advance a cache entry with the growing conversation while retaining
        # the explicit stable system/tool breakpoints above.
        body["cache_control"] = {"type": "ephemeral"}

        headers = {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "Accept": "text/event-stream",
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

                tool_blocks: dict[int, dict[str, Any]] = {}
                decoder = SSEDecoder()

                async for line in resp.aiter_lines():
                    done, data = decoder.parse(line)
                    if done:
                        break
                    if data is None:
                        continue
                    if not isinstance(data, dict):
                        continue

                    event_type = data.get("type", "")

                    if event_type == "error":
                        raise ProviderError(
                            f"模型服务错误: {data.get('error', data)}",
                            code="provider_error",
                        )

                    if event_type == "message_start":
                        message = data.get("message", {})
                        usage = message.get("usage", {}) if isinstance(message, dict) else {}
                        if isinstance(usage, dict) and usage:
                            self.last_usage.update(normalize_usage(usage))

                    # --- content block delta ---
                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if not isinstance(delta, dict):
                            continue
                        delta_type = delta.get("type", "")
                        if delta_type == "thinking_delta":
                            thinking = delta.get("thinking", "")
                            if isinstance(thinking, str) and thinking:
                                yield f"<<THINKING:{thinking}>>"
                        elif delta_type == "text_delta":
                            text = delta.get("text", "")
                            if isinstance(text, str) and text:
                                yield text
                        elif delta_type == "input_json_delta":
                            partial_json = delta.get("partial_json", "")
                            index = normalize_tool_call_index(data.get("index", 0))
                            if index is None:
                                continue
                            block = tool_blocks.get(index)
                            if isinstance(partial_json, str) and block is not None:
                                current = block["json"]
                                if len(current) + len(partial_json) > MAX_TOOL_ARGUMENT_CHARS:
                                    raise ProviderError(
                                        "工具调用参数超过大小限制",
                                        code="tool_arguments_too_large",
                                    )
                                block["json"] = current + partial_json

                    # --- tool use start ---
                    elif event_type == "content_block_start":
                        block = data.get("content_block", {})
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_use":
                            index = normalize_tool_call_index(data.get("index", 0))
                            if index is None:
                                continue
                            if index in tool_blocks:
                                raise ProviderError(
                                    "Anthropic 工具调用流重复开始同一内容块",
                                    code="malformed_tool_call",
                                )
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "")
                            initial_input = block.get("input")
                            tool_blocks[index] = {
                                "id": tool_id if isinstance(tool_id, str) else "",
                                "name": tool_name if isinstance(tool_name, str) else "",
                                "json": "",
                                "initial": initial_input if isinstance(initial_input, dict) else None,
                            }

                    # --- tool use end ---
                    elif event_type == "content_block_stop":
                        index = normalize_tool_call_index(data.get("index", 0))
                        if index is None:
                            continue
                        tool_block = tool_blocks.pop(index, None)
                        if tool_block and (
                            not tool_block["id"] or not tool_block["name"]
                        ):
                            raise ProviderError(
                                "Anthropic 工具调用流缺少调用 ID 或工具名",
                                code="incomplete_tool_call",
                                retryable=True,
                            )
                        if tool_block:
                            if tool_block["json"]:
                                try:
                                    tool_input = json.loads(tool_block["json"])
                                except json.JSONDecodeError as exc:
                                    raise ProviderError(
                                        f"工具 '{tool_block['name']}' 返回了无效 JSON 参数",
                                        code="malformed_tool_arguments",
                                    ) from exc
                                if not isinstance(tool_input, dict):
                                    raise ProviderError(
                                        f"工具 '{tool_block['name']}' 的参数必须是 JSON 对象",
                                        code="malformed_tool_arguments",
                                    )
                            else:
                                tool_input = tool_block["initial"] or {}
                            yield ToolCall(
                                id=tool_block["id"],
                                name=tool_block["name"],
                                input=tool_input,
                            )

                    # --- message delta (usage info) ---
                    elif event_type == "message_delta":
                        usage = data.get("usage", {})
                        if isinstance(usage, dict) and usage:
                            self.last_usage.update(normalize_usage(usage))

                    # --- message stop ---
                    elif event_type == "message_stop":
                        # Final usage from message_start or accumulated
                        pass

                self.last_stream_diagnostics = decoder.diagnostics
                decoder.validate()
                if tool_blocks:
                    raise ProviderError(
                        "Anthropic 工具调用流在 content_block_stop 前结束",
                        code="incomplete_tool_call",
                        retryable=True,
                    )

    # -- tool message formatting -----------------------------------------------

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = ""
    ) -> Message:
        content: list[dict] = []
        if text_prefix:
            content.append({"type": "text", "text": text_prefix})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        return {"role": "assistant", "content": content}

    def make_tool_result_message(
        self, tool_call_id: str, tool_name: str, result_text: str
    ) -> Message:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result_text,
                }
            ],
        }

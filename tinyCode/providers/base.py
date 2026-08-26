"""Abstract base provider, factory function, and shared types."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from contextvars import ContextVar

from tinyCode.config.models import ProviderConfig

#: A single chat message in provider-neutral format.
#: Values can be str, list[dict], or None (for tool messages).
Message = dict[str, Any]
MAX_TOOL_ARGUMENT_CHARS = 1_000_000
MAX_PARALLEL_TOOL_CALLS = 128
MAX_ERROR_BODY_BYTES = 16_384


def normalize_tool_call_index(value: object) -> int | None:
    """Normalize provider tool indexes without merging malformed values."""
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < MAX_PARALLEL_TOOL_CALLS
    ):
        return value
    return None


class ProviderError(RuntimeError):
    """Provider failure with a stable code and retry classification."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProviderHTTPError(ProviderError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        retryable = status_code in {408, 429} or 500 <= status_code < 600
        super().__init__(
            f"模型服务 HTTP {status_code}: {detail or '无错误详情'}",
            code=f"http_{status_code}",
            retryable=retryable,
        )


def build_api_url(base_url: str | None, endpoint: str) -> str:
    """Join a configurable API root with a versioned endpoint once."""
    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("Provider base_url 不能为空")
    endpoint = "/" + endpoint.lstrip("/")
    if base.endswith(endpoint):
        return base
    version_prefix = "/".join(endpoint.split("/")[:2])
    if version_prefix and base.endswith(version_prefix):
        return base + endpoint[len(version_prefix):]
    return base + endpoint


async def read_error_detail(response: Any) -> str:
    """Read a bounded provider error body without buffering it all in memory."""
    payload = bytearray()
    truncated = False
    async for chunk in response.aiter_bytes():
        remaining = MAX_ERROR_BODY_BYTES - len(payload)
        if remaining <= 0:
            truncated = True
            break
        payload.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
            break
    detail = payload.decode("utf-8", errors="replace")
    if truncated:
        detail += "…[错误响应已截断]"
    return detail


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage for one or more provider requests."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    available: bool = False

    @classmethod
    def from_raw(cls, raw: object) -> "TokenUsage":
        if not isinstance(raw, dict) or not raw:
            return cls()

        def _count(*names: str) -> int:
            for name in names:
                value = raw.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
            return 0

        input_tokens = _count("input_tokens", "prompt_tokens")
        output_tokens = _count("output_tokens", "completion_tokens")

        # Anthropic reports cached and newly cached input separately from the
        # uncached input_tokens field. Include all processed input in the turn.
        input_tokens += _count("cache_creation_input_tokens")
        input_tokens += _count("cache_read_input_tokens")

        raw_total = raw.get("total_tokens")
        total = (
            raw_total
            if isinstance(raw_total, int)
            and not isinstance(raw_total, bool)
            and raw_total >= 0
            else input_tokens + output_tokens
        )
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            available=True,
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            available=self.available or other.available,
        )


@dataclass(frozen=True)
class CacheUsage:
    """Provider-neutral prompt-cache accounting for one request or turn."""

    read_tokens: int = 0
    write_tokens: int = 0
    miss_tokens: int = 0
    available: bool = False

    @classmethod
    def from_raw(cls, raw: object) -> "CacheUsage":
        if not isinstance(raw, dict) or not raw:
            return cls()

        def _count(*names: str) -> int:
            for name in names:
                value = raw.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
            return 0

        # Chat Completions and Responses expose the cache counters in
        # differently named nested objects.  Keep this parsing here so all
        # frontends and trace sinks report the same meaning.
        details = raw.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = raw.get("input_tokens_details")
        if not isinstance(details, dict):
            details = {}

        read = _count("cache_read_input_tokens", "prompt_cache_hit_tokens")
        if not read:
            nested_read = details.get("cached_tokens")
            if isinstance(nested_read, int) and not isinstance(nested_read, bool):
                read = max(0, nested_read)

        write = _count("cache_creation_input_tokens")
        nested_write = details.get("cache_write_tokens")
        if not write and isinstance(nested_write, int) and not isinstance(nested_write, bool):
            write = max(0, nested_write)

        miss = _count("prompt_cache_miss_tokens")
        if not miss:
            # Anthropic's input_tokens excludes cache reads/writes.  OpenAI's
            # prompt/input tokens include them, hence subtract only nested
            # OpenAI-style values from that total.
            input_tokens = _count("prompt_tokens", "input_tokens")
            if details:
                miss = max(0, input_tokens - read - write)
            elif "cache_read_input_tokens" in raw or "cache_creation_input_tokens" in raw:
                miss = input_tokens

        available = bool(
            raw
            and (
                "prompt_cache_hit_tokens" in raw
                or "prompt_cache_miss_tokens" in raw
                or "cache_read_input_tokens" in raw
                or "cache_creation_input_tokens" in raw
                or bool(details)
            )
        )
        return cls(read, write, miss, available)

    def __add__(self, other: "CacheUsage") -> "CacheUsage":
        return CacheUsage(
            read_tokens=self.read_tokens + other.read_tokens,
            write_tokens=self.write_tokens + other.write_tokens,
            miss_tokens=self.miss_tokens + other.miss_tokens,
            available=self.available or other.available,
        )

    @property
    def hit_rate(self) -> float | None:
        total = self.read_tokens + self.miss_tokens
        return self.read_tokens / total if total else None


def normalize_usage(raw: object) -> dict[str, Any]:
    """Keep provider usage counters while discarding malformed SSE payloads."""
    if not isinstance(raw, dict):
        return {}
    nested_names = {"prompt_tokens_details", "input_tokens_details"}
    return {
        key: value
        for key, value in raw.items()
        if (
            isinstance(value, int) and not isinstance(value, bool)
        ) or (
            key in nested_names and isinstance(value, dict)
        )
    }


class BaseProvider(ABC):
    """Abstract interface for an LLM provider backend."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._last_usage_var: ContextVar[dict[str, Any]] = ContextVar(
            f"tinycode_provider_usage_{id(self)}", default={}
        )
        self.last_stream_diagnostics: dict[str, int] = {}

    @property
    def last_usage(self) -> dict[str, Any]:
        """Usage for the current async task, isolated from background calls."""
        return self._last_usage_var.get()

    @last_usage.setter
    def last_usage(self, value: dict[str, Any]) -> None:
        holder = self._last_usage_var.get()
        holder.clear()
        if isinstance(value, dict):
            holder.update(value)

    def begin_request(self) -> None:
        """Create a request-local usage holder before spawning timeout tasks."""
        self._last_usage_var.set({})

    @property
    def cache_usage(self) -> CacheUsage:
        return CacheUsage.from_raw(self.last_usage)

    @property
    def cache_hit(self) -> bool:
        return self.cache_usage.read_tokens > 0

    async def close(self) -> None:
        """Release provider-owned network resources."""
        return None

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system_blocks: list[dict] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        """Send messages and yield tokens or tool calls as they arrive via SSE.

        Args:
            messages: Ordered list of chat messages.
            tools: Optional tool definitions in provider-native format.
            system_blocks: Optional Anthropic-format system content blocks.

        Yields:
            Token strings (``str``) or ``ToolCall`` objects.
        """
        ...

    def supports_thinking(self) -> bool:
        """Whether this provider supports extended thinking."""
        return False

    # -- tool result formatting (provider-specific) ---------------------------

    def make_tool_calls_message(
        self, tool_calls: list[ToolCall], text_prefix: str = ""
    ) -> Message:
        """Create a single assistant message carrying one or more tool calls.

        Includes optional preceding text so the model's response is one turn.
        """
        raise NotImplementedError

    def make_tool_result_message(
        self, tool_call_id: str, tool_name: str, result_text: str
    ) -> Message:
        """Create a message carrying the tool execution result.

        The result is appended after the tool call message and before the
        follow-up model response.
        """
        raise NotImplementedError


def create_provider(config: ProviderConfig) -> BaseProvider:
    """Factory: instantiate the correct provider subclass for the given config."""
    if config.protocol == "anthropic":
        from tinyCode.providers.anthropic import AnthropicProvider

        return AnthropicProvider(config)

    if config.protocol == "openai":
        from tinyCode.providers.openai import OpenAIProvider

        return OpenAIProvider(config)

    if config.protocol == "deepseek":
        from tinyCode.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider(config)

    raise ValueError(f"不支持的协议: {config.protocol}")

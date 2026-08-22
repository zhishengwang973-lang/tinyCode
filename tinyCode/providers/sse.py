"""Small shared SSE decoder with diagnostics for provider streams."""

import json
from dataclasses import dataclass
from typing import Any

from tinyCode.providers.base import ProviderError

MAX_SSE_EVENT_CHARS = 2_000_000


@dataclass
class SSEDecoder:
    parsed_events: int = 0
    malformed_events: int = 0

    def parse(self, line: str) -> tuple[bool, Any | None]:
        """Return ``(done, payload)``; payload is None for ignored lines."""
        if not line or not line.startswith("data:"):
            return False, None
        if len(line) > MAX_SSE_EVENT_CHARS:
            raise ProviderError(
                f"模型流单个 SSE 事件超过 {MAX_SSE_EVENT_CHARS} 字符限制",
                code="sse_event_too_large",
            )
        data_str = line[len("data:"):].lstrip()
        if data_str.strip() == "[DONE]":
            return True, None
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            self.malformed_events += 1
            return False, None
        self.parsed_events += 1
        return False, payload

    def validate(self) -> None:
        if self.malformed_events:
            raise ProviderError(
                f"模型流包含 {self.malformed_events} 个损坏的 SSE JSON 事件",
                code="malformed_sse",
                retryable=True,
            )

    @property
    def diagnostics(self) -> dict[str, int]:
        return {
            "parsed_events": self.parsed_events,
            "malformed_events": self.malformed_events,
        }

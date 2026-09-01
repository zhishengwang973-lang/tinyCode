"""Conversation history — user/assistant/tool message storage."""

from copy import deepcopy
import math
import json

from tinyCode.providers.base import Message

CHARS_PER_TOKEN = 3.5


def estimate_text_tokens(text: str) -> int:
    """Conservative tokenizer-independent estimate for mixed CJK/ASCII text."""
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return math.ceil(ascii_chars / 4.0 + non_ascii_chars)


def _content_chars(content: object) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(item) for item in content)
    if isinstance(content, dict):
        return sum(_content_chars(value) for value in content.values())
    return len(str(content))


class ConversationHistory:
    """Ordered message list (user, assistant, tool). No system prompt — that
    is managed by the PromptBuilder and AgentLoop."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._deferred_user_messages: list[str] = []
        self._steering_user_messages: list[str] = []

    # -- mutation ------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        if not content:
            return
        self._messages.append({"role": "user", "content": content})

    def defer_user_message(self, content: str) -> None:
        """Queue asynchronous context until the next protocol-safe boundary."""
        if content:
            self._deferred_user_messages.append(content)

    @property
    def deferred_count(self) -> int:
        """Number of user/context messages waiting for a safe boundary."""
        return len(self._deferred_user_messages) + len(self._steering_user_messages)

    def queue_steering_message(self, content: str) -> None:
        """Queue user steering separately so cancellation can discard it."""
        if content:
            self._steering_user_messages.append(content)

    @property
    def steering_count(self) -> int:
        return len(self._steering_user_messages)

    def pending_steering_messages(self) -> list[str]:
        """Return a copy suitable for a durable runtime checkpoint."""
        return list(self._steering_user_messages)

    def restore_steering_messages(self, messages: list[str]) -> None:
        """Restore queued steering after a restart without duplicating it."""
        self._steering_user_messages = [
            str(message) for message in messages if str(message).strip()
        ]

    def discard_steering_messages(self) -> int:
        count = len(self._steering_user_messages)
        self._steering_user_messages.clear()
        return count

    def flush_steering(self) -> int:
        """Append queued steering without consuming background context."""
        pending = self._steering_user_messages
        self._steering_user_messages = []
        if len(pending) == 1:
            self.add_user_message(pending[0])
        elif pending:
            combined = "\n\n".join(
                f"[追加指令 {index}]\n{content}"
                for index, content in enumerate(pending, start=1)
            )
            self.add_user_message(combined)
        return len(pending)

    def flush_deferred(self) -> int:
        pending = self._deferred_user_messages + self._steering_user_messages
        self._deferred_user_messages = []
        self._steering_user_messages = []
        for content in pending:
            self.add_user_message(content)
        return len(pending)

    def add_assistant_message(self, content: str) -> None:
        if not content:
            return
        self._messages.append({"role": "assistant", "content": content})

    def add_raw_message(self, message: Message) -> None:
        if message.get("role") == "assistant":
            has_content = bool(message.get("content"))
            has_tool_calls = bool(message.get("tool_calls"))
            if not has_content and not has_tool_calls:
                return
        self._messages.append(deepcopy(message))

    def add_context_message(self, content: str) -> None:
        """Append a system-level context message (compression summary)."""
        self._messages.append({"role": "system", "content": content})

    def replace_messages(self, messages: list[Message]) -> None:
        self._messages = deepcopy(messages)

    def clear(self) -> None:
        self._messages.clear()
        self._deferred_user_messages.clear()
        self._steering_user_messages.clear()

    # -- access --------------------------------------------------------------

    def get_messages(self) -> list[Message]:
        """Return messages (no system prompt — caller adds prompt context)."""
        result: list[Message] = []
        for msg in self._messages:
            if msg.get("role") == "assistant":
                has_content = bool(msg.get("content"))
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_content and not has_tool_calls:
                    continue
            result.append(deepcopy(msg))
        return result

    def estimated_token_count(self) -> int:
        total_chars = 0
        for msg in self._messages:
            total_chars += _content_chars(msg.get("content"))
            if "tool_calls" in msg:
                total_chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
        # Preserve message structure while estimating mixed-language content.
        total = 0
        for msg in self._messages:
            total += estimate_text_tokens(str(msg.get("content", "")))
            if "tool_calls" in msg:
                total += estimate_text_tokens(
                    json.dumps(msg["tool_calls"], ensure_ascii=False)
                )
            total += 4  # role/framing overhead
        return max(0, total)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

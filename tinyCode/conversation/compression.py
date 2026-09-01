"""Compatibility wrapper — delegates to the two-layer token management system.

Layer 1: ``ToolResultTruncator`` — lightweight, runs before every API call.
Layer 2: ``StructuredSummarizer`` — expensive LLM call, only when needed.
"""

from dataclasses import dataclass

from tinyCode.conversation.history import ConversationHistory
from tinyCode.conversation.summarizer import StructuredSummarizer, SummaryResult
from tinyCode.conversation.truncator import ToolResultTruncator
from tinyCode.providers.base import BaseProvider, Message


@dataclass
class CompressionResult:
    """Public result (kept backward-compatible)."""
    was_compressed: bool = False
    messages_compressed: int = 0
    estimated_tokens_saved: int = 0
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    warning_issued: bool = False
    summary_result: SummaryResult | None = None
    model_request_made: bool = False
    error: str = ""


class ContextCompressor:
    """Orchestrates the two-layer token management pipeline.

    Usage (before each API call):
        1. ``truncate(messages)`` → lightweight tool-result truncation
        2. ``check_and_compress(history, provider)`` → expensive LLM summary
    """

    def __init__(self, model: str, provider: BaseProvider) -> None:
        self._truncator = ToolResultTruncator()
        self._summarizer = StructuredSummarizer(provider, model)
        self._warning_emitted = False

    # -- public API -----------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._summarizer.context_window

    @property
    def warning_threshold(self) -> int:
        return self._summarizer.trigger_threshold

    @property
    def circuit_open(self) -> bool:
        return self._summarizer.circuit_open

    def reset_circuit(self) -> None:
        self._summarizer.reset_circuit()

    def truncate(self, messages: list[Message]) -> list[Message]:
        """Layer 1: truncate oversized tool results (cheap, no LLM call)."""
        result, _ = self._truncator.process_round(messages)
        return result

    async def check_and_compress(
        self, history: ConversationHistory, provider: BaseProvider,
        *, extra_tokens: int = 0, force: bool = False,
    ) -> CompressionResult:
        """Layer 2: generate structured summary if near context limit.

        Returns a ``CompressionResult``.  The history is mutated in-place
        if compression occurred. ``force`` bypasses only the automatic token
        threshold; protocol-safe boundaries and all output safety checks still
        apply.
        """
        result = CompressionResult()

        messages = history.get_messages()
        estimated = StructuredSummarizer._estimate_tokens(messages) + max(0, extra_tokens)
        result.estimated_tokens_before = estimated
        result.estimated_tokens_after = estimated

        # Warning
        if not self._warning_emitted and estimated >= self._summarizer.trigger_threshold:
            result.warning_issued = True
            self._warning_emitted = True

        # Circuit breaker check
        if self._summarizer.circuit_open:
            # Below the hard limit we can keep serving the current turn without
            # spamming the UI on every round. Once the assembled request would
            # exceed the model window, however, returning a silent no-op would
            # send a request that is guaranteed to fail at the provider.
            if estimated >= self._summarizer.context_window:
                result.error = (
                    "摘要压缩已熔断且上下文达到模型窗口；"
                    "请使用 /compress 重置熔断后重试"
                )
            return result

        # Automatic compression is threshold-driven. An explicit /compress is
        # user intent and therefore bypasses only this policy gate.
        if not force and estimated < self._summarizer.trigger_threshold:
            return result

        result.model_request_made = True
        new_messages, summary = await self._summarizer.summarize(messages)
        result.summary_result = summary
        result.error = summary.error
        if summary.messages_compressed == 0:
            return result  # failed — circuit breaker already ticked

        # Replace history messages with compressed version
        history.replace_messages(new_messages)

        result.was_compressed = True
        result.messages_compressed = summary.messages_compressed
        result.estimated_tokens_saved = summary.tokens_saved
        result.estimated_tokens_after = (
            StructuredSummarizer._estimate_tokens(new_messages) + max(0, extra_tokens)
        )
        return result

    def reset_warning(self) -> None:
        self._warning_emitted = False

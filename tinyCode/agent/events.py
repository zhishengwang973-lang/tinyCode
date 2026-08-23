"""Agent event types — the event stream contract between Agent Loop and TUI."""

from dataclasses import dataclass
from enum import Enum

from tinyCode.providers.base import ToolCall
from tinyCode.tools.base import ToolResult


@dataclass
class UserMessageEvent:
    """用户发送了一条消息。"""
    text: str


@dataclass
class ThinkingEvent:
    """模型正在思考（Claude thinking / DeepSeek reasoning）。"""
    text: str
    label: str = "Thinking"


@dataclass
class TextDeltaEvent:
    """模型输出的一段增量文本。"""
    text: str


@dataclass
class ToolCallEvent:
    """模型请求调用一个工具。"""
    tool_call: ToolCall


@dataclass
class ToolResultEvent:
    """工具执行完毕，返回结果。"""
    tool_name: str
    result: ToolResult


@dataclass
class ToolBlockedEvent:
    """工具被拦截（plan-only 模式下写入类工具被阻止）。"""
    tool_name: str
    reason: str


@dataclass
class AgentDoneEvent:
    """Agent 循环终止。"""
    reason: str  # completed, paused, hard-limit, or cancelled reason

    @property
    def is_normal(self) -> bool:
        return self.reason == "no_tool_call"


@dataclass
class ErrorEvent:
    """不可恢复的错误。"""
    message: str
    code: str = "agent_error"
    retryable: bool = False


@dataclass
class RoundStartEvent:
    """新一轮 ReAct 回合开始。"""
    round_number: int
    max_rounds: int


class RoundLimitDecisionAction(Enum):
    """User choice after the current task consumes its soft round budget."""

    EXTEND = "extend"
    AUTO = "auto"
    STOP = "stop"


@dataclass(frozen=True)
class RoundLimitDecision:
    action: RoundLimitDecisionAction
    requested_limit: int | None = None


@dataclass
class RoundLimitReachedEvent:
    """The task is incomplete and is waiting for a round-budget decision."""

    round_number: int
    current_limit: int
    extension: int
    hard_limit: int
    stalled: bool
    future: object  # asyncio.Future[RoundLimitDecision]


@dataclass
class RoundLimitExtendedEvent:
    """The in-flight task received more rounds without restarting."""

    previous_limit: int
    new_limit: int
    hard_limit: int
    automatic: bool


@dataclass
class PlanOnlyToggleEvent:
    """plan-only 模式切换通知。"""
    enabled: bool


@dataclass
class HITLRequestEvent:
    """人在回路：需要用户审批才能继续执行工具。"""
    tool_name: str
    params: dict
    prompt: str
    future: object  # asyncio.Future[HITLDecision]


@dataclass
class TruncationEvent:
    """工具结果被截断的通知。"""
    tool_name: str
    original_chars: int
    file_path: str


@dataclass
class ContextCompressionEvent:
    """Context warning/compression performed between ReAct rounds."""
    warning_issued: bool
    was_compressed: bool
    estimated_tokens_before: int
    estimated_tokens_after: int
    error: str = ""


# Union type for consumers
AgentEvent = (
    UserMessageEvent
    | ThinkingEvent
    | TextDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | ToolBlockedEvent
    | AgentDoneEvent
    | ErrorEvent
    | RoundStartEvent
    | RoundLimitReachedEvent
    | RoundLimitExtendedEvent
    | PlanOnlyToggleEvent
    | HITLRequestEvent
    | TruncationEvent
    | ContextCompressionEvent
)

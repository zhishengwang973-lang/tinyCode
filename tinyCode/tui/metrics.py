"""Turn metrics independent of rendering and terminal state."""

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class TurnMetrics:
    started_at: float = field(default_factory=lambda: monotonic())
    turns: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0

    def record_round(self, round_number: int) -> None:
        self.turns = max(self.turns, max(0, round_number))

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_tool_result(self, success: bool) -> None:
        if success:
            self.successful_tool_calls += 1

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, monotonic() - self.started_at)

    @property
    def success_rate_text(self) -> str:
        if not self.tool_calls:
            return "—"
        return f"{self.successful_tool_calls / self.tool_calls * 100:.1f}%"

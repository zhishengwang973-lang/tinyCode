"""Sub-agent models — roles, tasks, status."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic

from tinyCode.time_utils import beijing_now_iso


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentRole:
    name: str
    description: str = ""
    tools_allow: list[str] | None = None   # None = all except blocked
    tools_deny: list[str] = field(default_factory=list)
    model: str | None = None               # None = inherit parent
    max_rounds: int = 24                    # hard cap
    initial_rounds: int = 8                 # first soft budget
    round_extension: int = 4
    finalization_rounds: int = 2
    permission: str = "normal"             # strict / normal / permissive
    timeout_seconds: float = 300.0
    system_prompt: str = ""                # Markdown body
    source: str = ""


@dataclass
class SubAgentTask:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: str | None = None                # None = fork mode
    task: str = ""
    status: TaskStatus = TaskStatus.QUEUED
    result: str = ""
    token_usage: int = 0
    round_count: int = 0
    started_at: str = ""
    finished_at: str = ""
    background: bool = False
    result_path: str = ""
    elapsed_seconds: float = 0.0
    _started_monotonic: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = beijing_now_iso()
        self._started_monotonic = monotonic()

    def complete(self, result: str, tokens: int = 0, rounds: int = 0) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.token_usage = tokens
        self.round_count = rounds
        self.finished_at = beijing_now_iso()
        self._finish_timer()

    def fail(
        self, error: str, *, tokens: int | None = None, rounds: int | None = None,
    ) -> None:
        self.status = TaskStatus.FAILED
        self.result = error
        if tokens is not None:
            self.token_usage = tokens
        if rounds is not None:
            self.round_count = rounds
        self.finished_at = beijing_now_iso()
        self._finish_timer()

    def cancel(self) -> None:
        self.status = TaskStatus.CANCELLED
        self.finished_at = beijing_now_iso()
        self._finish_timer()

    def _finish_timer(self) -> None:
        if self._started_monotonic:
            self.elapsed_seconds = max(
                self.elapsed_seconds, monotonic() - self._started_monotonic,
            )

    @property
    def duration_seconds(self) -> float:
        if not self._started_monotonic:
            return max(0.0, self.elapsed_seconds)
        if self.status == TaskStatus.RUNNING:
            return max(0.0, monotonic() - self._started_monotonic)
        return max(0.0, self.elapsed_seconds)

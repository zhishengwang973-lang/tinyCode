"""Sub-agent models — roles, tasks, status."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


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
    max_rounds: int = 5
    permission: str = "normal"             # strict / normal / permissive
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

    def start(self) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()

    def complete(self, result: str, tokens: int = 0, rounds: int = 0) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.token_usage = tokens
        self.round_count = rounds
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.result = error
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def cancel(self) -> None:
        self.status = TaskStatus.CANCELLED
        self.finished_at = datetime.now(timezone.utc).isoformat()

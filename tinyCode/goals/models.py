"""Typed durable state for a single Codex-style Goal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from tinyCode.time_utils import beijing_now_iso


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    BUDGET_LIMITED = "budget_limited"
    BLOCKED = "blocked"


@dataclass
class Goal:
    """A completion contract that belongs to exactly one chat session."""

    id: str
    session_id: str
    objective: str
    max_turns: int
    status: GoalStatus = GoalStatus.ACTIVE
    completed_turns: int = 0
    model_requests: int = 0
    tool_calls: int = 0
    completion_evidence: str = ""
    last_reason: str = ""
    created_at: str = field(default_factory=beijing_now_iso)
    updated_at: str = field(default_factory=beijing_now_iso)

    @classmethod
    def create(cls, session_id: str, objective: str, *, max_turns: int) -> "Goal":
        return cls(
            id=uuid4().hex[:12],
            session_id=session_id,
            objective=objective.strip(),
            max_turns=max(1, int(max_turns)),
        )

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.completed_turns)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            objective=str(data["objective"]),
            max_turns=max(1, int(data.get("max_turns", 1))),
            status=GoalStatus(data.get("status", GoalStatus.ACTIVE.value)),
            completed_turns=max(0, int(data.get("completed_turns", 0))),
            model_requests=max(0, int(data.get("model_requests", 0))),
            tool_calls=max(0, int(data.get("tool_calls", 0))),
            completion_evidence=str(data.get("completion_evidence", "")),
            last_reason=str(data.get("last_reason", "")),
            created_at=str(data.get("created_at", "")) or beijing_now_iso(),
            updated_at=str(data.get("updated_at", "")) or beijing_now_iso(),
        )

    def render(self) -> str:
        labels = {
            GoalStatus.ACTIVE: "进行中",
            GoalStatus.PAUSED: "已暂停",
            GoalStatus.COMPLETED: "已完成",
            GoalStatus.BUDGET_LIMITED: "预算已用尽",
            GoalStatus.BLOCKED: "已阻塞",
        }
        lines = [
            f"Goal · {labels[self.status]} · {self.id}",
            f"目标: {self.objective}",
            f"进度: {self.completed_turns}/{self.max_turns} 个执行回合 · "
            f"模型请求 {self.model_requests} 次 · 工具调用 {self.tool_calls} 次",
        ]
        if self.completion_evidence:
            lines.append(f"完成证据: {self.completion_evidence}")
        if self.last_reason:
            lines.append(f"状态说明: {self.last_reason}")
        return "\n".join(lines)

    def prompt_context(self) -> str:
        return "\n".join((
            "[持久 Goal] 以下目标在本线程中仍然有效。",
            f"目标：{self.objective}",
            "不要仅凭主观判断宣布完成。完成前必须核对文件、测试、命令输出、生成物或其他可复核证据，"
            "并调用 goal_complete 记录简明证据。",
            "若当前路径无法继续，请说明阻塞点和需要的输入；不要反复尝试同一无效操作。",
            f"Goal 执行预算剩余：{self.remaining_turns} 个回合。",
        ))

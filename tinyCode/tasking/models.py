"""Typed, provider-neutral task-plan data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from tinyCode.time_utils import beijing_now_iso


class ExecutorKind(str, Enum):
    MAIN = "main"
    SUBAGENT = "subagent"
    TEAM = "team"


class TaskNodeStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    DISCARDED = "discarded"


@dataclass
class TaskNode:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    read_scope: list[str] = field(default_factory=list)
    write_scope: list[str] = field(default_factory=list)
    acceptance: list[str] = field(default_factory=list)
    executor: ExecutorKind = ExecutorKind.MAIN
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    result: str = ""
    updated_at: str = field(default_factory=beijing_now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["executor"] = self.executor.value
        data["status"] = self.status.value
        return data


@dataclass
class TaskPlan:
    id: str
    goal: str
    tasks: list[TaskNode]
    mode: str
    status: TaskPlanStatus = TaskPlanStatus.ACTIVE
    created_at: str = field(default_factory=beijing_now_iso)
    updated_at: str = field(default_factory=beijing_now_iso)
    source: str = "planner"
    error: str = ""

    @classmethod
    def create(
        cls, goal: str, tasks: list[TaskNode], *, mode: str, source: str,
    ) -> "TaskPlan":
        return cls(
            id=uuid4().hex[:12], goal=goal, tasks=tasks, mode=mode,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "mode": self.mode,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "error": self.error,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPlan":
        tasks: list[TaskNode] = []
        raw_tasks = data.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue
            try:
                tasks.append(TaskNode(
                    id=str(raw["id"]),
                    title=str(raw["title"]),
                    description=str(raw["description"]),
                    depends_on=_string_list(raw.get("depends_on")),
                    read_scope=_string_list(raw.get("read_scope")),
                    write_scope=_string_list(raw.get("write_scope")),
                    acceptance=_string_list(raw.get("acceptance")),
                    executor=ExecutorKind(raw.get("executor", "main")),
                    status=TaskNodeStatus(raw.get("status", "pending")),
                    result=str(raw.get("result", "")),
                    updated_at=str(raw.get("updated_at", "")) or beijing_now_iso(),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return cls(
            id=str(data["id"]), goal=str(data.get("goal", "")), tasks=tasks,
            mode=str(data.get("mode", "modify")),
            status=TaskPlanStatus(data.get("status", "active")),
            created_at=str(data.get("created_at", "")) or beijing_now_iso(),
            updated_at=str(data.get("updated_at", "")) or beijing_now_iso(),
            source=str(data.get("source", "planner")),
            error=str(data.get("error", "")),
        )

    def render(self) -> str:
        icons = {
            TaskNodeStatus.PENDING: "○",
            TaskNodeStatus.IN_PROGRESS: "◐",
            TaskNodeStatus.BLOCKED: "!",
            TaskNodeStatus.COMPLETED: "✓",
            TaskNodeStatus.FAILED: "×",
            TaskNodeStatus.SKIPPED: "–",
        }
        done = sum(task.status is TaskNodeStatus.COMPLETED for task in self.tasks)
        lines = [
            f"任务计划 · {self.status.value} · {done}/{len(self.tasks)} 完成 · {self.id}",
        ]
        if self.status is TaskPlanStatus.DRAFT:
            lines.append("  等待审批：可用 /plan approve 开始执行，或 /plan revise <要求> 修订。")
        for index, task in enumerate(self.tasks, start=1):
            deps = f" ← {', '.join(task.depends_on)}" if task.depends_on else ""
            lines.append(
                f"  {icons[task.status]} {index}. {task.title}"
                f" [{task.executor.value}]{deps}"
            )
        return "\n".join(lines)

    def render_detail(self) -> str:
        """Render the user-reviewable plan, including scope and acceptance."""
        states = {
            TaskPlanStatus.DRAFT: "草案，等待审批",
            TaskPlanStatus.ACTIVE: "执行中",
            TaskPlanStatus.NEEDS_REVIEW: "等待核验",
            TaskPlanStatus.COMPLETED: "已完成",
            TaskPlanStatus.FAILED: "执行失败",
            TaskPlanStatus.INTERRUPTED: "已中断",
            TaskPlanStatus.DISCARDED: "已丢弃",
        }
        lines = [
            f"任务计划 · {states[self.status]} · {self.id}",
            f"目标：{self.goal}",
            f"规划来源：{self.source}",
        ]
        if self.error:
            lines.append(f"说明：{self.error}")
        for index, task in enumerate(self.tasks, start=1):
            deps = ", ".join(task.depends_on) or "无"
            read_scope = ", ".join(task.read_scope) or "未限定"
            write_scope = ", ".join(task.write_scope) or "无（只读）"
            acceptance = "；".join(task.acceptance) or "完成任务描述"
            lines.extend((
                f"\n{index}. [{task.status.value}] {task.title} · {task.executor.value}",
                f"   内容：{task.description}",
                f"   依赖：{deps}",
                f"   读取范围：{read_scope}",
                f"   写入范围：{write_scope}",
                f"   验收：{acceptance}",
            ))
        if self.status is TaskPlanStatus.DRAFT:
            lines.append("\n下一步：/plan revise <要求> 修订；/plan approve 批准执行；/plan discard 丢弃。")
        return "\n".join(lines)

    def prompt_context(self) -> str:
        lines = [
            f"任务计划 {self.id} 已创建。按依赖顺序执行；不要重复或跳过节点。",
            "完成节点的验收条件后，调用 task_plan_update 更新状态和简短结果。",
            "写入范围不明确时，先读取和确认；不要并行修改存在重叠范围的节点。",
        ]
        for task in self.tasks:
            deps = ", ".join(task.depends_on) or "无"
            scopes = ", ".join(task.write_scope) or "仅只读或待确认"
            acceptance = "；".join(task.acceptance) or "完成描述中的目标"
            lines.append(
                f"- [{task.id}] {task.title}；依赖: {deps}；"
                f"执行者: {task.executor.value}；写入范围: {scopes}；验收: {acceptance}"
            )
            if task.executor is ExecutorKind.SUBAGENT:
                lines.append(
                    f"  节点 {task.id} 必须使用 sub_agent 派发独立只读探索，"
                    "在取得结果后再继续；不得让 Subagent 修改工作区。"
                )
        return "\n".join(lines)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]

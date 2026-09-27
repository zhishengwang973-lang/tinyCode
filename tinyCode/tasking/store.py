"""Crash-tolerant project-local persistence for generic task plans."""

from __future__ import annotations

import json
from pathlib import Path

from tinyCode.storage.journal import atomic_write_text
from tinyCode.tasking.models import TaskPlan, TaskPlanStatus
from tinyCode.time_utils import beijing_now_iso


class TaskPlanStore:
    """Store plans independently from chat sessions and Team state."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._dir = self._root / ".tinyCode" / "task_plans"
        self.last_error = ""

    def save(self, plan: TaskPlan) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        plan.updated_at = beijing_now_iso()
        atomic_write_text(
            self._path(plan.id),
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    def load(self, plan_id: str) -> TaskPlan | None:
        self.last_error = ""
        try:
            raw = json.loads(self._path(plan_id).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("任务计划顶层必须是对象")
            return TaskPlan.from_dict(raw)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def list_recent(self, *, limit: int = 20) -> list[TaskPlan]:
        self.last_error = ""
        try:
            files = sorted(
                self._dir.glob("*.json"), key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        plans: list[TaskPlan] = []
        for path in files[:max(0, limit)]:
            plan = self.load(path.stem)
            if plan is not None:
                plans.append(plan)
        return plans

    def mark_interrupted_active_plans(self) -> int:
        """Make crashed planning runs explicit instead of silently resuming."""
        changed = 0
        for plan in self.list_recent(limit=100):
            if plan.status is not TaskPlanStatus.ACTIVE:
                continue
            plan.status = TaskPlanStatus.INTERRUPTED
            plan.error = "TinyCode 在任务计划仍活跃时退出；恢复前请核验文件和测试状态。"
            self.save(plan)
            changed += 1
        return changed

    def _path(self, plan_id: str) -> Path:
        safe = "".join(char for char in plan_id if char.isalnum() or char in "-_")
        if not safe or safe != plan_id:
            raise ValueError("任务计划 ID 无效")
        return self._dir / f"{safe}.json"

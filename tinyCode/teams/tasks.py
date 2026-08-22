"""Shared task list — create, view, list, update with dependency support."""

import json
from datetime import datetime, timezone
from pathlib import Path

from tinyCode.teams.models import TaskStatus, TeamTask
from tinyCode.storage.sessions import _atomic_write_text


class SharedTaskList:
    """Persistent task list for a team, stored as JSON."""

    def __init__(self, team_dir: Path) -> None:
        self._file = team_dir / "tasks.json"
        self._tasks: dict[str, TeamTask] = {}
        self._active_run_id = ""
        self.load_error = ""
        self._load()

    # -- CRUD ----------------------------------------------------------------

    def set_active_run(self, run_id: str) -> None:
        """Tag tasks subsequently created by members with the active execution."""
        self._active_run_id = run_id

    def create(self, name: str, description: str = "",
               depends_on: list[str] | None = None,
               preferred_member: str = "",
               run_id: str | None = None) -> TeamTask:
        task = TeamTask(name=name, description=description,
                        depends_on=depends_on or [],
                        preferred_member=preferred_member,
                        run_id=self._active_run_id if run_id is None else run_id)
        self._tasks[task.id] = task
        self._save()
        return task

    def get(self, task_id: str) -> TeamTask | None:
        return self._tasks.get(task_id)

    def list_all(self, status: TaskStatus | None = None) -> list[TeamTask]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at)

    def list_for_run(self, run_id: str) -> list[TeamTask]:
        return [task for task in self.list_all() if task.run_id == run_id]

    def update(self, task_id: str, **kwargs) -> TeamTask | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        text_fields = {
            "name", "description", "run_id", "preferred_member",
            "assigned_to", "result",
        }
        for key, value in kwargs.items():
            if key in text_fields and isinstance(value, str):
                setattr(task, key, value)
            elif key == "status" and isinstance(value, TaskStatus):
                task.status = value
            elif key == "depends_on" and isinstance(value, list) and all(
                isinstance(dep, str) for dep in value
            ):
                task.depends_on = value
        task.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return task

    def assign(self, task_id: str, member_name: str) -> TeamTask | None:
        return self.update(task_id, assigned_to=member_name,
                           status=TaskStatus.IN_PROGRESS)

    def complete(self, task_id: str, result: str = "") -> TeamTask | None:
        return self.update(task_id, status=TaskStatus.COMPLETED, result=result)

    def ready_tasks(self) -> list[TeamTask]:
        """Tasks whose dependencies are all completed and are unassigned."""
        ready: list[TeamTask] = []
        for t in self._tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            if t.assigned_to:
                continue
            deps_met = all(
                self._tasks.get(d) and self._tasks[d].status == TaskStatus.COMPLETED
                for d in t.depends_on
            )
            if deps_met:
                ready.append(t)
        return ready

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        data = {
            tid: {
                "id": t.id, "name": t.name, "description": t.description,
                "run_id": t.run_id, "preferred_member": t.preferred_member,
                "assigned_to": t.assigned_to, "depends_on": t.depends_on,
                "status": t.status.value, "result": t.result,
                "created_at": t.created_at, "updated_at": t.updated_at,
            }
            for tid, t in self._tasks.items()
        }
        _atomic_write_text(
            self._file,
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def _load(self) -> None:
        self.load_error = ""
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self.load_error = "tasks.json 顶层必须是对象"
                return
            for tid, d in data.items():
                if not isinstance(tid, str):
                    continue
                task = self._parse_task(tid, d)
                if task:
                    self._tasks[tid] = task
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _parse_task(tid: str, data: dict) -> TeamTask | None:
        if not isinstance(data, dict):
            return None
        try:
            status = TaskStatus(data.get("status", "pending"))
        except (ValueError, TypeError):
            return None
        depends_on = data.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) for dep in depends_on
        ):
            depends_on = []

        def _text(key: str, default: str = "") -> str:
            value = data.get(key, default)
            return value if isinstance(value, str) else default

        return TeamTask(
            id=_text("id", tid), name=_text("name"),
            description=_text("description"),
            run_id=_text("run_id"), preferred_member=_text("preferred_member"),
            assigned_to=_text("assigned_to"), depends_on=depends_on,
            status=status,
            result=_text("result"), created_at=_text("created_at"),
            updated_at=_text("updated_at"),
        )

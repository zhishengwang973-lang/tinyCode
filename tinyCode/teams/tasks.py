"""Shared task list — create, view, list, update with dependency support."""

import json
from pathlib import Path

from tinyCode.teams.models import TaskStatus, TeamTask
from tinyCode.storage.sessions import _atomic_write_text
from tinyCode.time_utils import beijing_now_iso, timestamp_sort_key


MAX_TEAM_TASKS = 128
MAX_TASK_NAME_CHARS = 200
MAX_TASK_DESCRIPTION_CHARS = 50_000
MAX_TASK_RESULT_CHARS = 100_000


class SharedTaskList:
    """Persistent task list for a team, stored as JSON."""

    def __init__(self, team_dir: Path) -> None:
        self._file = team_dir / "tasks.json"
        self._results_dir = team_dir / "results"
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
        self._prune_terminal_tasks()
        if len(self._tasks) >= MAX_TEAM_TASKS:
            raise ValueError(f"Team 任务数量已达到上限 {MAX_TEAM_TASKS}")
        if len(name) > MAX_TASK_NAME_CHARS:
            raise ValueError(f"任务名称超过 {MAX_TASK_NAME_CHARS} 字符上限")
        if len(description) > MAX_TASK_DESCRIPTION_CHARS:
            raise ValueError(
                f"任务描述超过 {MAX_TASK_DESCRIPTION_CHARS} 字符上限"
            )
        task = TeamTask(name=name, description=description,
                        depends_on=depends_on or [],
                        preferred_member=preferred_member,
                        run_id=self._active_run_id if run_id is None else run_id)
        while task.id in self._tasks:
            task = TeamTask(
                name=name, description=description,
                depends_on=depends_on or [],
                preferred_member=preferred_member,
                run_id=self._active_run_id if run_id is None else run_id,
            )
        self._tasks[task.id] = task
        self._save()
        return task

    def get(self, task_id: str) -> TeamTask | None:
        return self._tasks.get(task_id)

    def list_all(self, status: TaskStatus | None = None) -> list[TeamTask]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: timestamp_sort_key(t.created_at))

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
                if key == "result" and len(value) > MAX_TASK_RESULT_CHARS:
                    self._results_dir.mkdir(parents=True, exist_ok=True)
                    result_path = self._results_dir / f"{task.id}.md"
                    _atomic_write_text(result_path, value)
                    value = (
                        value[:MAX_TASK_RESULT_CHARS]
                        + f"\n\n[完整结果: {result_path}]"
                    )
                setattr(task, key, value)
            elif key == "status" and isinstance(value, TaskStatus):
                task.status = value
            elif key == "depends_on" and isinstance(value, list) and all(
                isinstance(dep, str) for dep in value
            ):
                task.depends_on = value
        task.updated_at = beijing_now_iso()
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

    def reconcile_interrupted(self) -> int:
        """Mark persisted in-flight tasks as failed before starting a new run."""
        count = 0
        for task in self._tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.FAILED
                task.result = "上一次 Team 进程中断，任务未完成"
                task.updated_at = beijing_now_iso()
                count += 1
        if count:
            self._save()
        return count

    def _prune_terminal_tasks(self) -> None:
        if len(self._tasks) < MAX_TEAM_TASKS:
            return
        terminal = [
            task for task in self.list_all()
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
            and task.run_id != self._active_run_id
        ]
        target_size = max(0, MAX_TEAM_TASKS - 16)
        for task in terminal:
            if len(self._tasks) <= target_size:
                break
            self._tasks.pop(task.id, None)
        if terminal:
            self._save()

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

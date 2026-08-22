"""Background task manager — track, notify, manage sub-agent tasks."""

import asyncio
from datetime import datetime, timezone

from tinyCode.subagent.models import SubAgentTask, TaskStatus


class BackgroundTaskManager:
    """Tracks background sub-agent tasks and injects results on completion."""

    def __init__(self, max_concurrent: int = 4) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent 必须大于 0")
        self._tasks: dict[str, SubAgentTask] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent

    def create(self, role: str | None, task_text: str, background: bool = False) -> SubAgentTask:
        task = SubAgentTask(role=role, task=task_text, background=background)
        self._tasks[task.id] = task
        return task

    def list_tasks(self) -> list[SubAgentTask]:
        return list(self._tasks.values())

    def get(self, task_id: str) -> SubAgentTask | None:
        return self._tasks.get(task_id)

    def attach(self, task_id: str, running: asyncio.Task) -> None:
        """Associate a model task with its actual asyncio execution."""
        self._running[task_id] = running
        def _done(finished: asyncio.Task, tid: str = task_id) -> None:
            self._forget_running(tid, finished)

        running.add_done_callback(_done)

    @property
    def can_start(self) -> bool:
        return sum(not task.done() for task in self._running.values()) < self._max_concurrent

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            task.cancel()
            running = self._running.get(task_id)
            if running is not None and not running.done():
                running.cancel()
            return True
        return False

    def inject_result(self, task: SubAgentTask, history) -> None:
        """Inject a completed task's result into the main conversation.

        The original background tool call already has its own tool result.
        Completion is later context, so it must not be an orphan tool message.
        """
        if task.status == TaskStatus.COMPLETED:
            content = f"[Sub-agent '{task.role or 'fork'}' ({task.id}) 完成]\n{task.result}"
        elif task.status == TaskStatus.FAILED:
            content = f"[Sub-agent '{task.role or 'fork'}' ({task.id}) 失败]\n{task.result}"
        else:
            content = f"[Sub-agent '{task.role or 'fork'}' ({task.id}) 已取消]"

        history.defer_user_message(content)

    def _forget_running(self, task_id: str, finished: asyncio.Task) -> None:
        if self._running.get(task_id) is finished:
            self._running.pop(task_id, None)
        if not finished.cancelled():
            finished.exception()

    async def shutdown(self) -> None:
        """Cancel and join every live background sub-agent."""
        running = list(self._running.values())
        for task in running:
            if not task.done():
                task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        self._running.clear()

    def get_status_summary(self) -> str:
        """Return a human-readable summary of all tasks."""
        if not self._tasks:
            return "没有后台任务"
        lines = ["后台任务:"]
        for t in sorted(self._tasks.values(), key=lambda x: x.started_at or "", reverse=True):
            status_icon = {
                TaskStatus.QUEUED: "⏳", TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✅", TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🚫",
            }.get(t.status, "❓")
            role = t.role or "fork"
            lines.append(f"  {status_icon} {t.id}  {role}  {t.task[:60]}")
        return "\n".join(lines)

"""Stable task-plan tools exposed to normal agent turns."""

from __future__ import annotations

from tinyCode.tasking.planner import TaskPlanningService
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.validation import require_string


class TaskPlanListTool(BaseTool):
    """Expose the current plan without making the model reconstruct it."""

    def __init__(self, service: TaskPlanningService) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "task_plan_list"

    @property
    def description(self) -> str:
        return "查看当前任务计划、依赖关系和每个节点的执行状态。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    async def execute(self) -> ToolResult:
        return ToolResult(success=True, content=self._service.render_active())


class TaskPlanUpdateTool(BaseTool):
    """Persist a node transition after its acceptance criteria are checked."""

    def __init__(self, service: TaskPlanningService) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "task_plan_update"

    @property
    def description(self) -> str:
        return "更新计划节点状态；completed 前必须满足该节点验收条件。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WRITE

    @property
    def available_in_inspect(self) -> bool:
        return True

    def may_modify(self, params: dict) -> bool:
        del params
        return False

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("task_id", "string", "任务计划节点 ID"),
            ToolParameter("status", "string", "pending/in_progress/blocked/completed/failed/skipped"),
            ToolParameter("result", "string", "完成证据或阻塞原因", required=False),
        ]

    async def execute(self, task_id: str, status: str, result: str = "") -> ToolResult:
        try:
            task_id = require_string(task_id, "task_id").strip()
            status = require_string(status, "status").strip().lower()
            result = require_string(result, "result")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))
        message = self._service.update_task(task_id, status, result)
        success = "已更新为" in message
        return ToolResult(success=success, content=message if success else "", error="" if success else message)

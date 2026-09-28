"""Stable Goal lifecycle tools exposed to tool-enabled turns."""

from __future__ import annotations

from tinyCode.goals.service import GoalService
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.validation import require_string


class GoalStatusTool(BaseTool):
    def __init__(self, service: GoalService) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "goal_status"

    @property
    def description(self) -> str:
        return "查看当前线程 Goal 的目标、预算、状态和已记录证据。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    async def execute(self) -> ToolResult:
        return ToolResult(success=True, content=self._service.render())


class GoalCompleteTool(BaseTool):
    def __init__(self, service: GoalService) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "goal_complete"

    @property
    def description(self) -> str:
        return (
            "仅当当前 Goal 已由测试、文件、命令输出或产物等可复核证据满足时调用，"
            "并记录证据摘要。不能仅凭主观判断调用。"
        )

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
        return [ToolParameter("evidence", "string", "可复核完成证据的简明摘要")]

    async def execute(self, evidence: str) -> ToolResult:
        try:
            evidence = require_string(evidence, "evidence")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))
        result = self._service.complete(evidence)
        return ToolResult(
            success=result == "Goal 已标记完成",
            content=result if result == "Goal 已标记完成" else "",
            error="" if result == "Goal 已标记完成" else result,
        )

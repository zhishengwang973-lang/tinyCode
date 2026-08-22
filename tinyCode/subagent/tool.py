"""Sub-agent tool — the single sub_agent entry point."""

import asyncio

from tinyCode.conversation.history import ConversationHistory
from tinyCode.subagent.manager import BackgroundTaskManager
from tinyCode.subagent.models import SubAgentRole
from tinyCode.subagent.runner import SubAgentRunner
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.validation import require_string


class SubAgentTool(BaseTool):
    """Single tool for creating sub-agents by role or fork."""

    def __init__(
        self,
        runner: SubAgentRunner,
        task_manager: BackgroundTaskManager,
        roles: dict[str, SubAgentRole],
        history: ConversationHistory,
    ) -> None:
        self._runner = runner
        self._task_manager = task_manager
        self._roles = roles
        self._history = history

    @property
    def name(self) -> str:
        return "sub_agent"

    @property
    def description(self) -> str:
        role_list = ", ".join(self._roles.keys()) if self._roles else "fork"
        return (
            "创建一个子工作器执行任务。可用角色: "
            f"{role_list}（或省略 role 使用 fork 模式继承当前对话）。"
            "后台运行: background=true。"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WRITE

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("task", "string", "要执行的任务描述"),
            ToolParameter("role", "string", "预定义角色名，省略则使用 fork 模式", required=False),
            ToolParameter("background", "boolean", "是否后台运行（fork 模式强制后台）", required=False),
        ]

    async def execute(
        self,
        task: str,
        role: str = "",
        background: bool = False,
    ) -> ToolResult:
        try:
            task = require_string(task, "task").strip()
            if not task:
                raise ValueError("task 不能为空")
            if role is not None:
                role = require_string(role, "role")
            if not isinstance(background, bool):
                raise ValueError("background 必须是布尔值")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))
        role_name = role.strip() if role else None
        is_fork = role_name is None

        # Fork mode: force background
        if is_fork:
            background = True

        # Validate role
        if role_name and role_name not in self._roles:
            available = ", ".join(self._roles.keys())
            return ToolResult(
                success=False, content="",
                error=f"未知角色: {role_name}。可用: {available}",
            )

        if background and not self._task_manager.can_start:
            return ToolResult(
                success=False,
                content="",
                error="后台子 Agent 已达到并发上限，请等待现有任务完成",
            )

        # Create task
        sub_task = self._task_manager.create(role_name, task, background=background)

        if background:
            running = asyncio.create_task(self._run_background(sub_task))
            self._task_manager.attach(sub_task.id, running)
            return ToolResult(
                success=True,
                content=f"后台任务 {sub_task.id} 已启动（角色: {role_name or 'fork'}）。使用 /tasks 查看状态。",
            )
        else:
            # Synchronous execution
            sub_task.start()
            try:
                result_text = await self._runner.run(sub_task, self._history)
                return ToolResult(success=True, content=result_text)
            except asyncio.CancelledError:
                sub_task.cancel()
                raise
            except Exception as exc:
                sub_task.fail(str(exc))
                return ToolResult(success=False, content="", error=str(exc))

    async def _run_background(self, task) -> None:
        """Run a task in background, injecting the result on completion."""
        task.start()
        try:
            await self._runner.run(task, self._history)
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            task.fail(str(exc))
        finally:
            self._task_manager.inject_result(task, self._history)

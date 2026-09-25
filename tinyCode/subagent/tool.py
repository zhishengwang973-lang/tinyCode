"""Sub-agent tool — the single sub_agent entry point."""

import asyncio
from collections.abc import Callable
import hashlib
import json
from typing import Any

from tinyCode.conversation.history import ConversationHistory
from tinyCode.subagent.manager import BackgroundTaskManager
from tinyCode.subagent.models import SubAgentRole, TaskStatus
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
        reload_roles: Callable[[], dict[str, SubAgentRole]] | None = None,
    ) -> None:
        self._runner = runner
        self._task_manager = task_manager
        self._roles = roles
        self._history = history
        self._reload_roles = reload_roles

    def _refresh_roles(self) -> None:
        if self._reload_roles is None:
            return
        refreshed = self._reload_roles()
        self._roles.clear()
        self._roles.update(refreshed)

    @property
    def name(self) -> str:
        return "sub_agent"

    @property
    def description(self) -> str:
        self._refresh_roles()
        role_list = "; ".join(
            f"{role.name}: {(role.description or '无描述')[:240]}"
            for role in list(self._roles.values())[:50]
        ) or "fork"
        return (
            "创建一个子工作器执行任务。可用角色: "
            f"{role_list}（或省略 role 使用 fork 模式继承当前对话）。"
            "后台运行: background=true；后台任务强制只读，适合 inspect 审查任务。"
        )

    @property
    def category(self) -> ToolCategory:
        # Keep orchestration serialized even when a particular worker is
        # read-only. Call-level side effects are reported by may_modify().
        return ToolCategory.WRITE

    @property
    def available_in_inspect(self) -> bool:
        return True

    def may_modify(self, params: dict[str, Any]) -> bool:
        manifest = self._capability_manifest(params)
        return bool(manifest["may_modify_workspace"])

    def security_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Scope persisted approval to the exact delegated capabilities."""
        manifest = self._capability_manifest(params)
        canonical = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        scoped = dict(params)
        scoped["command"] = f"subagent-capabilities:{fingerprint}"
        return scoped

    @property
    def timeout_exempt(self) -> bool:
        # Foreground workers enforce their own bounded wall-clock deadline;
        # the ordinary per-tool timeout would otherwise cancel them at 30s.
        return True

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("task", "string", "要执行的任务描述"),
            ToolParameter("role", "string", "预定义角色名，省略则使用 fork 模式", required=False),
            ToolParameter("background", "boolean", "是否后台运行（fork 模式强制后台）", required=False),
        ]

    def approval_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Expose the delegated worker's real capabilities in the HITL card."""
        visible = dict(params)
        visible["capabilities"] = self._capability_manifest(params)
        return visible

    def _capability_manifest(self, params: dict[str, Any]) -> dict[str, object]:
        self._refresh_roles()
        raw_role = params.get("role")
        role_name = (
            raw_role.strip()
            if isinstance(raw_role, str) and raw_role.strip()
            else None
        )
        raw_background = params.get("background", False)
        background = raw_background if isinstance(raw_background, bool) else False
        return self._runner.capability_manifest(
            role_name,
            background=background,
        )

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
        self._refresh_roles()
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
                content=(
                    f"后台任务 {sub_task.id} 已启动（角色: {role_name or 'fork'}）。"
                    f"当前任务依赖其结果时调用 sub_agent_wait(task_id='{sub_task.id}')；"
                    "不要在主任务重复扫描已经委派的范围；"
                    "用户可使用 /tasks 查看状态。"
                ),
            )
        else:
            # Synchronous execution
            self._task_manager.mark_started(sub_task)
            try:
                result_text = await self._runner.run(sub_task, self._history)
                self._task_manager.sync()
                return ToolResult(success=True, content=result_text)
            except asyncio.CancelledError:
                sub_task.cancel()
                self._task_manager.sync()
                raise
            except Exception as exc:
                sub_task.fail(str(exc))
                self._task_manager.sync()
                return ToolResult(success=False, content="", error=str(exc))

    async def _run_background(self, task) -> None:
        """Run a task in background, injecting the result on completion."""
        self._task_manager.mark_started(task)
        try:
            await self._runner.run(task, self._history)
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            task.fail(str(exc))
        finally:
            self._task_manager.publish(task)


class SubAgentWaitTool(BaseTool):
    """Join a background worker and return its terminal result."""

    def __init__(self, task_manager: BackgroundTaskManager) -> None:
        self._task_manager = task_manager

    @property
    def name(self) -> str:
        return "sub_agent_wait"

    @property
    def description(self) -> str:
        return (
            "等待指定后台 Subagent 完成并取得结果。仅在当前任务确实依赖"
            "该结果时调用；超时不会取消后台任务。"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def timeout_exempt(self) -> bool:
        # This tool has its own bounded timeout. The generic 30-second tool
        # deadline must not preempt a legitimate wait for a long worker.
        return True

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("task_id", "string", "sub_agent 返回的后台任务 ID"),
            ToolParameter(
                "timeout_seconds", "number", "等待秒数，范围 0.1-300，默认 60",
                required=False, default=60.0,
            ),
        ]

    async def execute(
        self, task_id: str, timeout_seconds: float = 60.0,
    ) -> ToolResult:
        try:
            task_id = require_string(task_id, "task_id").strip()
            if not task_id:
                raise ValueError("task_id 不能为空")
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not 0.1 <= float(timeout_seconds) <= 300
            ):
                raise ValueError("timeout_seconds 必须是 0.1 到 300 之间的数字")
            task = await self._task_manager.wait(
                task_id, timeout_seconds=float(timeout_seconds),
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, content="",
                error=f"等待后台任务 {task_id} 超时；任务仍在运行",
            )
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        if task is None:
            return ToolResult(
                success=False, content="", error=f"后台任务 {task_id} 不存在",
            )
        if task.status == TaskStatus.COMPLETED:
            return ToolResult(success=True, content=task.result)
        return ToolResult(
            success=False,
            content=task.result,
            error=f"后台任务 {task_id} 状态为 {task.status.value}",
        )

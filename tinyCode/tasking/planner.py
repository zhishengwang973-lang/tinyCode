"""Conservative task-plan generation and validation for normal agent turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tinyCode.agent.task_mode import TaskMode
from tinyCode.config.models import TaskPlanningConfig
from tinyCode.providers.base import BaseProvider, TokenUsage
from tinyCode.tasking.models import ExecutorKind, TaskNode, TaskNodeStatus, TaskPlan, TaskPlanStatus
from tinyCode.tasking.store import TaskPlanStore


MAX_PLAN_OUTPUT_CHARS = 12_000
MAX_TASKS = 8
_MAX_TEXT = 1_000
_COMPLEX_MARKERS = (
    "重构", "迁移", "跨模块", "多模块", "整个项目", "全量", "端到端", "架构",
    "全面", "系统性", "多个文件", "多个模块", "评测", "恢复机制", "并发",
)
_DELIVERABLE_MARKERS = ("测试", "文档", "验证", "审查", "实现", "修复", "优化", "集成")
_EXPLICIT_MARKERS = ("任务拆分", "拆分任务", "执行计划", "任务计划", "分步骤", "分阶段")


@dataclass(frozen=True)
class TaskPlanningResult:
    plan: TaskPlan | None = None
    model_requests: int = 0
    tokens: int = 0
    tokens_available: bool = False
    source: str = ""
    error: str = ""


class TaskPlanningService:
    """Create an optional durable DAG before a complex normal-agent turn."""

    def __init__(
        self, config: TaskPlanningConfig, provider: BaseProvider, store: TaskPlanStore,
    ) -> None:
        self.config = config
        self._provider = provider
        self._store = store
        self._active: TaskPlan | None = None

    @property
    def active_plan(self) -> TaskPlan | None:
        return self._active

    def should_plan(self, text: str, mode: TaskMode) -> bool:
        if not self.config.enabled or mode is TaskMode.DIRECT:
            return False
        normalized = text.strip()
        if any(marker in normalized for marker in _EXPLICIT_MARKERS):
            return True
        if len(normalized) < self.config.min_task_chars:
            return False
        complexity = sum(marker in normalized for marker in _COMPLEX_MARKERS)
        deliverables = sum(marker in normalized for marker in _DELIVERABLE_MARKERS)
        return complexity >= 1 and deliverables >= 2

    async def create_if_needed(
        self, text: str, mode: TaskMode,
    ) -> TaskPlanningResult:
        if not self.should_plan(text, mode):
            return TaskPlanningResult()
        nodes, error, tokens, tokens_available = await self._request_plan(text, mode)
        source = "model"
        if not nodes:
            nodes = self._fallback_plan(text, mode)
            source = "fallback"
        try:
            plan = TaskPlan.create(text, nodes, mode=mode.value, source=source)
            self._validate_plan(plan)
            self._start_next_ready(plan)
            self._store.save(plan)
            self._active = plan
            return TaskPlanningResult(
                plan=plan, model_requests=1, tokens=tokens,
                tokens_available=tokens_available, source=source, error=error,
            )
        except (OSError, ValueError) as exc:
            return TaskPlanningResult(
                model_requests=1, tokens=tokens, tokens_available=tokens_available,
                error=f"任务计划保存失败: {type(exc).__name__}: {exc}",
            )

    def clear_active(self, *, outcome: str) -> TaskPlan | None:
        plan = self._active
        self._active = None
        if plan is None:
            return None
        if outcome == "completed":
            if all(task.status.value in {"completed", "skipped"} for task in plan.tasks):
                plan.status = TaskPlanStatus.COMPLETED
            else:
                plan.status = TaskPlanStatus.NEEDS_REVIEW
                plan.error = "Agent 已结束，但仍有未确认节点；请根据最终回答和测试结果核验。"
        elif outcome == "cancelled":
            plan.status = TaskPlanStatus.INTERRUPTED
            plan.error = "任务被取消；恢复前请核验已发生的文件变更。"
        else:
            plan.status = TaskPlanStatus.FAILED
            plan.error = "Agent 未正常完成；请根据错误和工作区状态决定如何恢复。"
        try:
            self._store.save(plan)
        except OSError:
            pass
        return plan

    def render_active(self) -> str:
        return self._active.render() if self._active is not None else "当前没有活动任务计划"

    def render_plan(self, plan_id: str = "") -> str:
        if not plan_id and self._active is not None:
            return self._active.render()
        if plan_id:
            plan = self._store.load(plan_id)
            return plan.render() if plan is not None else f"任务计划 {plan_id} 不存在"
        plans = self._store.list_recent(limit=10)
        if not plans:
            return "尚无任务计划"
        return "\n".join(
            f"{plan.id} · {plan.status.value} · {plan.goal[:60]}"
            for plan in plans
        )

    def update_task(self, task_id: str, status: str, result: str = "") -> str:
        plan = self._active
        if plan is None:
            return "当前没有活动任务计划"
        try:
            next_status = TaskNodeStatus(status)
        except ValueError:
            return "任务状态必须是 pending、in_progress、blocked、completed、failed 或 skipped"
        task = next((item for item in plan.tasks if item.id == task_id), None)
        if task is None:
            return f"任务计划中不存在节点: {task_id}"
        if next_status is TaskNodeStatus.IN_PROGRESS:
            incomplete = []
            for dependency in task.depends_on:
                item = next((node for node in plan.tasks if node.id == dependency), None)
                if item is None or item.status is not TaskNodeStatus.COMPLETED:
                    incomplete.append(dependency)
            if incomplete:
                return "依赖尚未完成: " + ", ".join(incomplete)
        if task.status in {TaskNodeStatus.COMPLETED, TaskNodeStatus.SKIPPED}:
            return f"任务 {task_id} 已是终态，不能再次修改"
        task.status = next_status
        task.result = result.strip()[:2_000]
        if next_status is TaskNodeStatus.COMPLETED:
            self._start_next_ready(plan)
        try:
            self._store.save(plan)
        except OSError as exc:
            return f"任务状态保存失败: {type(exc).__name__}: {exc}"
        return f"任务 {task_id} 已更新为 {next_status.value}"

    def observe_successful_workspace_write(self, tool_name: str) -> TaskPlan | None:
        """Advance only the deterministic fallback's reconnaissance stage.

        A write proves that initial scope discovery has concluded, but it does
        not prove implementation or verification acceptance.  Rich/model-made
        plans remain explicit-only to avoid inventing progress.
        """
        if tool_name not in {
            "write_file", "edit_file", "apply_patch", "delete_file",
        }:
            return None
        plan = self._active
        if plan is None or plan.source != "fallback":
            return None
        scope = next((task for task in plan.tasks if task.id == "scope"), None)
        if scope is None or scope.status is not TaskNodeStatus.IN_PROGRESS:
            return None
        scope.status = TaskNodeStatus.COMPLETED
        scope.result = "已完成范围确认，开始执行首次工作区写入。"
        self._start_next_ready(plan)
        try:
            self._store.save(plan)
        except OSError:
            return None
        return plan

    async def _request_plan(
        self, text: str, mode: TaskMode,
    ) -> tuple[list[TaskNode], str, int, bool]:
        prompt = (
            "你是 TinyCode 的任务规划器。将用户目标拆成 2 至 8 个有向无环任务。"
            "只输出 JSON 数组，禁止 Markdown、解释或工具调用。每项包含："
            "id、title、description、depends_on、read_scope、write_scope、acceptance、executor。"
            "depends_on 只能引用前面任务的 id。executor 只能是 main 或 subagent；"
            "subagent 仅用于独立只读分析，任何写入任务一律 main。"
            "文件范围只能基于用户已明确提供的信息；不确定就使用空数组，不能编造路径。"
            "避免重叠写入范围；最终验证必须依赖所有写入任务。"
            f"\n任务模式: {mode.value}\n用户目标: {text}"
        )
        chunks: list[str] = []
        error = ""
        try:
            begin = getattr(self._provider, "begin_request", None)
            if begin:
                begin()
            async for chunk in self._provider.chat_stream([{"role": "user", "content": prompt}]):
                if not isinstance(chunk, str):
                    continue
                if sum(map(len, chunks)) + len(chunk) > MAX_PLAN_OUTPUT_CHARS:
                    raise ValueError("任务规划输出超过上限")
                if not chunk.startswith("<<"):
                    chunks.append(chunk)
            nodes = self._parse_nodes("".join(chunks))
        except Exception as exc:
            nodes = []
            error = f"规划模型不可用: {type(exc).__name__}: {exc}"
        usage = TokenUsage.from_raw(getattr(self._provider, "last_usage", None))
        return nodes, error, usage.total_tokens, usage.available

    def _parse_nodes(self, text: str) -> list[TaskNode]:
        match = re.search(r"\[[\s\S]*\]", text.strip())
        if not match:
            return []
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list) or not 2 <= len(raw) <= self.config.max_tasks:
            return []
        nodes: list[TaskNode] = []
        known: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                return []
            task_id, title, description = item.get("id"), item.get("title"), item.get("description")
            deps = _strings(item.get("depends_on"), 16, 80)
            if (
                not isinstance(task_id, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", task_id)
                or task_id in known or not isinstance(title, str) or not title.strip()
                or not isinstance(description, str) or not description.strip()
                or any(dependency not in known for dependency in deps)
            ):
                return []
            executor = item.get("executor", "main")
            if executor not in {ExecutorKind.MAIN.value, ExecutorKind.SUBAGENT.value}:
                return []
            write_scope = _strings(item.get("write_scope"), 12, 300)
            if executor == ExecutorKind.SUBAGENT.value and write_scope:
                return []
            nodes.append(TaskNode(
                id=task_id, title=title.strip()[:160], description=description.strip()[:_MAX_TEXT],
                depends_on=deps, read_scope=_strings(item.get("read_scope"), 20, 300),
                write_scope=write_scope, acceptance=_strings(item.get("acceptance"), 10, 500),
                executor=ExecutorKind(executor),
            ))
            known.add(task_id)
        return nodes

    def _fallback_plan(self, text: str, mode: TaskMode) -> list[TaskNode]:
        scope = TaskNode(
            id="scope", title="确认范围与现状",
            description="读取必要的项目结构和相关实现，确认影响范围后再进行后续动作。",
            acceptance=["已确认涉及模块、约束和风险"],
        )
        execute = TaskNode(
            id="execute", title="执行核心任务", description=text[:_MAX_TEXT],
            depends_on=[scope.id], acceptance=["核心目标已完成或已明确阻塞原因"],
        )
        if mode is TaskMode.INSPECT:
            execute.description = "在只读范围内分析并回答：" + text[:_MAX_TEXT]
        verify = TaskNode(
            id="verify", title="验证与交付",
            description="核对结果、运行必要验证并给出限制说明。",
            depends_on=[execute.id], acceptance=["结果已核对", "最终回答说明验证情况"],
        )
        return [scope, execute, verify]

    @staticmethod
    def _start_next_ready(plan: TaskPlan) -> None:
        for task in plan.tasks:
            if task.status is not TaskNodeStatus.PENDING:
                continue
            dependencies = [
                next((item for item in plan.tasks if item.id == dependency), None)
                for dependency in task.depends_on
            ]
            if all(item is not None and item.status is TaskNodeStatus.COMPLETED for item in dependencies):
                task.status = TaskNodeStatus.IN_PROGRESS
                return

    @staticmethod
    def _validate_plan(plan: TaskPlan) -> None:
        if not 2 <= len(plan.tasks) <= MAX_TASKS:
            raise ValueError("任务计划数量不在允许范围内")
        ids = {task.id for task in plan.tasks}
        if len(ids) != len(plan.tasks):
            raise ValueError("任务 ID 必须唯一")
        for task in plan.tasks:
            if any(dep not in ids or dep == task.id for dep in task.depends_on):
                raise ValueError("任务依赖无效")
        for index, left in enumerate(plan.tasks):
            for right in plan.tasks[index + 1:]:
                if not _write_scopes_overlap(left.write_scope, right.write_scope):
                    continue
                if not (_depends_on(left, right.id, plan.tasks) or _depends_on(right, left.id, plan.tasks)):
                    raise ValueError(
                        f"并行任务写入范围重叠: {left.id} 与 {right.id}"
                    )


def _strings(value: object, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list) or len(value) > max_items:
        return []
    return [item.strip()[:max_chars] for item in value if isinstance(item, str) and item.strip()]


def _depends_on(task: TaskNode, ancestor_id: str, tasks: list[TaskNode]) -> bool:
    by_id = {item.id: item for item in tasks}
    pending = list(task.depends_on)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == ancestor_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        parent = by_id.get(current)
        if parent is not None:
            pending.extend(parent.depends_on)
    return False


def _write_scopes_overlap(left: list[str], right: list[str]) -> bool:
    for first in left:
        first = first.strip("/")
        for second in right:
            second = second.strip("/")
            if first and second and (first == second or first.startswith(second + "/") or second.startswith(first + "/")):
                return True
    return False

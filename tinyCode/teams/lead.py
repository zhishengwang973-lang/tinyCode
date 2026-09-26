"""Lead agent — orchestrates team: split, assign, merge, synthesize."""

import asyncio
import inspect
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.member import TeamMember
from tinyCode.teams.models import MessageType, TaskStatus, TeamDef, TeamMessage
from tinyCode.teams.tasks import SharedTaskList
from tinyCode.providers.base import TokenUsage


# A Team plan is a small JSON routing artifact.  A bounded response prevents
# a malformed planner stream from becoming an outsized, duplicated context.
MAX_TEAM_PLAN_CHARS = 16_000


class LeadAgent:
    """Orchestrator: decomposes goals, assigns to members, merges results."""

    def __init__(
        self,
        team_def: TeamDef,
        team_dir: Path,
        members: dict[str, TeamMember],
        task_list: SharedTaskList,
        merger,  # GitMerger
        provider: Any = None,
        lead_instructions: str = "",
        progress: Callable[[str], Awaitable[None] | None] | None = None,
        worktree_baselines: dict[str, dict[str, str]] | None = None,
        validation_commands: list[str] | None = None,
    ) -> None:
        self._def = team_def
        self._dir = team_dir
        self._members = members
        self._tasks = task_list
        self._merger = merger
        self._provider = provider
        self._lead_instructions = lead_instructions.strip()
        self._progress = progress
        self._worktree_baselines = worktree_baselines or {}
        self._validation_commands = validation_commands or []
        self._mailbox = Mailbox(team_dir, "lead")
        self._last_mail_id = ""
        self._member_reports: list[str] = []
        self._active = True
        self._run_id = ""
        self._running: set[asyncio.Task] = set()
        self._planner_model_requests = 0
        self._planner_tokens = 0
        self._planner_tokens_available = False
        self._member_turns = 0
        self._member_model_requests = 0
        self._member_tokens = 0
        self._member_tokens_available = False
        self._merge_failed = False

    # -- public API -----------------------------------------------------------

    async def execute(self, goal: str) -> str:
        """Execute a team goal end-to-end."""
        try:
            self._reset_run_metrics()
            self._run_id = uuid.uuid4().hex
            self._tasks.set_active_run(self._run_id)
            await self._emit_progress("正在拆解 Team 任务")
            await self._decompose(goal)

            await self._emit_progress(
                f"已创建 {len(self._current_tasks())} 个任务，开始并行执行"
            )
            await self._dispatch_loop()
            self._drain_lead_mail()

            await self._emit_progress("成员任务结束，开始验证并合并工作树")
            merge_results = await self._merge_all()

            await self._emit_progress("Team 执行结束")
            has_task_failure = any(
                task.status == TaskStatus.FAILED for task in self._current_tasks()
            )
            heading = (
                "## Team 部分完成"
                if has_task_failure or self._merge_failed
                else "## Team 执行完成"
            )
            reports = (
                "\n成员消息:\n" + "\n".join(
                    f"- {report}" for report in self._member_reports[-20:]
                )
                if self._member_reports else ""
            )
            return (
                f"{heading}\n\n{merge_results}\n\n"
                f"任务统计: {self._task_summary()}\n{self._metrics_summary()}"
                f"{reports}"
            )
        finally:
            running = list(self._running)
            for task in running:
                if not task.done():
                    task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
            self._running.clear()
            closers = []
            for member in self._members.values():
                close = getattr(member, "close", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        closers.append(result)
            if closers:
                await asyncio.gather(*closers, return_exceptions=True)

    # -- internals ------------------------------------------------------------

    async def _decompose(self, goal: str) -> None:
        """Create a real member-specific plan, with a deterministic fallback."""
        plan = await self._request_plan(goal)
        if not plan:
            plan = self._fallback_plan(goal)

        created_ids: list[str] = []
        for index, item in enumerate(plan):
            dependency_ids = [
                created_ids[dep]
                for dep in item.get("depends_on", [])
                if isinstance(dep, int) and 0 <= dep < len(created_ids)
            ]
            task = self._tasks.create(
                name=item["name"],
                description=item["description"],
                depends_on=dependency_ids,
                preferred_member=item.get("member", ""),
                run_id=self._run_id,
            )
            created_ids.append(task.id)

    async def _request_plan(self, goal: str) -> list[dict[str, Any]]:
        if self._provider is None:
            return []
        members = [
            {"name": member.name, "role": member.role or "未指定"}
            for member in self._def.members
        ]
        lead_context = (
            f"\nLead 角色要求:\n{self._lead_instructions}\n"
            if self._lead_instructions else ""
        )
        prompt = (
            "你是 Agent Team 的调度 Lead。请把目标拆成互不重复、边界清晰、"
            "可独立验收的任务，并为每个任务指定最合适的成员。"
            "只输出 JSON 数组，不要 Markdown。每项必须包含 name、description、"
            "member、depends_on；depends_on 是此前任务的零基索引数组。"
            "description 必须明确目标、验收条件以及独占的文件或模块范围；"
            "无法划分独占写入范围的任务应标记为只读分析。"
            "不要让多个成员修改重叠文件或重复实现同一内容。\n"
            f"成员: {json.dumps(members, ensure_ascii=False)}\n"
            f"目标: {goal}{lead_context}"
        )
        chunks: list[str] = []
        output_chars = 0
        try:
            self._planner_model_requests += 1
            begin = getattr(self._provider, "begin_request", None)
            if begin:
                begin()
            async for chunk in self._provider.chat_stream(
                [{"role": "user", "content": prompt}],
            ):
                if isinstance(chunk, str):
                    output_chars += len(chunk)
                    if output_chars > MAX_TEAM_PLAN_CHARS:
                        raise ValueError(
                            f"Team 规划模型流超过 {MAX_TEAM_PLAN_CHARS} 字符上限"
                        )
                    if not chunk.startswith("<<"):
                        chunks.append(chunk)
        except Exception:
            return []
        finally:
            usage = TokenUsage.from_raw(getattr(self._provider, "last_usage", None))
            if usage.available:
                self._planner_tokens_available = True
                self._planner_tokens += usage.total_tokens
        return self._parse_plan("".join(chunks))

    def _parse_plan(self, raw: str) -> list[dict[str, Any]]:
        match = re.search(r"\[[\s\S]*\]", raw.strip())
        if not match:
            return []
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list) or not value or len(value) > 32:
            return []
        member_names = set(self._members)
        plan: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                return []
            name = item.get("name")
            description = item.get("description")
            member = item.get("member")
            dependencies = item.get("depends_on", [])
            if (
                not isinstance(name, str) or not name.strip()
                or not isinstance(description, str) or not description.strip()
                or member not in member_names
                or not isinstance(dependencies, list)
                or not all(
                    isinstance(dep, int) and not isinstance(dep, bool)
                    and 0 <= dep < index
                    for dep in dependencies
                )
            ):
                return []
            plan.append({
                "name": name.strip()[:120],
                "description": description.strip(),
                "member": member,
                "depends_on": dependencies,
            })
        return plan

    def _fallback_plan(self, goal: str) -> list[dict[str, Any]]:
        scopes = [
            "负责核心分析与主要实现；先声明独占文件范围，再完成修改并自测",
            "负责只读复现、边界分析和测试方案；除非获得独占文件范围，不修改核心实现",
            "负责只读集成审查、兼容性和最终验收；报告遗漏，不修改其他成员负责的文件",
        ]
        plan: list[dict[str, Any]] = []
        for index, member in enumerate(self._def.members):
            role = f"角色 {member.role}；" if member.role else ""
            scope = scopes[min(index, len(scopes) - 1)]
            plan.append({
                "name": f"{member.name}-{index + 1}",
                "description": f"总目标：{goal}\n你的专属范围：{role}{scope}。避免重复其他成员的范围。",
                "member": member.name,
                "depends_on": [],
            })
        return plan

    async def _dispatch_loop(self) -> None:
        """Incrementally dispatch ready tasks to idle members."""
        running_by_member: dict[str, asyncio.Task] = {}
        while self._active:
            self._drain_lead_mail()
            ready = [task for task in self._tasks.ready_tasks() if self._is_current(task)]
            idle = [
                member for member in self._members.values()
                if member.defn.name not in running_by_member
            ]

            for task in ready:
                if not idle:
                    break
                member = next(
                    (candidate for candidate in idle
                     if candidate.defn.name == task.preferred_member),
                    None,
                )
                if task.preferred_member and member is None:
                    continue
                member = member or idle[0]
                idle.remove(member)
                self._tasks.assign(task.id, member.defn.name)
                await self._emit_progress(
                    f"{member.defn.name} 开始任务：{task.name}"
                )
                running = asyncio.create_task(self._run_member_task(member, task))
                running_by_member[member.defn.name] = running
                self._running.add(running)

            current = self._current_tasks()
            pending = [task for task in current if task.status == TaskStatus.PENDING]
            in_progress = [task for task in current if task.status == TaskStatus.IN_PROGRESS]
            if not pending and not in_progress:
                break

            if not running_by_member:
                # No ready task and nobody running means dependency deadlock.
                for task in pending:
                    self._tasks.update(
                        task.id,
                        status=TaskStatus.FAILED,
                        result="任务依赖无法满足",
                    )
                break

            done, _ = await asyncio.wait(
                running_by_member.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for finished in done:
                self._running.discard(finished)
                member_name = next(
                    name for name, task in running_by_member.items()
                    if task is finished
                )
                running_by_member.pop(member_name)
                await finished
            self._drain_lead_mail()

    async def _run_member_task(self, member: TeamMember, task) -> None:
        try:
            result = await member.run(task.description)
            self._tasks.complete(task.id, result)
            await self._emit_progress(
                f"{member.defn.name} 已完成：{task.name}"
            )
        except asyncio.CancelledError:
            self._tasks.update(
                task.id, status=TaskStatus.FAILED, result="Team 执行已取消",
            )
            raise
        except Exception as exc:
            self._tasks.update(task.id, status=TaskStatus.FAILED, result=str(exc))
            await self._emit_progress(
                f"{member.defn.name} 任务失败：{task.name}（{exc}）"
            )
        finally:
            self._member_turns += max(0, getattr(member, "last_turns", 0))
            self._member_model_requests += max(
                0, getattr(member, "last_model_requests", 0),
            )
            if getattr(member, "last_tokens_available", False):
                self._member_tokens_available = True
                self._member_tokens += max(0, getattr(member, "last_tokens", 0))

    async def _merge_all(self) -> str:
        """Incrementally merge each member's worktree."""
        results: list[str] = []
        for name, member in self._members.items():
            member_tasks = [
                task for task in self._current_tasks()
                if task.assigned_to == name
            ]
            if not member_tasks:
                results.append(f"  {name}: 本轮未分配任务，已跳过合并")
                continue
            if any(task.status == TaskStatus.FAILED for task in member_tasks):
                self._merge_failed = True
                results.append(f"  {name}: 存在失败任务，已跳过自动合并")
                continue
            wt = member.defn.worktree
            if not wt:
                continue
            baseline = self._worktree_baselines.get(name, {})
            branch = baseline.get("branch", "")
            branch_reader = getattr(self._merger, "worktree_branch", None)
            if branch_reader is not None:
                branch_ok, actual_branch = await branch_reader(member.workspace)
                if not branch_ok:
                    self._merge_failed = True
                    results.append(f"  {name}: 无法读取实际分支: {actual_branch}")
                    continue
                if branch and branch != actual_branch:
                    self._merge_failed = True
                    results.append(
                        f"  {name}: 分支在任务期间从 {branch} 变为 {actual_branch}，已拒绝合并"
                    )
                    continue
                branch = actual_branch
            branch = branch or f"tinyCode/{wt.replace('/', '-')}"
            validator = getattr(self._merger, "validate_worktree", None)
            if validator is not None:
                valid, validation_msg = await validator(
                    member.workspace, self._validation_commands,
                )
                if not valid:
                    self._merge_failed = True
                    results.append(
                        f"  {name} ({branch}): {validation_msg}，已跳过合并"
                    )
                    continue
            prepare = getattr(self._merger, "prepare_worktree", None)
            if prepare is not None:
                prepared, prepare_msg = await prepare(member.workspace, name)
                if not prepared:
                    self._merge_failed = True
                    results.append(f"  {name} ({branch}): {prepare_msg}")
                    continue
            head_reader = getattr(self._merger, "worktree_head", None)
            if head_reader is not None and baseline.get("head"):
                head_ok, current_head = await head_reader(member.workspace)
                if not head_ok:
                    self._merge_failed = True
                    results.append(f"  {name} ({branch}): {current_head}")
                    continue
                if current_head == baseline["head"]:
                    results.append(f"  {name} ({branch}): 本轮没有文件变更")
                    continue
            if self._def.merge_policy == "review":
                results.append(f"  {name} ({branch}): 变更已提交，等待统一审核")
                continue
            if self._def.merge_policy == "none":
                results.append(f"  {name} ({branch}): 变更已保留，未执行合并")
                continue
            target_branch = baseline.get("target_branch", "")
            if target_branch:
                ok, msg = await self._merger.merge(branch, target_branch)
            else:
                ok, msg = await self._merger.merge(branch)
            if not ok:
                self._merge_failed = True
            results.append(f"  {name} ({branch}): {msg}")
        return "\n".join(results)

    def _task_summary(self) -> str:
        all_tasks = self._current_tasks()
        done = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)
        return f"{done}/{len(all_tasks)} 完成，{failed} 失败"

    def _metrics_summary(self) -> str:
        requests = self._planner_model_requests + self._member_model_requests
        tokens_available = (
            self._planner_tokens_available or self._member_tokens_available
        )
        tokens = self._planner_tokens + self._member_tokens
        token_text = f"{tokens:,}" if tokens_available else "不可用"
        return (
            "Team 统计: "
            f"Turn {self._member_turns} · 模型请求 {requests} 次 · "
            f"Token {token_text}"
        )

    def _reset_run_metrics(self) -> None:
        self._planner_model_requests = 0
        self._planner_tokens = 0
        self._planner_tokens_available = False
        self._member_turns = 0
        self._member_model_requests = 0
        self._member_tokens = 0
        self._member_tokens_available = False
        self._merge_failed = False

    def _is_current(self, task) -> bool:
        return bool(self._run_id) and task.run_id == self._run_id

    def _current_tasks(self) -> list:
        return self._tasks.list_for_run(self._run_id)

    async def _emit_progress(self, message: str) -> None:
        if self._progress is None:
            return
        result = self._progress(message)
        if inspect.isawaitable(result):
            await result

    def _drain_lead_mail(self) -> None:
        messages = self._mailbox.read_new(self._last_mail_id)
        if messages:
            self._last_mail_id = messages[-1].id
        for message in messages:
            if message.msg_type not in {MessageType.TEXT, MessageType.BROADCAST}:
                continue
            sender = message.from_member or "member"
            self._member_reports.append(f"{sender}: {message.content}")

    # -- messaging ------------------------------------------------------------

    def send_to_member(self, member_name: str, content: str) -> None:
        msg = TeamMessage(from_member="lead", to_member=member_name,
                          msg_type=MessageType.TEXT, content=content)
        self._mailbox.send(msg)

    def broadcast(self, content: str) -> None:
        msg = TeamMessage(from_member="lead", msg_type=MessageType.BROADCAST,
                          content=content)
        all_names = list(self._members.keys())
        self._mailbox.broadcast(msg, all_names)

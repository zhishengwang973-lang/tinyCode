"""Team orchestration service — builds and runs LeadAgent from a team name."""

import asyncio
import os
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

from tinyCode.teams.lead import LeadAgent
from tinyCode.teams.lease import TeamRunLease
from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.member import TeamMember
from tinyCode.teams.merger import GitMerger
from tinyCode.teams.persistence import get_team_dir, load_team_def
from tinyCode.teams.scheduler import DispatchScheduler
from tinyCode.teams.tasks import SharedTaskList
from tinyCode.teams.tools import create_team_tools
from tinyCode.subagent.filter import ToolFilter
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tools.base import ToolCategory
from tinyCode.tools.tool_result_read import ToolResultReadTool
from tinyCode.tools.tool_result_search import ToolResultSearchTool
from tinyCode.conversation.truncator import default_storage_dir
from tinyCode.security import SecurityLevel
from tinyCode.worktree.validator import resolve_worktree_path, validate_name


async def run_team(
    name: str,
    goal: str,
    *,
    provider: Any = None,
    tool_registry: Any = None,
    tool_executor: Any = None,
    merger: Any = None,
    repo_root: Path | None = None,
    roles: dict[str, Any] | None = None,
    preapproved: bool = False,
    progress: Callable[[str], Awaitable[None] | None] | None = None,
    definition: Any = None,
    state_dir: Path | None = None,
) -> str:
    team_def = definition or load_team_def(name)
    if team_def is None:
        return f"Team '{name}' 不存在"
    if not team_def.members:
        return f"Team '{name}' 没有可用成员"
    if provider is None or tool_registry is None or tool_executor is None:
        return "Team 运行需要 provider、tool_registry 和 tool_executor"
    unsupported = [member.name for member in team_def.members if member.backend != "coro"]
    if unsupported:
        return "当前仅支持 Team backend='coro'；不支持: " + ", ".join(unsupported)
    approval_members = [member.name for member in team_def.members if member.needs_approval]
    if approval_members and not preapproved:
        return "以下 Team 成员需要显式批准后才能运行: " + ", ".join(approval_members)
    if len(team_def.members) > 1 and any(not member.worktree for member in team_def.members):
        return "多成员 Team 必须为每个成员配置独立 worktree"

    team_dir = state_dir or get_team_dir(name)
    team_dir.mkdir(parents=True, exist_ok=True)
    task_list = SharedTaskList(team_dir)
    task_load_error = getattr(task_list, "load_error", "")
    if isinstance(task_load_error, str) and task_load_error:
        return f"Team '{name}' 的任务状态损坏，已拒绝覆盖: {task_load_error}"
    root = (repo_root or Path.cwd()).resolve()
    member_names = [member.name for member in team_def.members]
    members: dict[str, TeamMember] = {}
    merge_service = merger or GitMerger(
        provider,
        root,
        allow_llm_conflicts=team_def.allow_llm_conflict_resolution,
    )
    worktree_baselines: dict[str, dict[str, str]] = {}
    for member in team_def.members:
        workspace = root
        if member.worktree:
            valid, error = validate_name(member.worktree)
            if not valid:
                return f"Team 成员 '{member.name}' 的 worktree 无效: {error}"
            workspace = resolve_worktree_path(root, member.worktree)
            if not workspace.is_dir():
                return f"Team 成员 '{member.name}' 的 worktree 不存在: {workspace}"
            try:
                os.utime(workspace, None)
            except OSError as exc:
                return f"Team 成员 '{member.name}' 的 worktree 无法标记为活跃: {exc}"
            inspect_worktree = getattr(merge_service, "inspect_worktree", None)
            if inspect_worktree is not None:
                baseline_ok, baseline = await inspect_worktree(workspace)
                if not baseline_ok:
                    return (
                        f"Team 成员 '{member.name}' 的 worktree 无法作为安全基线: "
                        f"{baseline}"
                    )
                if isinstance(baseline, dict):
                    worktree_baselines[member.name] = baseline

        # Each member gets its own registry so collaboration tools can be
        # bound to that member without mutating the main agent's registry.
        role = (roles or {}).get(member.role)
        parent_tools = [tool.name for tool in tool_registry.list_tools()]
        read_tools = {
            tool.name for tool in tool_registry.list_tools()
            if tool.category == ToolCategory.READ
        }
        allowed_tools = set(ToolFilter(
            role,
            parent_tools=parent_tools,
            read_tools=read_tools,
        ).filter(parent_tools))

        member_registry = ToolRegistry()
        for tool in tool_registry.list_tools():
            if tool.name in {
                "sub_agent", "skill_loader", "request_user_input",
            } or tool.name.startswith("team_") or tool.name not in allowed_tools:
                continue
            if tool.name == "tool_result_read":
                member_registry.register(ToolResultReadTool(default_storage_dir(workspace)))
            elif tool.name == "tool_result_search":
                member_registry.register(ToolResultSearchTool(default_storage_dir(workspace)))
            else:
                member_registry.register(tool)
        mailbox = Mailbox(team_dir, member.name)
        for team_tool in create_team_tools(
            team_dir, task_list, mailbox, member.name, member_names,
        ):
            member_registry.register(team_tool)

        permission_name = getattr(role, "permission", "normal")
        try:
            member_security_level = SecurityLevel(permission_name)
        except ValueError:
            member_security_level = SecurityLevel.NORMAL

        role_max_rounds = getattr(role, "max_rounds", team_def.max_rounds_per_member)
        if not isinstance(role_max_rounds, int) or isinstance(role_max_rounds, bool):
            role_max_rounds = team_def.max_rounds_per_member
        member_max_rounds = max(
            1, min(team_def.max_rounds_per_member, role_max_rounds),
        )
        role_timeout = getattr(role, "timeout_seconds", 600.0)
        if (
            isinstance(role_timeout, bool)
            or not isinstance(role_timeout, (int, float))
            or role_timeout <= 0
        ):
            role_timeout = 600.0
        role_prompt = getattr(role, "system_prompt", "")
        role_model = getattr(role, "model", "")
        effective_model = member.model or (
            role_model if isinstance(role_model, str) else ""
        )

        members[member.name] = TeamMember(
            member_def=member,
            team_dir=team_dir,
            provider=provider,
            tool_registry=member_registry,
            tool_executor=tool_executor,
            max_rounds=member_max_rounds,
            workspace=workspace,
            security_level=member_security_level,
            preapproved=preapproved,
            instructions=role_prompt if isinstance(role_prompt, str) else "",
            model=effective_model,
            timeout_seconds=float(role_timeout),
        )
    lease = TeamRunLease(team_dir)
    lease_ok, lease_error = lease.acquire()
    if not lease_ok:
        return lease_error
    try:
        task_list.reconcile_interrupted()
        lead = LeadAgent(
            team_def=team_def,
            team_dir=team_dir,
            members=members,
            task_list=task_list,
            merger=merge_service,
            provider=provider,
            lead_instructions=_lead_instructions(team_def, roles),
            progress=progress,
            worktree_baselines=worktree_baselines,
            validation_commands=team_def.validation_commands,
        )
        try:
            return await asyncio.wait_for(
                lead.execute(goal), timeout=team_def.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return f"Team 执行超过总时限 {team_def.timeout_seconds:g}s，已安全取消运行中的成员"
    finally:
        lease.release()


def _lead_instructions(team_def, roles: dict[str, Any] | None) -> str:
    parts: list[str] = []
    role = (roles or {}).get(team_def.lead_role)
    role_prompt = getattr(role, "system_prompt", "")
    if isinstance(role_prompt, str) and role_prompt.strip():
        parts.append(role_prompt.strip())
    if team_def.dispatch_mode:
        scheduler = DispatchScheduler()
        scheduler.set_lock_1(True)
        scheduler.set_lock_2(True)
        parts.append(scheduler.get_workflow_instructions())
    return "\n\n".join(parts)

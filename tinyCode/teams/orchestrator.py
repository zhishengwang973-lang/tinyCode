"""Team orchestration service — builds and runs LeadAgent from a team name."""

import os
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

from tinyCode.teams.lead import LeadAgent
from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.member import TeamMember
from tinyCode.teams.merger import GitMerger
from tinyCode.teams.persistence import get_team_dir, load_team_def
from tinyCode.teams.scheduler import DispatchScheduler
from tinyCode.teams.tasks import SharedTaskList
from tinyCode.teams.tools import create_team_tools
from tinyCode.tools.registry import ToolRegistry
from tinyCode.security import SecurityLevel
from tinyCode.worktree.validator import name_to_dirname, validate_name


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
) -> str:
    team_def = load_team_def(name)
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

    team_dir = get_team_dir(name)
    task_list = SharedTaskList(team_dir)
    task_load_error = getattr(task_list, "load_error", "")
    if isinstance(task_load_error, str) and task_load_error:
        return f"Team '{name}' 的任务状态损坏，已拒绝覆盖: {task_load_error}"
    root = (repo_root or Path.cwd()).resolve()
    member_names = [member.name for member in team_def.members]
    members: dict[str, TeamMember] = {}
    for member in team_def.members:
        workspace = root
        if member.worktree:
            valid, error = validate_name(member.worktree)
            if not valid:
                return f"Team 成员 '{member.name}' 的 worktree 无效: {error}"
            workspace = root / ".tinyCode" / "worktrees" / name_to_dirname(member.worktree)
            if not workspace.is_dir():
                return f"Team 成员 '{member.name}' 的 worktree 不存在: {workspace}"
            try:
                os.utime(workspace, None)
            except OSError as exc:
                return f"Team 成员 '{member.name}' 的 worktree 无法标记为活跃: {exc}"

        # Each member gets its own registry so collaboration tools can be
        # bound to that member without mutating the main agent's registry.
        member_registry = ToolRegistry()
        for tool in tool_registry.list_tools():
            if tool.name in {
                "sub_agent", "skill_loader", "request_user_input",
            } or tool.name.startswith("team_"):
                continue
            member_registry.register(tool)
        mailbox = Mailbox(team_dir, member.name)
        for team_tool in create_team_tools(
            team_dir, task_list, mailbox, member.name, member_names,
        ):
            member_registry.register(team_tool)

        role = (roles or {}).get(member.role)
        permission_name = getattr(role, "permission", "normal")
        try:
            member_security_level = SecurityLevel(permission_name)
        except ValueError:
            member_security_level = SecurityLevel.NORMAL

        members[member.name] = TeamMember(
            member_def=member,
            team_dir=team_dir,
            provider=provider,
            tool_registry=member_registry,
            tool_executor=tool_executor,
            max_rounds=team_def.max_rounds_per_member,
            workspace=workspace,
            security_level=member_security_level,
            preapproved=preapproved,
        )
    lead = LeadAgent(
        team_def=team_def,
        team_dir=team_dir,
        members=members,
        task_list=task_list,
        merger=merger or GitMerger(provider, root),
        provider=provider,
        lead_instructions=_lead_instructions(team_def, roles),
        progress=progress,
    )
    return await lead.execute(goal)


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

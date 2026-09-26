"""Team management command."""

from collections.abc import Awaitable, Callable

from tinyCode.commands.types import CommandMeta, CommandType
from tinyCode.teams.orchestrator import run_team
from tinyCode.teams.persistence import get_team_dir, list_team_defs, load_team_def

TeamRunner = Callable[[str, str], Awaitable[str]]
Confirmer = Callable[[str], Awaitable[bool]]


def create(
    runner: TeamRunner | None = None,
    confirmer: Confirmer | None = None,
    review_service=None,
) -> CommandMeta:
    team_runner = runner or run_team

    async def handler(args: list[str]) -> str:
        sub = args[0].lower() if args else "list"
        if sub == "list":
            teams = list_team_defs()
            if not teams:
                return "没有已定义的 Team。在 ~/.tinyCode/teams/ 创建 JSON 定义文件。"
            lines = ["已定义的 Team:"]
            for name in teams:
                tdef = load_team_def(name)
                if tdef:
                    count = len(tdef.members)
                    lines.append(f"  {name} — {tdef.description} ({count} 成员)")
            return "\n".join(lines)

        elif sub == "show":
            if len(args) < 2:
                return "用法: /team show <名称>"
            tdef = load_team_def(args[1])
            if tdef is None:
                return f"Team '{args[1]}' 不存在"
            members = "\n".join(
                f"  - {m.name} (role={m.role}, backend={m.backend}, wt={m.worktree})"
                for m in tdef.members
            )
            return (
                f"Team: {tdef.name}\n描述: {tdef.description}\n"
                f"Lead 角色: {tdef.lead_role}\n调度模式: {tdef.dispatch_mode}\n\n"
                f"成员:\n{members}"
            )

        elif sub == "dir":
            if len(args) < 2:
                return "用法: /team dir <名称>"
            d = get_team_dir(args[1])
            return f"Team 工作目录: {d}"

        elif sub == "run":
            if len(args) < 3:
                return "用法: /team run <名称> <目标>"
            team_name = args[1]
            goal = " ".join(args[2:]).strip()
            if not goal:
                return "用法: /team run <名称> <目标>"
            if confirmer is not None:
                approved = await confirmer(
                    f"Team '{team_name}' 将启动多个 Agent，并可能修改工作树、执行命令和提交 Git 变更。是否继续？"
                )
                if not approved:
                    return "Team 执行已取消"
            return await team_runner(team_name, goal)

        elif sub == "review":
            if review_service is None:
                return "当前运行环境未启用 Team 审核服务"
            action = args[1].lower() if len(args) > 1 else "list"
            if action == "list":
                records = review_service.list_reviews()
                if not records:
                    return "没有 Team 审核记录"
                return "Team 审核记录:\n" + "\n".join(
                    f"  {record.run_id} · {record.status} · {record.goal[:60]}"
                    for record in records
                )
            if action == "show" and len(args) >= 3:
                return review_service.show_review(args[2])
            if action in {"apply", "discard"} and len(args) >= 3:
                run_id = args[2]
                if confirmer is not None:
                    verb = "应用到当前分支" if action == "apply" else "永久丢弃"
                    approved = await confirmer(
                        f"即将{verb} Team 审核 {run_id}，是否继续？"
                    )
                    if not approved:
                        return "Team 审核操作已取消"
                operation = (
                    review_service.apply if action == "apply"
                    else review_service.discard
                )
                _ok, message = await operation(run_id)
                return message
            return (
                "用法: /team review [list | show <编号> | "
                "apply <编号> | discard <编号>]"
            )

        return f"未知子命令: {args[0]}。可用: list, show, dir, run, review"

    return CommandMeta(
        name="team",
        aliases=["tm"],
        description="管理 Agent Team 与待审核变更",
        usage=(
            "/team [list | show <名称> | dir <名称> | run <名称> <目标> | "
            "review [list|show|apply|discard]]"
        ),
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

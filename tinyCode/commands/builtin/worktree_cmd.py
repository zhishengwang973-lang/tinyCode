"""Worktree management command."""

from pathlib import Path
from collections.abc import Callable

from tinyCode.commands.types import CommandMeta, CommandType
from tinyCode.worktree.manager import GitWorktreeManager
from tinyCode.worktree.initializer import WorktreeInitializer


def create(
    manager: GitWorktreeManager,
    workspace_changed: Callable[[], None] | None = None,
) -> CommandMeta:
    initializer = WorktreeInitializer(manager.repo_root)

    async def handler(args: list[str]) -> str:
        if not manager.is_available:
            return manager.availability_error
        sub = args[0].lower() if args else "status"
        if sub == "status":
            info = await manager.status()
            if info is None:
                return "当前不在工作目录中（位于主仓库）"
            return (
                f"当前工作目录: {info.name}\n"
                f"路径: {info.path}\n"
                f"分支: {info.branch}\n"
                f"HEAD: {info.head_commit[:12]}\n"
                f"有修改: {'是' if info.has_changes else '否'}"
            )

        elif sub == "list":
            worktrees = await manager.list_worktrees()
            if not worktrees:
                return "没有工作目录"
            lines = ["工作目录列表:"]
            for wt in worktrees:
                marker = "●" if wt.is_active else "○"
                dirty = " *" if wt.has_changes else ""
                lines.append(f"  {marker} {wt.name}{dirty}  ({wt.branch})")
            return "\n".join(lines)

        elif sub == "create":
            if len(args) < 2:
                return "用法: /worktree create <名称> [分支]"
            name = args[1]
            branch = args[2] if len(args) > 2 else ""
            info, err = await manager.create(name, branch)
            if info:
                init_logs = initializer.initialize(Path(info.path))
                details = (
                    "\n初始化:\n" + "\n".join(f"  {line}" for line in init_logs)
                    if init_logs else ""
                )
                return (
                    f"工作目录已创建: {info.name}\n路径: {info.path}\n"
                    f"分支: {info.branch}{details}"
                )
            return f"创建失败: {err}"

        elif sub == "enter":
            if len(args) < 2:
                return "用法: /worktree enter <名称>"
            ok, msg = await manager.enter(args[1])
            if ok:
                if workspace_changed:
                    workspace_changed()
                return f"已进入工作目录: {args[1]}"
            return f"进入失败: {msg}"

        elif sub == "exit":
            force = "--force" in args
            name = next((a for a in args[1:] if not a.startswith("-")), manager.active)
            if not name:
                return "用法: /worktree exit <名称> [--force]"
            ok, msg = await manager.exit(name, force=force)
            if ok and workspace_changed:
                workspace_changed()
            return msg if ok else f"退出失败: {msg}"

        return f"未知子命令: {args[0]}。可用: status, list, create, enter, exit"

    return CommandMeta(
        name="worktree",
        aliases=["wt"],
        description="管理 Git 工作目录（status/list/create/enter/exit）",
        usage="/worktree [status | list | create <name> [branch] | enter <name> | exit <name> [--force]]",
        cmd_type=CommandType.UI,
        handler=handler,
    )

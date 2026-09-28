"""Goal lifecycle command."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if not args:
            return ui.get_goal_status()
        subcommand = args[0].lower()
        if subcommand == "pause":
            return ui.pause_goal()
        if subcommand == "resume":
            additional = 0
            if len(args) > 1:
                try:
                    additional = int(args[1])
                except ValueError:
                    return "用法: /goal resume [新增回合数]"
                if additional < 0:
                    return "新增回合数不能小于 0"
            return ui.resume_goal(additional)
        if subcommand == "clear":
            return ui.clear_goal()
        if subcommand in {"status", "show"}:
            return ui.get_goal_status()
        return ui.start_goal(" ".join(args))

    return CommandMeta(
        name="goal",
        aliases=["g"],
        description="创建或管理当前会话的持久 Goal",
        usage="/goal <目标> | /goal [status | pause | resume [新增回合数] | clear]",
        cmd_type=CommandType.UI,
        handler=handler,
    )

"""Mode command — manage security policy only."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if not args:
            return (
                f"当前模式:\n"
                f"  安全等级: {ui.get_security_level()}\n"
                f"\n用法: /mode security <strict|normal|permissive>"
            )

        sub = args[0].lower()
        if sub == "plan":
            return "Plan-only 已废弃；请使用 /plan <目标> 创建计划，并用 /plan approve 批准后执行。"
        elif sub == "security":
            if len(args) != 2:
                return f"用法: /mode security <strict|normal|permissive>\n当前: {ui.get_security_level()}"
            if args[1].lower() not in {"strict", "normal", "permissive"}:
                return (
                    "安全等级必须是 strict、normal 或 permissive\n"
                    f"当前: {ui.get_security_level()}"
                )
            return f"安全等级: {ui.set_security_level(args[1])}"
        else:
            return f"未知子命令: {sub}。可用: security"

    return CommandMeta(
        name="mode",
        description="设置安全等级",
        usage="/mode [security <strict|normal|permissive>]",
        cmd_type=CommandType.UI,
        handler=handler,
    )

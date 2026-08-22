"""Mode command — toggle plan-only or set security level."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if not args:
            return (
                f"当前模式:\n"
                f"  Plan-only: {'ON' if ui.get_plan_only() else 'OFF'}\n"
                f"  安全等级: {ui.get_security_level()}\n"
                f"\n用法: /mode plan | /mode security <strict|normal|permissive>"
            )

        sub = args[0].lower()
        if sub == "plan":
            if len(args) != 1:
                return "用法: /mode plan"
            new_state = ui.toggle_plan_mode()
            return f"Plan-only 模式: {'ON' if new_state else 'OFF'}"
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
            return f"未知子命令: {sub}。可用: plan, security"

    return CommandMeta(
        name="mode",
        description="切换模式（plan-only / security level）",
        usage="/mode [plan | security <strict|normal|permissive>]",
        cmd_type=CommandType.UI,
        handler=handler,
    )

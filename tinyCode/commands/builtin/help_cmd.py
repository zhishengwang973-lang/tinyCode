"""Help command."""

from tinyCode.commands.registry import CommandRegistry
from tinyCode.commands.types import CommandMeta, CommandType


def create(registry: CommandRegistry) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if args:
            # Show detail for one command
            cmd = registry.lookup(args[0])
            if cmd is None:
                return f"未知命令: /{args[0]}"
            aliases = ", ".join(f"/{a}" for a in cmd.aliases) if cmd.aliases else "无"
            return (
                f"/{cmd.name} — {cmd.description}\n"
                f"用法: {cmd.usage}\n"
                f"别名: {aliases}\n"
                f"类型: {cmd.cmd_type.value}"
            )

        # List all
        lines = ["可用命令:"]
        for cmd in registry.list_visible():
            aliases = f"（别名: {', '.join(f'/{a}' for a in cmd.aliases)}）" if cmd.aliases else ""
            lines.append(f"  /{cmd.name:<12} {cmd.description:<40} {aliases}")
        lines.append("\n输入 /help <命令名> 查看详细用法")
        return "\n".join(lines)

    return CommandMeta(
        name="help",
        aliases=["h", "?"],
        description="显示帮助信息",
        usage="/help [命令名]",
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

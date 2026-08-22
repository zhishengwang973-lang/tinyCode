"""Graceful application exit command."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if args:
            return "用法: /exit"
        ui.request_exit()
        return "正在保存当前会话并安全退出…"

    return CommandMeta(
        name="exit",
        aliases=["quit", "q"],
        description="保存当前会话并安全退出 TinyCode",
        usage="/exit",
        cmd_type=CommandType.UI,
        handler=handler,
    )

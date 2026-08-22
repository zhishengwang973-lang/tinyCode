"""Clear command — clear conversation."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        ui.clear_conversation()
        return "对话已清空"

    return CommandMeta(
        name="clear",
        aliases=["cls", "reset"],
        description="清空当前对话",
        usage="/clear",
        cmd_type=CommandType.UI,
        handler=handler,
    )

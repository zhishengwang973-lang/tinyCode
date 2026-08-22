"""Compress command — manual context compression trigger."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        return await ui.trigger_compress()

    return CommandMeta(
        name="compress",
        aliases=["zip"],
        description="手动触发上下文压缩",
        usage="/compress",
        cmd_type=CommandType.UI,
        handler=handler,
    )

"""Cancel the active foreground agent task."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if args:
            return "用法: /cancel"
        if ui.cancel_active_turn():
            return "正在取消当前任务…"
        return "当前没有正在执行的任务"

    return CommandMeta(
        name="cancel",
        description="取消当前任务并保留已完成的进度",
        usage="/cancel",
        cmd_type=CommandType.UI,
        handler=handler,
    )

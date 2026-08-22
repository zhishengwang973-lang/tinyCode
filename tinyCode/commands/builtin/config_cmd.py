"""Runtime configuration command."""

from tinyCode.commands.types import CommandMeta, CommandType, ParamHint, UIControl
from tinyCode.config.constants import MAX_ALLOWED_ROUNDS

_USAGE = f"/config max-rounds [1-{MAX_ALLOWED_ROUNDS}]"
_RANGE_ERROR = f"最大轮次必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if not args:
            return (
                f"当前会话最大轮次: {ui.get_max_rounds()}\n"
                f"用法: {_USAGE}"
            )

        option = args[0].lower()
        if option not in {"max-rounds", "max_rounds"} or len(args) > 2:
            return f"用法: {_USAGE}"

        if len(args) == 1:
            return f"当前会话最大轮次: {ui.get_max_rounds()}"

        try:
            value = int(args[1])
        except ValueError:
            return _RANGE_ERROR

        if not 1 <= value <= MAX_ALLOWED_ROUNDS:
            return _RANGE_ERROR

        ui.set_max_rounds(value)
        return f"当前会话最大轮次已设为 {value}（重启后恢复配置文件值）"

    return CommandMeta(
        name="config",
        aliases=["cfg"],
        description="查看或修改当前会话的运行参数",
        usage=_USAGE,
        cmd_type=CommandType.UI,
        params=[ParamHint(
            "max-rounds",
            f"最大任务轮次（1-{MAX_ALLOWED_ROUNDS}）",
        )],
        handler=handler,
    )

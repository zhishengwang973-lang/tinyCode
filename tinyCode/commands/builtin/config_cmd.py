"""Runtime configuration command."""

from tinyCode.commands.types import CommandMeta, CommandType, ParamHint, UIControl
from tinyCode.config.constants import MAX_ALLOWED_ROUNDS, SUPPORTED_ROUND_LIMIT_ACTIONS

_USAGE = (
    "/config [max-rounds|round-extension|hard-max-rounds|"
    "round-limit-action] [值]"
)


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if not args:
            return (
                "当前会话轮次策略:\n"
                f"  初始预算: {ui.get_max_rounds()}\n"
                f"  续跑步长: {ui.get_round_extension()}\n"
                f"  硬上限: {ui.get_hard_max_rounds()}\n"
                f"  达到预算时: {ui.get_round_limit_action()}\n"
                f"用法: {_USAGE}"
            )

        option = args[0].lower()
        aliases = {
            "max-rounds": "max-rounds",
            "max_rounds": "max-rounds",
            "round-extension": "round-extension",
            "round_extension": "round-extension",
            "hard-max-rounds": "hard-max-rounds",
            "hard_max_rounds": "hard-max-rounds",
            "round-limit-action": "round-limit-action",
            "round_limit_action": "round-limit-action",
        }
        normalized = aliases.get(option)
        if normalized is None or len(args) > 2:
            return f"用法: {_USAGE}"

        if len(args) == 1:
            getters = {
                "max-rounds": ("当前会话初始轮次预算", ui.get_max_rounds),
                "round-extension": ("当前会话续跑步长", ui.get_round_extension),
                "hard-max-rounds": ("当前会话轮次硬上限", ui.get_hard_max_rounds),
                "round-limit-action": (
                    "当前会话预算耗尽策略", ui.get_round_limit_action,
                ),
            }
            label, getter = getters[normalized]
            return f"{label}: {getter()}"

        raw_value = args[1]
        if normalized == "round-limit-action":
            value = raw_value.lower()
            if value not in SUPPORTED_ROUND_LIMIT_ACTIONS:
                return "round_limit_action 必须是 ask、auto 或 stop"
            current = ui.set_round_limit_action(value)
            return f"当前会话预算耗尽策略已设为 {current}（重启后恢复配置文件值）"

        try:
            if normalized == "max-rounds" and raw_value.startswith("+"):
                value = ui.get_max_rounds() + int(raw_value[1:])
            else:
                value = int(raw_value)
        except ValueError:
            if normalized == "max-rounds":
                return f"最大轮次必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"
            return f"值必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"

        if not 1 <= value <= MAX_ALLOWED_ROUNDS:
            if normalized == "max-rounds":
                return f"最大轮次必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"
            return f"值必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"

        try:
            if normalized == "max-rounds":
                current = ui.set_max_rounds(value)
                label = "初始轮次预算"
            elif normalized == "round-extension":
                current = ui.set_round_extension(value)
                label = "续跑步长"
            else:
                current = ui.set_hard_max_rounds(value)
                label = "轮次硬上限"
        except ValueError as exc:
            return str(exc)
        return f"当前会话{label}已设为 {current}（重启后恢复配置文件值）"

    return CommandMeta(
        name="config",
        aliases=["cfg"],
        description="查看或修改当前会话的运行参数",
        usage=_USAGE,
        cmd_type=CommandType.UI,
        params=[
            ParamHint("max-rounds", "任务初始软预算"),
            ParamHint("round-extension", "每次续跑增加轮数"),
            ParamHint("hard-max-rounds", "任务轮次硬上限"),
            ParamHint("round-limit-action", "ask / auto / stop"),
        ],
        handler=handler,
    )

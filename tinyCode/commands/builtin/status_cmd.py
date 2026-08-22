"""Status command — comprehensive system status."""

import platform
from pathlib import Path

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        cwd = Path.cwd()
        runtime = ui.get_runtime_status()
        return (
            f"  工作目录: {cwd}\n"
            f"  操作系统: {platform.system()} {platform.release()}\n"
            f"  对话状态: {runtime['state']} (turn {runtime['turn_id']}, "
            f"last {runtime['last_outcome']})\n"
            f"  Plan-only: {'ON' if ui.get_plan_only() else 'OFF'}\n"
            f"  安全等级: {ui.get_security_level()}\n"
            f"  最大轮次: {ui.get_max_rounds()}\n"
            f"  估算 token: {ui.get_token_count():,}\n"
        )

    return CommandMeta(
        name="status",
        aliases=["st", "info"],
        description="显示综合状态",
        usage="/status",
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

"""Inspect the effective system prompt without calling the model."""

from tinyCode.commands.types import CommandMeta, CommandType, ParamHint, UIControl

_SECTIONS = {"all", "base", "instructions", "skills", "environment", "notes", "injection"}
_USAGE = "/prompt [all|base|instructions|skills|environment|notes|injection]"


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if len(args) > 1:
            return f"用法: {_USAGE}"
        section = args[0].lower() if args else "all"
        if section not in _SECTIONS:
            return f"未知部分: {section}\n用法: {_USAGE}"
        return ui.get_system_prompt(section)

    return CommandMeta(
        name="prompt",
        aliases=["system-prompt", "sp"],
        description="查看当前实际生效的系统提示词",
        usage=_USAGE,
        cmd_type=CommandType.LOCAL,
        params=[ParamHint("section", "要查看的提示词部分")],
        handler=handler,
    )

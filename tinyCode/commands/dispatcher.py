"""Command dispatcher — routes parsed commands to handlers by type."""

from tinyCode.commands.parser import ParsedCommand, is_command, parse
from tinyCode.commands.registry import CommandRegistry
from tinyCode.commands.types import CommandType, UIControl


class CommandDispatcher:
    """Routes user input: commands (``/`` prefix) to handlers, else to AI."""

    def __init__(self, registry: CommandRegistry, ui: UIControl) -> None:
        self._registry = registry
        self._ui = ui

    # -- public API -----------------------------------------------------------

    def is_command(self, text: str) -> bool:
        return is_command(text)

    async def dispatch(self, text: str) -> tuple[bool, str | None]:
        """Route *text*.

        Returns ``(was_command, result_or_None)``.
        - If *text* is a command: returns ``(True, result_string)``.
        - If *text* is not a command: returns ``(False, None)`` — caller
          should send to the AI.
        """
        if not is_command(text):
            return False, None

        try:
            parsed = parse(text)
        except ValueError:
            return True, "命令解析失败"

        cmd = self._registry.lookup(parsed.command_name)
        if cmd is None:
            return True, f"未知命令: /{parsed.command_name}\n输入 /help 查看可用命令列表。"

        if cmd.handler is None:
            return True, f"命令 /{cmd.name} 尚未实现"

        try:
            if cmd.cmd_type == CommandType.UI:
                result = await cmd.handler(parsed.args)
                return True, result

            elif cmd.cmd_type == CommandType.LOCAL:
                result = await cmd.handler(parsed.args)
                return True, result

            elif cmd.cmd_type == CommandType.PROMPT_INJECT:
                # Handler returns a prompt; inject into conversation
                prompt = await cmd.handler(parsed.args)
                if prompt:
                    self._ui.send_to_conversation(prompt)
                return True, None  # prompt injected silently
        except Exception as exc:
            return True, f"命令执行失败: {exc}"

        return True, "未知命令类型"

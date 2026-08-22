"""Command registry — register, lookup, detect alias conflicts."""

from tinyCode.commands.types import CommandMeta, CommandType


class CommandRegistry:
    """Central registry for all slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, CommandMeta] = {}
        self._aliases: dict[str, str] = {}  # alias → canonical name

    def register(self, cmd: CommandMeta) -> None:
        """Register a command. Raises ValueError on name or alias conflict."""
        if not isinstance(cmd.name, str) or not cmd.name:
            raise ValueError("命令名必须是非空字符串")
        if not isinstance(cmd.description, str):
            raise ValueError(f"命令描述必须是字符串: /{cmd.name}")
        if not isinstance(cmd.usage, str):
            raise ValueError(f"命令用法必须是字符串: /{cmd.name}")
        if not isinstance(cmd.cmd_type, CommandType):
            raise ValueError(f"命令类型无效: /{cmd.name}")
        if cmd.handler is not None and not callable(cmd.handler):
            raise ValueError(f"命令 handler 必须可调用: /{cmd.name}")
        if not isinstance(cmd.aliases, list):
            raise ValueError(f"命令别名必须是字符串列表: /{cmd.name}")
        for alias in cmd.aliases:
            if not isinstance(alias, str) or not alias:
                raise ValueError(f"命令别名必须是非空字符串: /{cmd.name}")

        name = cmd.name.lower()
        if name in self._commands:
            raise ValueError(f"命令名冲突: /{name}")
        if name in self._aliases:
            raise ValueError(f"命令名与别名冲突: /{name}（已被 /{self._aliases[name]} 使用）")
        self._commands[name] = cmd

        for alias in cmd.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._aliases:
                raise ValueError(f"别名冲突: /{alias_lower}（已被 /{self._aliases[alias_lower]} 使用）")
            if alias_lower in self._commands:
                raise ValueError(f"别名与命令名冲突: /{alias_lower}")
            self._aliases[alias_lower] = name

    def lookup(self, name: str) -> CommandMeta | None:
        """Find a command by name or alias (case-insensitive)."""
        n = name.lower()
        if n in self._commands:
            return self._commands[n]
        if n in self._aliases:
            return self._commands[self._aliases[n]]
        return None

    def list_visible(self) -> list[CommandMeta]:
        """Return all non-hidden commands, alphabetically sorted."""
        return sorted(
            [c for c in self._commands.values() if not c.hidden],
            key=lambda c: c.name,
        )

    def list_all(self) -> list[CommandMeta]:
        """Return all commands including hidden ones."""
        return sorted(self._commands.values(), key=lambda c: c.name)

    def get_completions(self, prefix: str) -> list[str]:
        """Return command names starting with *prefix* (case-insensitive)."""
        p = prefix.lower()
        results: list[str] = []
        for cmd in self._commands.values():
            if cmd.hidden:
                continue
            if cmd.name.startswith(p):
                results.append(f"/{cmd.name}")
            for alias in cmd.aliases:
                if alias.startswith(p):
                    results.append(f"/{alias}")
        return sorted(results, key=lambda x: x.strip("/"))

"""Command parser — parse /name args from raw input text."""

from dataclasses import dataclass


@dataclass
class ParsedCommand:
    command_name: str
    args: list[str]


def is_command(text: str) -> bool:
    """Check if *text* starts with a ``/`` command prefix."""
    return text.strip().startswith("/")


def parse(text: str) -> ParsedCommand:
    """Parse a ``/command arg1 arg2 ...`` input.

    The command name is case-insensitive.  Everything after the first space
    is treated as arguments (split by whitespace).
    """
    text = text.strip()
    if not text.startswith("/"):
        raise ValueError(f"不是命令: {text}")

    # Remove leading / and split
    inner = text[1:].strip()
    parts = inner.split(maxsplit=1)
    if not parts:
        raise ValueError("命令名不能为空")
    name = parts[0].lower()
    args: list[str] = []
    if len(parts) > 1:
        # Split remaining by whitespace — but preserve quoted strings
        args = _smart_split(parts[1])
    return ParsedCommand(command_name=name, args=args)


def _smart_split(text: str) -> list[str]:
    """Split text by whitespace, preserving quoted strings."""
    import shlex
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"参数解析失败: {exc}") from exc

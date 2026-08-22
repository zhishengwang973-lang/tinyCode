"""Conservative classification for shell commands with no side effects."""

from __future__ import annotations

import shlex
from pathlib import Path


_READ_ONLY_COMMANDS = {
    "[",
    "basename",
    "cat",
    "cut",
    "date",
    "df",
    "dirname",
    "du",
    "echo",
    "env",
    "file",
    "find",
    "grep",
    "head",
    "id",
    "less",
    "ls",
    "more",
    "printenv",
    "printf",
    "ps",
    "pwd",
    "readlink",
    "realpath",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "test",
    "tree",
    "tr",
    "uname",
    "uniq",
    "wc",
    "whereis",
    "which",
    "whoami",
}

_READ_ONLY_GIT_SUBCOMMANDS = {
    "blame",
    "describe",
    "diff",
    "grep",
    "log",
    "ls-files",
    "ls-tree",
    "name-rev",
    "rev-parse",
    "shortlog",
    "show",
    "status",
}

_COMMAND_SEPARATORS = {"|", "&&", "||", ";"}
_UNSAFE_SHELL_TOKENS = {
    "&", "<", ">", "<<", ">>", "(", ")",
    ">&", "<&", "&>", "&>>", "|&",
}
_OUTPUT_REDIRECT_TOKENS = {">", ">>"}
_UNSAFE_FIND_OPTIONS = {
    "-delete",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprintf",
    "-ok",
    "-okdir",
}


def is_read_only_command(command: str) -> bool:
    """Return True only when every shell segment is known to be read-only."""
    if not isinstance(command, str) or not command.strip():
        return False
    if "`" in command or "$(" in command or "\n" in command or "\r" in command:
        return False

    try:
        lexer = shlex.shlex(
            command,
            posix=True,
            punctuation_chars="|&;<>()",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False

    if not tokens:
        return False

    stripped_tokens = _strip_safe_dev_null_redirects(tokens)
    if stripped_tokens is None or any(
        token in _UNSAFE_SHELL_TOKENS for token in stripped_tokens
    ):
        return False

    segment: list[str] = []
    for token in stripped_tokens + [";"]:
        if token not in _COMMAND_SEPARATORS:
            segment.append(token)
            continue
        if not _is_read_only_segment(segment):
            return False
        segment = []
    return not segment


def _strip_safe_dev_null_redirects(tokens: list[str]) -> list[str] | None:
    """Remove output redirects that can only discard data.

    ``shlex`` tokenizes ``2>/dev/null`` as ``["2", ">", "/dev/null"]``.
    Treating every output redirect as mutating made ordinary inspection
    commands unexpectedly enter HITL and appear to hang.  Only the exact
    null-device target is safe; every other redirect remains ambiguous.
    """
    cleaned: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _OUTPUT_REDIRECT_TOKENS:
            cleaned.append(token)
            index += 1
            continue

        if index + 1 >= len(tokens) or tokens[index + 1] != "/dev/null":
            return None
        if cleaned and cleaned[-1] in {"1", "2"}:
            cleaned.pop()
        index += 2
    return cleaned


def _is_read_only_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    command = Path(tokens[0]).name
    args = tokens[1:]

    if command == "git":
        return bool(args) and args[0] in _READ_ONLY_GIT_SUBCOMMANDS
    if command not in _READ_ONLY_COMMANDS:
        return False
    if command == "find" and any(arg in _UNSAFE_FIND_OPTIONS for arg in args):
        return False
    if command == "sed" and any(
        arg == "-i" or arg.startswith("-i") and len(arg) > 2
        for arg in args
    ):
        return False
    return True

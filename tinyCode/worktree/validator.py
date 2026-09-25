"""Worktree name validator — strict character set, length, traversal prevention."""

import re
from pathlib import Path
from urllib.parse import quote, unquote

#: Allowed characters per segment
_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

MAX_SEGMENT_LEN = 64
MAX_TOTAL_LEN = 255


def validate_name(name: str) -> tuple[bool, str]:
    """Validate a worktree name.

    Returns ``(is_valid, error_message)``.
    """
    if not name:
        return False, "名称不能为空"
    if len(name) > MAX_TOTAL_LEN:
        return False, f"名称过长（最大 {MAX_TOTAL_LEN} 字符）"

    segments = name.split("/")
    if not segments:
        return False, "无效名称"

    for seg in segments:
        if seg in (".", "..", ""):
            return False, f"名称包含非法路径段: '{seg}'"
        if len(seg) > MAX_SEGMENT_LEN:
            return False, f"名称段过长: '{seg}'（最大 {MAX_SEGMENT_LEN} 字符）"
        if not _SEGMENT_RE.match(seg):
            return False, f"名称包含非法字符: '{seg}'（仅允许 a-z A-Z 0-9 _ -）"

    return True, ""


def name_to_branch(name: str) -> str:
    """Convert a worktree name to a git branch name.

    Preserve validated path segments so distinct names remain distinct.
    """
    return "tinyCode/" + name


def name_to_dirname(name: str) -> str:
    """Convert a worktree name to one collision-free subdirectory name."""
    return quote(name, safe="-_")


def dirname_to_name(dirname: str) -> str:
    """Decode a directory name produced by :func:`name_to_dirname`."""
    return unquote(dirname)


def legacy_name_to_dirname(name: str) -> str:
    """Return the pre-encoding directory name for recovery compatibility."""
    return name.replace("/", "-")


def resolve_worktree_path(repo_root: Path, name: str) -> Path:
    """Resolve the collision-free managed path for a validated name."""
    return repo_root / ".tinyCode" / "worktrees" / name_to_dirname(name)

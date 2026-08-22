"""Helpers for reading persisted oversized tool results."""

from pathlib import Path

from tinyCode.conversation.truncator import DEFAULT_STORAGE_DIR
from tinyCode.tools.validation import require_string


def resolve_tool_result_path(file_path: str, storage_dir: Path | None = None) -> Path:
    """Resolve a stored tool-result path without allowing storage escape."""
    file_path = require_string(file_path, "file_path")
    root = (storage_dir or DEFAULT_STORAGE_DIR).resolve()
    path = Path(file_path)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        raise ValueError(f"无效工具结果路径: {file_path}")
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"只能读取 tool_results 存储目录内的文件: {file_path}")
    return resolved


def require_positive_int(value: object, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是正整数")
    if value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value

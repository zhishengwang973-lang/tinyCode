"""Task-local workspace root used by tools running concurrently."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


_WORKSPACE_ROOT: ContextVar[Path | None] = ContextVar(
    "tinycode_workspace_root",
    default=None,
)


def get_workspace_root() -> Path:
    root = _WORKSPACE_ROOT.get()
    return (root or Path.cwd()).resolve()


@contextmanager
def use_workspace(root: Path) -> Iterator[Path]:
    """Set the workspace for the current async context only."""
    resolved = root.resolve()
    token = _WORKSPACE_ROOT.set(resolved)
    try:
        yield resolved
    finally:
        _WORKSPACE_ROOT.reset(token)

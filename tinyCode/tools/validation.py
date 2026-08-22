"""Shared validation helpers for model-facing tool parameters."""

from typing import Any


def require_string(value: Any, field_name: str) -> str:
    """Return *value* if it is a string, otherwise raise a readable error."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串")
    return value

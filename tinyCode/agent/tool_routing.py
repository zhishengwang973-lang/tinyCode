"""Compatibility exports for task-mode routing."""

from tinyCode.agent.task_mode import (
    TaskMode,
    classify_task_mode,
    should_enable_tools,
    task_mode_instruction,
)

__all__ = [
    "TaskMode",
    "classify_task_mode",
    "should_enable_tools",
    "task_mode_instruction",
]

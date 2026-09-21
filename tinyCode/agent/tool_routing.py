"""Compatibility exports for task-mode routing."""

from tinyCode.agent.task_mode import (
    RuleTaskModeDecision,
    TaskMode,
    classify_task_mode,
    classify_task_mode_rule,
    should_enable_tools,
    task_mode_instruction,
)

__all__ = [
    "RuleTaskModeDecision",
    "TaskMode",
    "classify_task_mode",
    "classify_task_mode_rule",
    "should_enable_tools",
    "task_mode_instruction",
]

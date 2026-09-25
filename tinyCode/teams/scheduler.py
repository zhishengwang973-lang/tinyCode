"""Dispatch mode — pure scheduling with double-lock enforcement."""

from tinyCode.tools.base import ToolCategory

# Tools stripped from Lead in dispatch mode
_DISPATCH_DENY_TOOLS = {
    "read_file", "write_file", "edit_file", "delete_file", "run_command",
    "grep", "glob", "list_files",
}

# Workflow stages injected in dispatch mode
_DISPATCH_WORKFLOW = """\
[纯调度模式] 你是 Team 规划器。当前阶段只生成结构化任务计划，不能调用工具。
计划会由运行时负责创建任务、并发调度、等待成员并合并结果。

工作流程：
1. **理解需求**: 分析用户目标，明确范围和验收条件
2. **模块拆分**: 将目标拆分为可独立执行的任务单元（每个 5-15 分钟）
3. **依赖分析**: 标记任务间的先后依赖关系
4. **人员匹配**: 为每个任务选择最合适的成员角色
5. **任务委派**: 为每项任务明确指定唯一成员
6. **进度监控**: 由运行时根据依赖图调度，不要在计划中虚构工具调用
7. **增量收集**: 成员完成后由运行时收集并合并
8. **质量验证**: 对关键结果做一致性检查
9. **冲突仲裁**: 合并时遇到冲突，调用 LLM 裁决
10. **综合报告**: 汇总所有结果，生成用户可读的最终报告

不要自己读文件、写代码、执行命令或声称已经调用工具——这些是成员和运行时的工作。"""


class DispatchScheduler:
    """Enforces dispatch mode with double-lock checking."""

    def __init__(self) -> None:
        self._lock_1 = False  # TUI toggle
        self._lock_2 = False  # config/cli flag

    @property
    def is_active(self) -> bool:
        return self._lock_1 and self._lock_2

    def set_lock_1(self, enabled: bool) -> None:
        self._lock_1 = enabled

    def set_lock_2(self, enabled: bool) -> None:
        self._lock_2 = enabled

    def filter_tools(self, tools: list) -> list:
        """Remove code-read/write/exec tools if dispatch is active."""
        if not self.is_active:
            return tools
        return [
            t for t in tools
            if _tool_name(t) not in _DISPATCH_DENY_TOOLS
        ]

    def get_workflow_instructions(self) -> str:
        if not self.is_active:
            return ""
        return _DISPATCH_WORKFLOW


def _tool_name(tool_def: dict) -> str:
    if isinstance(tool_def.get("function"), dict):
        return tool_def["function"].get("name", "")
    return tool_def.get("name", "")

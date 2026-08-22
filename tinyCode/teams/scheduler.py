"""Dispatch mode — pure scheduling with double-lock enforcement."""

from tinyCode.tools.base import ToolCategory

# Tools stripped from Lead in dispatch mode
_DISPATCH_DENY_TOOLS = {
    "read_file", "write_file", "edit_file", "delete_file", "run_command",
    "grep", "glob", "list_files",
}

# Workflow stages injected in dispatch mode
_DISPATCH_WORKFLOW = """\
[纯调度模式] 你是指挥官（Lead）。你只能：
- 使用 sub_agent 或 team 工具分配工作
- 使用 team_send_message 与成员通信
- 使用 team_create_task / team_list_tasks 管理任务
- 终止成员、查看结果、综合报告

工作流程：
1. **理解需求**: 分析用户目标，明确范围和验收条件
2. **模块拆分**: 将目标拆分为可独立执行的任务单元（每个 5-15 分钟）
3. **依赖分析**: 标记任务间的先后依赖关系
4. **人员匹配**: 为每个任务选择最合适的成员角色
5. **任务委派**: 创建任务并分配给成员（通过 team_create_task + team_send_message）
6. **进度监控**: 定期检查任务状态（team_list_tasks），处理阻塞
7. **增量收集**: 成员完成后立即查看结果，增量合并
8. **质量验证**: 对关键结果做一致性检查
9. **冲突仲裁**: 合并时遇到冲突，调用 LLM 裁决
10. **综合报告**: 汇总所有结果，生成用户可读的最终报告

不要自己读文件、写代码或执行命令——这些是成员的工作。"""


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

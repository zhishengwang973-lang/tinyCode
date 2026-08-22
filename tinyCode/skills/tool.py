"""Skill loader tool — the built-in tool that activates skills on demand."""

from tinyCode.skills.registry import SkillRegistry
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult


class SkillTool(BaseTool):
    """System-level tool for loading and activating skills.

    This tool is always available regardless of skill-level tool whitelists,
    enabling skills to be nested (one skill can trigger activation of another).
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "skill_loader"

    @property
    def description(self) -> str:
        available = ", ".join(s.name for s in self._registry.list_available()[:20])
        return (
            "激活一个 Skill 以获取专业的 SOP 指令。"
            f"可用 Skills: {available}"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("name", "string", "要激活的 Skill 名称"),
        ]

    async def execute(self, name: str) -> ToolResult:
        skill = self._registry.activate(name)
        if skill is None:
            available = ", ".join(s.name for s in self._registry.list_available())
            return ToolResult(
                success=False, content="",
                error=f"Skill '{name}' 不存在。可用: {available}",
            )
        return ToolResult(
            success=True,
            content=(
                f"Skill '{name}' 已激活。\n\n"
                f"模式: {skill.meta.mode.value}\n"
                f"工具白名单: {skill.meta.tools or '全部'}\n\n"
                f"--- SOP 指令 ---\n{skill.body}"
            ),
        )

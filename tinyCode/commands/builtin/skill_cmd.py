"""Skill management command."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl
from tinyCode.skills.registry import SkillRegistry


def create(registry: SkillRegistry, ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        sub = args[0].lower() if args else "list"
        if sub == "list":
            skills = registry.list_available()
            if not skills:
                return "没有可用的 Skill"
            lines = ["可用 Skills:"]
            for s in skills:
                activated_names = {item.meta.name for item in registry.activated}
                activated = "●" if s.name in activated_names else "○"
                lines.append(f"  {activated} /{s.name} — {s.description}")
            lines.append("\n使用 /skill <名字> 查看详情，skill_loader 工具激活")
            return "\n".join(lines)

        elif sub == "reload":
            registry.load_all()
            return f"已重新扫描，共 {len(registry.list_available())} 个 Skills"

        elif sub == "clear":
            registry.clear_activated()
            return "已清空所有激活的 Skill"

        else:
            # Show detail for one skill
            meta = registry.get_meta(args[0])
            if meta is None:
                return f"Skill '{args[0]}' 不存在。输入 /skill list 查看可用列表"
            skill = registry.get_skill(args[0])
            body_preview = skill.body[:500] if skill else "(未加载)"
            return (
                f"/{meta.name} — {meta.description}\n"
                f"模式: {meta.mode.value}\n"
                f"工具: {meta.tools or '全部'}\n"
                f"来源: {meta.source}\n\n"
                f"--- SOP 指令（前 500 字符）---\n{body_preview}"
            )

    return CommandMeta(
        name="skill",
        aliases=["skills"],
        description="管理 Skills（list / reload / clear / <名字>）",
        usage="/skill [list | reload | clear | <名字>]",
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

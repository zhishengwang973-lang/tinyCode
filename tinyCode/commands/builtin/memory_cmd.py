"""Memory command — view, clear, edit notes."""

from pathlib import Path

from tinyCode.commands.types import CommandMeta, CommandType
from tinyCode.notes.manager import AutoNoteManager


def create(note_manager: AutoNoteManager) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        sub = args[0].lower() if args else "show"
        if sub == "show":
            category = args[1] if len(args) > 1 else None
            if category:
                content = note_manager.read_note(category)
                return f"--- {category} ---\n{content or '(空)'}"
            # Show all categories
            lines = ["笔记分类:"]
            for cat in ["用户偏好", "纠正反馈", "项目知识", "参考资料"]:
                note = note_manager.read_note(cat)
                size = len(note) if note else 0
                lines.append(f"  {cat}: {size} 字符")
            return "\n".join(lines)

        elif sub == "clear":
            if len(args) < 2:
                return "用法: /memory clear <分类名>"
            return note_manager.clear_note(args[1])

        elif sub == "edit":
            if len(args) < 2:
                return "用法: /memory edit <分类名>"
            fp = note_manager.get_note_path(args[1])
            if fp:
                return f"笔记文件路径: {fp}\n用编辑器打开即可修改。"
            return f"未知分类: {args[1]}"

        return f"未知子命令: {args[0]}。可用: show, clear, edit"

    return CommandMeta(
        name="memory",
        aliases=["mem", "notes"],
        description="管理笔记（show / clear / edit）",
        usage="/memory [show [分类] | clear <分类> | edit <分类>]",
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

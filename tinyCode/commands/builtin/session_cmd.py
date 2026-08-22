"""Session command — list, load, new, delete."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    def _display_text(value, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value or default
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return default

    def _display_count(value) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    async def handler(args: list[str]) -> str:
        sub = args[0].lower() if args else "list"
        if sub == "list":
            sessions = ui.get_session_list()
            if not sessions:
                return "没有保存的会话"
            lines = ["会话列表:"]
            for s in sessions[:20]:
                sid = _display_text(s.get("id"), "?")[:12]
                title = _display_text(s.get("title"), "无标题")[:50]
                count = _display_count(s.get("message_count"))
                last = _display_text(s.get("last_active_at"), "")[:16]
                lines.append(f"  {sid}  {title}  ({count} 条消息, {last})")
            lines.append("\n/session load <ID> 加载 | /session delete <ID> 删除")
            return "\n".join(lines)

        elif sub == "load":
            if len(args) < 2:
                return "用法: /session load <会话ID>"
            return ui.load_session(args[1])

        elif sub == "delete":
            if len(args) < 2:
                return "用法: /session delete <会话ID>"
            return ui.delete_session(args[1])

        elif sub == "new":
            return ui.new_session()

        return f"未知子命令: {args[0]}。可用: list, load, delete, new"

    return CommandMeta(
        name="session",
        aliases=["sess"],
        description="管理会话（list / load / delete / new）",
        usage="/session [list | load <id> | delete <id> | new]",
        cmd_type=CommandType.UI,
        handler=handler,
    )

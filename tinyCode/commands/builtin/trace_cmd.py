"""Execution-trace inspection and runtime controls."""

from collections.abc import Awaitable, Callable

from tinyCode.commands.types import CommandMeta, CommandType
from tinyCode.tracing.recorder import TraceRecorder


Confirmer = Callable[[str], Awaitable[bool]]
_USAGE = "/trace [status|last|open|path|on|off|clear]"


def create(
    recorder: TraceRecorder,
    confirmer: Confirmer | None = None,
) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if len(args) > 1:
            return f"用法: {_USAGE}"
        action = args[0].lower() if args else "status"
        if action == "status":
            return recorder.status_text()
        if action == "last":
            return recorder.render_last_text()
        if action == "path":
            path = recorder.latest_path()
            return f"最近 Trace: {path}" if path else "暂无执行 Trace"
        if action == "open":
            path = recorder.open_last()
            if path is None:
                detail = f"：{recorder.last_error}" if recorder.last_error else ""
                return f"暂无可打开的执行 Trace{detail}"
            if recorder.last_error:
                return (
                    f"已生成 Trace，但未能自动打开: {path}\n"
                    f"原因: {recorder.last_error}"
                )
            return f"已生成并打开 Trace: {path}"
        if action == "on":
            recorder.set_enabled(True)
            return "执行 Trace 已开启（仅当前进程；重启后恢复配置文件值）"
        if action == "off":
            recorder.set_enabled(False)
            return "执行 Trace 已关闭（仅当前进程；重启后恢复配置文件值）"
        if action == "clear":
            if confirmer is not None and not await confirmer(
                f"将永久删除 {recorder.storage_dir} 中的全部 Trace，是否继续？"
            ):
                return "已取消清理 Trace"
            removed = recorder.clear()
            return f"已删除 {removed} 个 Trace 文件"
        return f"未知子命令: {action}。用法: {_USAGE}"

    return CommandMeta(
        name="trace",
        aliases=["tr"],
        description="查看和管理本地执行 Trace",
        usage=_USAGE,
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

"""Tasks command — manage background sub-agent tasks."""

from tinyCode.commands.types import CommandMeta, CommandType
from tinyCode.subagent.manager import BackgroundTaskManager
from tinyCode.time_utils import format_beijing_time


def create(task_manager: BackgroundTaskManager) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        sub = args[0].lower() if args else "list"
        if sub == "list":
            return task_manager.get_status_summary()

        elif sub == "detail":
            if len(args) < 2:
                return "用法: /tasks detail <任务ID>"
            task = task_manager.get(args[1])
            if task is None:
                return f"任务 {args[1]} 不存在"
            return (
                f"任务: {task.id}\n"
                f"角色: {task.role or 'fork'}\n"
                f"状态: {task.status.value}\n"
                f"任务: {task.task}\n"
                f"轮次: {task.round_count}\n"
                f"开始: {format_beijing_time(task.started_at)}\n"
                f"结束: {format_beijing_time(task.finished_at)}\n"
                f"后台: {'是' if task.background else '否'}\n\n"
                f"结果:\n{task.result[:2000] if task.result else '(无)'}"
            )

        elif sub == "kill":
            if len(args) < 2:
                return "用法: /tasks kill <任务ID>"
            if task_manager.cancel(args[1]):
                return f"任务 {args[1]} 已取消"
            return f"任务 {args[1]} 不存在或已完成"

        return f"未知子命令: {args[0]}。可用: list, detail, kill"

    return CommandMeta(
        name="tasks",
        aliases=["bg"],
        description="管理后台任务（list / detail / kill）",
        usage="/tasks [list | detail <id> | kill <id>]",
        cmd_type=CommandType.LOCAL,
        handler=handler,
    )

"""Explicit user-approved planning workflow."""

from tinyCode.commands.types import CommandMeta, CommandType, UIControl


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if not args or args[0].lower() in {"show", "status"}:
            return ui.show_plan_draft()

        subcommand = args[0].lower()
        if subcommand in {"approve", "apply"}:
            if len(args) != 1:
                return "用法: /plan approve"
            return ui.approve_plan_draft()
        if subcommand in {"discard", "clear"}:
            if len(args) != 1:
                return "用法: /plan discard"
            return ui.discard_plan_draft()
        if subcommand in {"revise", "edit"}:
            request = " ".join(args[1:]).strip()
            if not request:
                return "用法: /plan revise <修订要求>"
            return await ui.revise_plan_draft(request)
        if subcommand in {"new", "create"}:
            objective = " ".join(args[1:]).strip()
        else:
            # The short form is deliberately the common path:
            # /plan 为 API 增加分页并补测试
            objective = " ".join(args).strip()
        if not objective:
            return "用法: /plan <任务目标>"
        return await ui.create_plan_draft(objective)

    return CommandMeta(
        name="plan",
        aliases=["p"],
        description="创建、修订和批准用户任务计划",
        usage=(
            "/plan <目标> | /plan [show|revise <要求>|approve|discard]"
        ),
        cmd_type=CommandType.UI,
        handler=handler,
    )

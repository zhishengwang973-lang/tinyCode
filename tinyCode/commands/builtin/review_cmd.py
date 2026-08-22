"""Review command — inject code review prompt into conversation."""

from tinyCode.commands.types import CommandMeta, CommandType

_REVIEW_PROMPT = (
    "请审查当前工作目录中的未提交更改。检查以下方面：\n"
    "1. 逻辑错误和边界条件\n"
    "2. 代码风格和可读性\n"
    "3. 安全漏洞\n"
    "4. 性能问题\n"
    "5. 可复用性和简化机会\n\n"
    "先读取相关文件，再给出审查意见。"
)


def create() -> CommandMeta:
    async def handler(args: list[str]) -> str:
        if args:
            return f"请审查以下文件/目录: {' '.join(args)}"
        return _REVIEW_PROMPT

    return CommandMeta(
        name="review",
        aliases=["cr", "audit"],
        description="请求代码审查（将提示词注入对话）",
        usage="/review [文件或目录]",
        cmd_type=CommandType.PROMPT_INJECT,
        handler=handler,
    )

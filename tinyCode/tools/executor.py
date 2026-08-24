"""Tool executor — runs a tool with timeout and error handling."""

import asyncio
from tinyCode.tools.base import BaseTool, ToolResult


class ToolExecutor:
    """Executes a tool with timeout and structured error handling."""

    def __init__(self, default_timeout: float = 30.0) -> None:
        self._default_timeout = default_timeout

    async def execute(
        self,
        tool: BaseTool | None,
        params: dict,
        timeout: float | None = None,
    ) -> ToolResult:
        """Run *tool* with *params*, enforcing a timeout.

        On timeout or unexpected exception, returns a structured failure
        ``ToolResult`` so the model can adjust rather than crashing.
        """
        if tool is None:
            return ToolResult(success=False, content="", error="未知工具")

        if not isinstance(params, dict):
            return ToolResult(success=False, content="", error="工具参数必须是对象")

        effective_timeout = (
            timeout
            if timeout is not None
            else None if tool.timeout_exempt
            else self._default_timeout
        )
        try:
            execution = tool.execute(**params)
            result = (
                await execution
                if effective_timeout is None
                else await asyncio.wait_for(execution, timeout=effective_timeout)
            )
            if not isinstance(result, ToolResult):
                return ToolResult(
                    success=False,
                    content="",
                    error=f"工具返回值必须是 ToolResult: {tool.name}",
                )
            if (
                not isinstance(result.success, bool)
                or not isinstance(result.content, str)
                or not isinstance(result.error, str)
            ):
                return ToolResult(
                    success=False,
                    content="",
                    error=f"工具返回的 ToolResult 字段类型无效: {tool.name}",
                )
            if not result.success and not result.error.strip():
                return ToolResult(
                    success=False,
                    content=result.content,
                    error=f"工具 '{tool.name}' 执行失败，但未提供错误详情",
                )
            return result
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content="",
                error=f"工具 '{tool.name}' 执行超时（{effective_timeout}s）",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content="",
                error=f"工具 '{tool.name}' 执行异常: {type(exc).__name__}: {exc}",
            )

"""Read a bounded line window from persisted oversized tool-result files."""

from pathlib import Path

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.tool_result_store import require_positive_int, resolve_tool_result_path

DEFAULT_LIMIT = 80
MAX_LIMIT = 200


class ToolResultReadTool(BaseTool):
    """Read a small slice of a stored oversized tool result."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = storage_dir

    @property
    def name(self) -> str:
        return "tool_result_read"

    @property
    def description(self) -> str:
        return "按行号读取已落盘的大型工具结果文件片段，避免把完整大文件放回上下文。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("file_path", "string", "工具结果文件路径，必须位于 tool_results 存储目录内。"),
            ToolParameter("start_line", "integer", "起始行号，从 1 开始。", required=False, default=1),
            ToolParameter("limit", "integer", "最多读取的行数，最大 200。", required=False, default=DEFAULT_LIMIT),
        ]

    async def execute(
        self, file_path: str, start_line: int = 1, limit: int = DEFAULT_LIMIT,
    ) -> ToolResult:
        try:
            resolved = resolve_tool_result_path(file_path, self._storage_dir)
            start_line = require_positive_int(start_line, "start_line", 1)
            limit = min(require_positive_int(limit, "limit", DEFAULT_LIMIT), MAX_LIMIT)
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        if not resolved.exists():
            return ToolResult(success=False, content="", error=f"工具结果文件不存在: {file_path}")
        if not resolved.is_file():
            return ToolResult(success=False, content="", error=f"路径不是文件: {file_path}")

        end_line = start_line + limit - 1
        lines: list[str] = []
        try:
            with resolved.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if line_no < start_line:
                        continue
                    if line_no > end_line:
                        break
                    lines.append(f"{resolved.name}:{line_no}: {line.rstrip()}")
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"读取工具结果失败: {exc}")

        if not lines:
            return ToolResult(success=True, content="(请求范围内无内容)")

        return ToolResult(
            success=True,
            content=f"读取 {len(lines)} 行（{start_line}-{start_line + len(lines) - 1}）\n"
            + "\n".join(lines),
        )

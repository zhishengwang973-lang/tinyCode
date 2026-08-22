"""Search persisted oversized tool-result files."""

import re
from pathlib import Path

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.tool_result_store import resolve_tool_result_path
from tinyCode.tools.validation import require_string

MAX_RESULTS = 50
SNIPPET_LENGTH = 300


class ToolResultSearchTool(BaseTool):
    """Search a stored oversized tool result without reading it all."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = storage_dir

    @property
    def storage_dir(self) -> Path | None:
        return self._storage_dir

    def set_storage_dir(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir.resolve()

    @property
    def name(self) -> str:
        return "tool_result_search"

    @property
    def description(self) -> str:
        return "在已落盘的大型工具结果文件中搜索正则表达式，返回少量命中行。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("file_path", "string", "工具结果文件路径，必须位于 tool_results 存储目录内。"),
            ToolParameter("pattern", "string", "要搜索的正则表达式。"),
        ]

    async def execute(self, file_path: str, pattern: str) -> ToolResult:
        try:
            resolved = resolve_tool_result_path(file_path, self._storage_dir)
            pattern = require_string(pattern, "pattern")
            regex = re.compile(pattern)
        except (ValueError, re.error) as exc:
            return ToolResult(success=False, content="", error=str(exc))

        if not resolved.exists():
            return ToolResult(success=False, content="", error=f"工具结果文件不存在: {file_path}")
        if not resolved.is_file():
            return ToolResult(success=False, content="", error=f"路径不是文件: {file_path}")

        results: list[str] = []
        truncated = False
        try:
            with resolved.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_no, line in enumerate(handle, start=1):
                    text = line.rstrip("\n")
                    if regex.search(text):
                        results.append(f"{resolved.name}:{line_no}: {text[:SNIPPET_LENGTH]}")
                        if len(results) >= MAX_RESULTS:
                            truncated = True
                            break
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"读取工具结果失败: {exc}")

        if not results:
            return ToolResult(success=True, content="(无匹配)")

        header = f"找到 {len(results)} 条匹配"
        if truncated:
            header += f"（已截断到前 {MAX_RESULTS} 条）"
        return ToolResult(success=True, content=header + "\n" + "\n".join(results))

"""Glob tool — find files by pattern."""

import asyncio
from pathlib import Path
from itertools import islice

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string
from tinyCode.security.sensitive_paths import is_sensitive_path

MAX_RESULTS = 50


class GlobTool(BaseTool):
    """Find files matching a glob pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "按 glob 模式查找文件。返回匹配的文件路径列表。最多返回 50 条。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("pattern", "string", "Glob 模式，如 '**/*.py' 或 'src/**/*.ts'。"),
        ]

    async def execute(self, pattern: str) -> ToolResult:
        cwd = get_workspace_root()
        try:
            pattern = require_string(pattern, "pattern")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))
        p = Path(pattern)
        if p.is_absolute():
            return ToolResult(success=False, content="", error=f"不允许绝对路径: {pattern}")
        if ".." in p.parts:
            return ToolResult(success=False, content="", error=f"路径遍历不被允许: {pattern}")

        return await asyncio.to_thread(self._search, cwd, pattern)

    @staticmethod
    def _search(cwd: Path, pattern: str) -> ToolResult:
        try:
            matches = sorted(islice(cwd.glob(pattern), MAX_RESULTS + 1))
        except Exception as exc:
            return ToolResult(
                success=False, content="", error=f"Glob 模式无效: {exc}"
            )

        if not matches:
            return ToolResult(success=True, content="(无匹配文件)")

        truncated = len(matches) > MAX_RESULTS
        matches = matches[:MAX_RESULTS]

        lines = []
        for m in matches:
            rel = m.relative_to(cwd) if m.is_relative_to(cwd) else m
            if is_sensitive_path(str(rel)):
                continue
            lines.append(str(rel))

        if not lines:
            return ToolResult(success=True, content="(无匹配文件)")

        summary = (
            f"找到 {len(lines)} 个匹配（已截断到前 {MAX_RESULTS} 条）\n"
            if truncated
            else f"找到 {len(lines)} 个匹配\n"
        )
        return ToolResult(success=True, content=summary + "\n".join(lines))

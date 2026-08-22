"""Read file tool."""

import asyncio
from pathlib import Path

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string
from tinyCode.security.sensitive_paths import is_sensitive_path


class ReadFileTool(BaseTool):
    """Read the contents of a file."""

    _allowed_encodings = ("utf-8", "gbk", "latin-1")
    _max_chars = 200_000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取指定文件的内容。返回文件文本。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径，相对于工作目录。"),
            ToolParameter("offset", "integer", "从第几个字符开始读取，默认 0。", required=False),
            ToolParameter("limit", "integer", "最多读取字符数，默认且最大 200000。", required=False),
        ]

    async def execute(
        self,
        path: str,
        offset: int = 0,
        limit: int = _max_chars,
    ) -> ToolResult:
        try:
            resolved = self._resolve(path)
        except ValueError as e:
            return ToolResult(success=False, content="", error=str(e))

        if is_sensitive_path(path):
            return ToolResult(
                success=False,
                content="",
                error="拒绝读取包含模型凭据的本地配置文件",
            )

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return ToolResult(success=False, content="", error="offset 必须是非负整数")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self._max_chars
        ):
            return ToolResult(
                success=False,
                content="",
                error=f"limit 必须是 1 到 {self._max_chars} 之间的整数",
            )

        if not resolved.exists():
            return ToolResult(success=False, content="", error=f"文件不存在: {path}")
        if not resolved.is_file():
            return ToolResult(success=False, content="", error=f"路径不是文件: {path}")

        # Offset scans over very large files are blocking I/O. Keep them off
        # the event-loop thread so progress, cancellation and timeouts remain
        # responsive even when the requested offset is near EOF.
        return await asyncio.to_thread(self._read, resolved, path, offset, limit)

    def _read(
        self, resolved: Path, display_path: str, offset: int, limit: int,
    ) -> ToolResult:
        for enc in self._allowed_encodings:
            try:
                with resolved.open("r", encoding=enc) as handle:
                    remaining = offset
                    while remaining:
                        skipped = handle.read(min(remaining, 8192))
                        if not skipped:
                            break
                        remaining -= len(skipped)
                    text = handle.read(limit + 1)
                truncated = len(text) > limit
                text = text[:limit]
                if truncated:
                    text += (
                        f"\n\n[内容已分页：当前 offset={offset}, limit={limit}；"
                        f"继续读取请使用 offset={offset + limit}]"
                    )
                return ToolResult(success=True, content=text)
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"读取文件失败: {display_path}: {exc}",
                )

        return ToolResult(
            success=False,
            content="",
            error=(
                "无法解码文件（尝试了 "
                f"{', '.join(self._allowed_encodings)}）: {display_path}"
            ),
        )

    def _resolve(self, path: str) -> Path:
        path = require_string(path, "path")
        p = Path(path)
        if p.is_absolute():
            raise ValueError(f"不允许绝对路径: {path}")
        if ".." in p.parts:
            raise ValueError(f"路径遍历不被允许: {path}")
        cwd = get_workspace_root()
        # Resolve symlinks so direct tool calls cannot escape the project root.
        try:
            resolved = (cwd / p).resolve(strict=False)
        except Exception:
            raise ValueError(f"无效路径: {path}")
        try:
            resolved.relative_to(cwd)
        except ValueError:
            raise ValueError(f"路径遍历不被允许: {path}")
        return resolved

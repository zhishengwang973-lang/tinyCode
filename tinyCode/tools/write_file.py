"""Write file tool."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string
from tinyCode.security.sensitive_paths import is_sensitive_path


MAX_WRITE_CHARS = 5_000_000


class WriteFileTool(BaseTool):
    """Write content to a file (creates if not exists)."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "将内容写入文件。如果文件已存在，操作失败并提示。目录不存在时自动创建。"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径，相对于工作目录。"),
            ToolParameter("content", "string", "要写入的完整文本内容。"),
        ]

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            resolved = self._resolve(path)
            content = require_string(content, "content")
        except ValueError as e:
            return ToolResult(success=False, content="", error=str(e))

        if is_sensitive_path(path):
            return ToolResult(
                success=False,
                content="",
                error="拒绝写入包含模型凭据的本地配置文件",
            )

        if len(content) > MAX_WRITE_CHARS:
            return ToolResult(
                success=False,
                content="",
                error=f"写入内容过大，write_file 最大支持 {MAX_WRITE_CHARS} 字符",
            )

        if resolved.exists():
            return ToolResult(
                success=False,
                content="",
                error=f"文件已存在: {path}。请使用 edit_file 修改内容，或先删除再写入。",
            )

        tmp_path: Path | None = None
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=resolved.parent,
                prefix=f".{resolved.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            # link() is an atomic create-if-absent operation; it cannot
            # overwrite a file that appeared after the exists() check.
            os.link(tmp_path, resolved)
        except FileExistsError:
            return ToolResult(
                success=False,
                content="",
                error=f"文件已存在: {path}。请使用 edit_file 修改内容。",
            )
        except OSError as exc:
            return ToolResult(
                success=False,
                content="",
                error=f"写入文件失败: {path}: {exc}",
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return ToolResult(success=True, content=f"已写入文件: {path} ({len(content)} 字符)")

    def _resolve(self, path: str) -> Path:
        path = require_string(path, "path")
        p = Path(path)
        if p.is_absolute():
            raise ValueError(f"不允许绝对路径: {path}")
        if ".." in p.parts:
            raise ValueError(f"路径遍历不被允许: {path}")
        cwd = get_workspace_root()
        resolved = (cwd / p).resolve(strict=False)
        try:
            resolved.relative_to(cwd)
        except ValueError:
            raise ValueError(f"路径遍历不被允许: {path}")
        return resolved

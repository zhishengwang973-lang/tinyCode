"""Delete file tool — remove a single project-local regular file."""

import os
from pathlib import Path

from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string
from tinyCode.security.sensitive_paths import is_sensitive_path


class DeleteFileTool(BaseTool):
    """Delete one regular file inside the current project directory."""

    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return (
            "删除工作目录中的单个普通文件。只接受相对路径；不支持目录、通配符、"
            "递归删除或指向项目外的符号链接。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("path", "string", "要删除的文件路径，相对于工作目录。"),
        ]

    def workspace_lock_paths(self, params: dict) -> tuple[str, ...] | None:
        path = params.get("path")
        return (path,) if isinstance(path, str) else None

    async def execute(self, path: str) -> ToolResult:
        try:
            candidate = self._resolve(path)
        except ValueError as e:
            return ToolResult(success=False, content="", error=str(e))

        if is_sensitive_path(path):
            return ToolResult(
                success=False,
                content="",
                error="拒绝删除包含模型凭据的本地配置文件",
            )

        try:
            original_stat = candidate.stat()
            if candidate.is_symlink() or not candidate.is_file():
                return ToolResult(success=False, content="", error=f"路径不是普通文件: {path}")
        except FileNotFoundError:
            return ToolResult(success=False, content="", error=f"文件不存在: {path}")
        except OSError as exc:
            return ToolResult(success=False, content="", error=f"读取文件状态失败: {path}: {exc}")

        try:
            current_stat = candidate.stat()
            if (
                current_stat.st_ino != original_stat.st_ino
                or current_stat.st_size != original_stat.st_size
                or current_stat.st_mtime_ns != original_stat.st_mtime_ns
            ):
                return ToolResult(
                    success=False,
                    content="",
                    error="文件在删除期间已被其他进程修改，已拒绝删除",
                )
            candidate.unlink()
            try:
                directory_fd = os.open(candidate.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Directory fsync is a durability enhancement and not
                # available on every platform/filesystem.
                pass
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"删除文件失败: {path}: {e}",
            )

        return ToolResult(success=True, content=f"已删除文件: {path}")

    def _resolve(self, path: str) -> Path:
        path = require_string(path, "path")
        p = Path(path)
        if p.is_absolute():
            raise ValueError(f"不允许绝对路径: {path}")
        if ".." in p.parts:
            raise ValueError(f"路径遍历不被允许: {path}")

        cwd = get_workspace_root()
        candidate = cwd / p
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(cwd)
        except ValueError:
            raise ValueError(f"路径遍历不被允许: {path}")
        return candidate

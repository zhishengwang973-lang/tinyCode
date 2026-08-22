"""Delete file tool — remove a single project-local regular file."""

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

        if not candidate.exists():
            return ToolResult(success=False, content="", error=f"文件不存在: {path}")
        if candidate.is_symlink() or not candidate.is_file():
            return ToolResult(success=False, content="", error=f"路径不是普通文件: {path}")

        try:
            candidate.unlink()
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

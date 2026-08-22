"""Edit file tool — exact-match replacement."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string
from tinyCode.security.sensitive_paths import is_sensitive_path


MAX_EDIT_FILE_BYTES = 5_000_000


class EditFileTool(BaseTool):
    """Replace a unique string in a file with another string."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "精确替换文件中的某段文本。old_string 必须在文件中恰好出现一次，"
            "否则操作失败。匹配时区分大小写，不忽略空白。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径，相对于工作目录。"),
            ToolParameter("old_string", "string", "要替换的原文。必须与文件中内容逐字符完全匹配。"),
            ToolParameter("new_string", "string", "替换后的新文本。"),
        ]

    async def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        try:
            resolved = self._resolve(path)
            old_string = require_string(old_string, "old_string")
            new_string = require_string(new_string, "new_string")
        except ValueError as e:
            return ToolResult(success=False, content="", error=str(e))
        if is_sensitive_path(path):
            return ToolResult(
                success=False,
                content="",
                error="拒绝修改包含模型凭据的本地配置文件",
            )
        if old_string == "":
            return ToolResult(success=False, content="", error="old_string 不能为空")

        if not resolved.exists():
            return ToolResult(success=False, content="", error=f"文件不存在: {path}")
        if not resolved.is_file():
            return ToolResult(success=False, content="", error=f"路径不是文件: {path}")
        try:
            original_stat = resolved.stat()
            if original_stat.st_size > MAX_EDIT_FILE_BYTES:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"文件过大，edit_file 最大支持 {MAX_EDIT_FILE_BYTES} 字节: {path}",
                )
        except OSError as e:
            return ToolResult(success=False, content="", error=f"读取文件状态失败: {path}: {e}")

        try:
            original = resolved.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"读取文件失败: {path}: {e}",
            )
        count = original.count(old_string)

        if count == 0:
            return ToolResult(
                success=False,
                content="",
                error=(
                    f"未找到匹配的原文。请确认 old_string 与文件中的文本逐字符一致"
                    f"（包括空白和换行）。文件内容预览（前 500 字符）:\n{original[:500]}"
                ),
            )
        if count > 1:
            return ToolResult(
                success=False,
                content="",
                error=(
                    f"找到 {count} 处匹配，old_string 必须在文件中只出现一次。"
                    f"请提供足够长的上下文以确保唯一性。"
                ),
            )

        new_content = original.replace(old_string, new_string, 1)
        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=resolved.parent,
                prefix=f".{resolved.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(new_content)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.chmod(tmp_path, original_stat.st_mode)
            current_stat = resolved.stat()
            if (
                current_stat.st_ino != original_stat.st_ino
                or current_stat.st_size != original_stat.st_size
                or current_stat.st_mtime_ns != original_stat.st_mtime_ns
            ):
                raise RuntimeError("文件在编辑期间已被其他进程修改，已拒绝覆盖")
            tmp_path.replace(resolved)
        except Exception as e:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return ToolResult(
                success=False,
                content="",
                error=f"写入文件失败: {path}: {e}",
            )
        return ToolResult(
            success=True,
            content=f"已编辑文件: {path}（替换了 1 处）",
        )

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

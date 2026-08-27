"""Grep tool — search for a pattern in file contents."""

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string

MAX_RESULTS = 50
SNIPPET_LENGTH = 300  # max chars of context per match
MAX_FILE_BYTES = 2_000_000
MAX_PATTERN_CHARS = 1_000
RG_TIMEOUT_SECONDS = 30.0
_NESTED_REPEAT_RE = re.compile(
    r"\((?:[^()\\]|\\.)*[*+](?:[^()\\]|\\.)*\)\s*(?:[*+]|\{\d*,?\d*\})"
)


class GrepTool(BaseTool):
    """Search file contents for a regex pattern."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "在文件内容中搜索正则表达式。返回文件名和匹配行。最多 50 条。"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("pattern", "string", "正则表达式，如 'def main' 或 'import.*os'。"),
        ]

    async def execute(self, pattern: str) -> ToolResult:
        cwd = get_workspace_root()
        try:
            pattern = require_string(pattern, "pattern")
            if len(pattern) > MAX_PATTERN_CHARS:
                raise ValueError(
                    f"pattern 最大支持 {MAX_PATTERN_CHARS} 个字符"
                )
            if _NESTED_REPEAT_RE.search(pattern):
                raise ValueError("pattern 包含可能导致灾难性回溯的嵌套重复")
            regex = re.compile(pattern)
        except (ValueError, re.error) as exc:
            return ToolResult(
                success=False, content="", error=f"正则表达式无效: {exc}"
            )

        # Repository traversal and decoding are blocking filesystem work. Run
        # them outside the event-loop thread so progress, cancellation and
        # other background tasks remain responsive on large projects.
        return await asyncio.to_thread(self._search, cwd, regex, pattern)

    @staticmethod
    def _search(cwd: Path, regex: re.Pattern[str], pattern: str) -> ToolResult:
        """Use ripgrep when installed; retain the portable Python fallback."""
        rg = shutil.which("rg")
        if rg:
            result = GrepTool._search_with_rg(cwd, pattern, rg)
            if result is not None:
                return result
        return GrepTool._search_python(cwd, regex)

    @staticmethod
    def _search_with_rg(
        cwd: Path, pattern: str, executable: str,
    ) -> ToolResult | None:
        """Return ``None`` when ripgrep cannot safely replace Python regex."""
        command = [
            executable,
            "--json",
            "--no-messages",
            "--no-ignore",
            "--max-filesize", "2M",
            "--max-count", str(MAX_RESULTS),
            "--max-columns", str(SNIPPET_LENGTH),
            "--glob", "!.git/**",
            "--glob", "!.hg/**",
            "--glob", "!.svn/**",
            "--glob", "!node_modules/**",
            "--glob", "!.venv/**",
            "--glob", "!venv/**",
            "--glob", "!__pycache__/**",
            "--",
            pattern,
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return None

        results: list[str] = []
        truncated = False
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "match":
                    continue
                data = event.get("data")
                if not isinstance(data, dict):
                    continue
                path_data = data.get("path")
                line_data = data.get("lines")
                line_number = data.get("line_number")
                if not isinstance(path_data, dict) or not isinstance(line_data, dict):
                    continue
                path = path_data.get("text")
                line = line_data.get("text")
                if (
                    not isinstance(path, str)
                    or not isinstance(line, str)
                    or not isinstance(line_number, int)
                ):
                    continue
                results.append(f"{path}:{line_number}: {line.rstrip()[:SNIPPET_LENGTH]}")
                if len(results) >= MAX_RESULTS:
                    truncated = True
                    process.terminate()
                    break
            return_code = process.wait(timeout=RG_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
            return None
        finally:
            if process.stdout is not None:
                process.stdout.close()

        # Exit 2 means the server's Rust regex engine rejected a construct
        # accepted by Python. Preserve compatibility by using the fallback.
        if return_code not in {0, 1, -15}:
            return None
        if not results:
            return ToolResult(success=True, content="(无匹配)")
        header = f"找到 {len(results)} 条匹配"
        if truncated:
            header += f"（已截断到前 {MAX_RESULTS} 条）"
        return ToolResult(success=True, content=header + "\n" + "\n".join(results))

    @staticmethod
    def _search_python(cwd: Path, regex: re.Pattern[str]) -> ToolResult:
        results: list[str] = []
        truncated = False

        # Walk all files (skip common binary/dot dirs)
        skip_dirs = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}
        for file_path in cwd.rglob("*"):
            if file_path.is_dir():
                continue
            # Skip hidden and binary-looking
            if any(part.startswith(".") for part in file_path.parts[len(cwd.parts):]):
                continue
            if any(d in file_path.parts for d in skip_dirs):
                continue
            suffix = file_path.suffix.lower()
            if suffix in {".pyc", ".pyo", ".exe", ".dll", ".so", ".o", ".bin", ".jpg", ".png", ".pdf"}:
                continue

            try:
                if not file_path.resolve().is_relative_to(cwd.resolve()):
                    continue
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = file_path.relative_to(cwd) if file_path.is_relative_to(cwd) else file_path
                    snippet = line[:SNIPPET_LENGTH]
                    results.append(f"{rel}:{line_no}: {snippet}")
                    if len(results) >= MAX_RESULTS:
                        truncated = True
                        break
            if truncated:
                break

        if not results:
            return ToolResult(success=True, content="(无匹配)")

        header = f"找到 {len(results)} 条匹配"
        if truncated:
            header += f"（已截断到前 {MAX_RESULTS} 条）"
        return ToolResult(success=True, content=header + "\n" + "\n".join(results))

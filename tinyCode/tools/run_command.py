"""Run command tool — execute a shell command safely."""

import asyncio
import os
import signal
import shlex
from pathlib import Path

from tinyCode.conversation.truncator import DEFAULT_PER_RESULT_THRESHOLD
from tinyCode.security.blacklist import check_blacklist
from tinyCode.security.sensitive_paths import command_references_sensitive_path
from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string

# Commands that are never allowed
_BLOCKED_COMMANDS = {
    "rm", "sudo", "chmod", "chown", "su", "shutdown", "reboot",
    "mkfs", "dd", ":(){",  # fork bomb pattern
}

# Commands that require no arguments to be interactive
_INTERACTIVE_COMMANDS = {
    "vim", "vi", "nano", "emacs", "ssh", "telnet", "top", "htop",
    "less", "more", "man",
}

# This must remain above the conversation truncator threshold. Otherwise a
# command can discard its large output before the session layer has a chance
# to persist it and expose tool_result_search/tool_result_read.
OUTPUT_LIMIT = DEFAULT_PER_RESULT_THRESHOLD * 2
COMMAND_TIMEOUT = 25.0
_SENSITIVE_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _sanitized_subprocess_env() -> dict[str, str]:
    """Keep ordinary build environment while withholding credential-like keys."""
    return {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }


async def _terminate_process(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None:
        return
    try:
        if os.name == "posix":
            # The shell may have exited while a background descendant still
            # owns stdout/stderr. Kill the process group even when the direct
            # child's returncode is already set, otherwise those descendants
            # survive a timeout and keep the pipes open.
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            if proc.returncode is not None:
                return
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
        return
    except asyncio.TimeoutError:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass


async def _read_limited(
    stream: asyncio.StreamReader | None,
    byte_limit: int = OUTPUT_LIMIT * 4,
) -> tuple[bytes, bool]:
    """Drain a stream fully while retaining only a bounded prefix."""
    if stream is None:
        return b"", False
    kept = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = byte_limit - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])
        if len(chunk) > max(0, remaining):
            truncated = True
    return bytes(kept), truncated


class RunCommandTool(BaseTool):
    """Execute a shell command within the project directory."""

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "在工作目录中执行一条 shell 命令。"
            f"stdout/stderr 各最多保留 {OUTPUT_LIMIT} 字符，超出会截断；"
            "超过会话阈值的结果将保存到项目缓存供分段读取。"
            "禁止交互式命令和危险命令（rm/sudo/chmod 等）。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("command", "string", "要执行的 shell 命令。"),
        ]

    async def execute(self, command: str) -> ToolResult:
        try:
            command = require_string(command, "command")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        try:
            command_parts = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=f"命令解析失败: {exc}")
        if not command_parts:
            return ToolResult(success=False, content="", error="command 不能为空")
        cmd_name = Path(command_parts[0]).name if command_parts else ""

        # Block dangerous command patterns using the central security rules.
        blacklisted = check_blacklist(command)
        if blacklisted:
            return ToolResult(success=False, content="", error=blacklisted)
        if command_references_sensitive_path(command):
            return ToolResult(
                success=False,
                content="",
                error="拒绝通过命令访问包含模型凭据的本地配置或环境变量",
            )

        # Block simple dangerous command names as an extra local guard.
        if cmd_name in _BLOCKED_COMMANDS:
            return ToolResult(
                success=False,
                content="",
                error=f"禁止执行危险命令: {cmd_name}",
            )
        if cmd_name in _INTERACTIVE_COMMANDS:
            return ToolResult(
                success=False,
                content="",
                error=f"禁止交互式命令: {cmd_name}",
            )

        proc: asyncio.subprocess.Process | None = None
        try:
            if os.name == "posix":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(get_workspace_root()),
                    start_new_session=True,
                    env=_sanitized_subprocess_env(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(get_workspace_root()),
                    env=_sanitized_subprocess_env(),
                )
            _, stdout_data, stderr_data = await asyncio.wait_for(
                asyncio.gather(
                    proc.wait(),
                    _read_limited(proc.stdout),
                    _read_limited(proc.stderr),
                ),
                timeout=COMMAND_TIMEOUT,
            )
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            return ToolResult(
                success=False,
                content="",
                error=f"命令执行超时（{COMMAND_TIMEOUT:g}s）",
            )
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise
        except Exception as exc:
            await _terminate_process(proc)
            return ToolResult(
                success=False,
                content="",
                error=f"命令执行异常: {type(exc).__name__}: {exc}",
            )

        stdout_bytes, stdout_truncated = stdout_data
        stderr_bytes, stderr_truncated = stderr_data
        stdout_decoded = stdout_bytes.decode("utf-8", errors="replace")
        stderr_decoded = stderr_bytes.decode("utf-8", errors="replace")
        stdout = stdout_decoded[:OUTPUT_LIMIT]
        stderr = stderr_decoded[:OUTPUT_LIMIT]
        truncated = (
            stdout_truncated
            or stderr_truncated
            or len(stdout_decoded) > OUTPUT_LIMIT
            or len(stderr_decoded) > OUTPUT_LIMIT
        )

        result_lines = []
        if proc.returncode == 0:
            result_lines.append(f"退出码: 0")
        else:
            result_lines.append(f"退出码: {proc.returncode}（非零）")
        result_lines.append(f"\n--- stdout ---\n{stdout or '(无输出)'}")
        if stderr:
            result_lines.append(f"\n--- stderr ---\n{stderr}")
        if truncated:
            result_lines.append("\n(输出已截断)")

        return ToolResult(
            success=proc.returncode == 0,
            content="\n".join(result_lines),
        )

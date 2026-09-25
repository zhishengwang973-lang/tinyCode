"""Git merger — incremental merge with LLM conflict resolution."""

import asyncio
import os
import re
import shlex
from pathlib import Path

from tinyCode.providers.base import BaseProvider
from tinyCode.security.sensitive_paths import is_sensitive_path
from tinyCode.tools.run_command import _read_limited, _terminate_process

MAX_CONFLICT_FILE_CHARS = 500_000
MAX_RESOLUTION_CHARS = 1_000_000

def _status_path(line: str) -> str:
    """Extract a best-effort path from a porcelain-v1 status row."""
    raw = line[3:].strip() if len(line) >= 3 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"')


class GitMerger:
    """Merges worktree branches back to main, using LLM for conflicts."""

    def __init__(
        self,
        provider: BaseProvider,
        repo_root: Path,
        *,
        allow_llm_conflicts: bool = True,
    ) -> None:
        self._provider = provider
        self._repo_root = repo_root
        self._allow_llm_conflicts = allow_llm_conflicts

    async def merge(self, source_branch: str, target_branch: str = "") -> tuple[bool, str]:
        """Merge *source_branch* into *target_branch*.

        Returns ``(success, message)``.
        """
        # Never switch the user's checkout behind their back.  Merging also
        # requires a clean target so unrelated work cannot be staged later.
        code, current, err = await self._git("branch", "--show-current")
        if code != 0:
            return False, f"无法读取当前分支: {err}"
        target_branch = target_branch or current.strip()
        if not target_branch:
            return False, "当前处于 detached HEAD，无法确定 Team 合并目标"
        if current.strip() != target_branch:
            return False, f"当前分支是 {current.strip() or '(detached)'}，请先切换到 {target_branch}"
        code, status, err = await self._git("status", "--porcelain")
        if code != 0:
            return False, f"无法检查工作区状态: {err}"
        if status.strip():
            return False, "目标工作区存在未提交修改，已拒绝自动合并"

        # 2. Merge source
        code, out, err = await self._git("merge", source_branch, "--no-commit", "--no-ff")
        if code == 0:
            merge_head_code, _, _ = await self._git(
                "rev-parse", "--verify", "-q", "MERGE_HEAD",
            )
            if merge_head_code != 0:
                return True, f"{source_branch} 已是最新，无需合并"
            commit_code, _, commit_err = await self._git(
                "commit", "-m", f"Merge {source_branch}",
            )
            if commit_code == 0:
                return True, f"已合并 {source_branch}（无冲突）"
            await self._git("merge", "--abort")
            return False, f"合并提交失败，已回滚: {commit_err}"

        # 3. Conflicts — use LLM
        conflict_files = await self._get_conflict_files()
        if not conflict_files:
            await self._git("merge", "--abort")
            return False, f"合并冲突但无法定位冲突文件，已回滚"
        if not self._allow_llm_conflicts:
            await self._git("merge", "--abort")
            return False, (
                f"合并 {source_branch} 发生冲突；自动 LLM 裁决未启用，已回滚"
            )

        try:
            resolved = await self._resolve_conflicts(conflict_files)
        except asyncio.CancelledError:
            await self._git("merge", "--abort")
            raise
        except Exception as exc:
            await self._git("merge", "--abort")
            return False, (
                "合并冲突处理失败，已回滚: "
                f"{type(exc).__name__}: {exc}"
            )
        if not resolved:
            await self._git("merge", "--abort")
            return False, f"合并 {source_branch} 失败：LLM 无法解决冲突，已回滚"

        add_code, _, add_err = await self._git("add", "--", *conflict_files)
        if add_code != 0:
            await self._git("merge", "--abort")
            return False, f"暂存冲突解决结果失败，已回滚: {add_err}"
        unresolved_code, unresolved, unresolved_err = await self._git(
            "diff", "--name-only", "--diff-filter=U",
        )
        if unresolved_code != 0 or unresolved.strip():
            await self._git("merge", "--abort")
            detail = unresolved.strip() or unresolved_err
            return False, f"冲突仍未完全解决，已回滚: {detail}"
        commit_code, _, commit_err = await self._git(
            "commit", "-m", f"Merge {source_branch} (LLM resolved conflicts)",
        )
        if commit_code != 0:
            await self._git("merge", "--abort")
            return False, f"冲突解决提交失败，已回滚: {commit_err}"
        return True, f"已合并 {source_branch}（LLM 解决 {len(conflict_files)} 个冲突）"

    async def inspect_worktree(
        self, worktree: Path, target_branch: str = "",
    ) -> tuple[bool, dict[str, str] | str]:
        """Capture a clean, already-integrated baseline for one member worktree."""
        code, top, err = await self._git(
            "-C", str(worktree), "rev-parse", "--show-toplevel",
        )
        if code != 0 or Path(top.strip()).resolve() != worktree.resolve():
            return False, f"不是有效 Git worktree: {err or worktree}"
        code, status, err = await self._git(
            "-C", str(worktree), "status", "--porcelain",
        )
        if code != 0:
            return False, f"检查 worktree 失败: {err}"
        if status.strip():
            return False, "worktree 存在任务开始前的未提交修改"
        code, branch, err = await self._git(
            "-C", str(worktree), "branch", "--show-current",
        )
        if code != 0 or not branch.strip():
            return False, f"无法读取 worktree 分支: {err or 'detached HEAD'}"
        code, head, err = await self._git(
            "-C", str(worktree), "rev-parse", "HEAD",
        )
        if code != 0 or not head.strip():
            return False, f"无法读取 worktree HEAD: {err}"
        if not target_branch:
            code, target, err = await self._git("branch", "--show-current")
            if code != 0 or not target.strip():
                return False, f"无法读取合并目标分支: {err or 'detached HEAD'}"
            target_branch = target.strip()
        code, _, _ = await self._git(
            "merge-base", "--is-ancestor", head.strip(), target_branch,
        )
        if code != 0:
            return False, (
                f"worktree 分支 '{branch.strip()}' 在本轮开始前已有未合并提交"
            )
        return True, {
            "branch": branch.strip(),
            "head": head.strip(),
            "target_branch": target_branch,
        }

    async def worktree_branch(self, worktree: Path) -> tuple[bool, str]:
        code, branch, err = await self._git(
            "-C", str(worktree), "branch", "--show-current",
        )
        if code != 0 or not branch.strip():
            return False, err or "worktree 处于 detached HEAD"
        return True, branch.strip()

    async def worktree_head(self, worktree: Path) -> tuple[bool, str]:
        code, head, err = await self._git(
            "-C", str(worktree), "rev-parse", "HEAD",
        )
        if code != 0 or not head.strip():
            return False, err or "无法读取 worktree HEAD"
        return True, head.strip()

    async def prepare_worktree(self, worktree: Path, member_name: str) -> tuple[bool, str]:
        """Commit a member's uncommitted work before its branch is merged."""
        code, status, err = await self._git("-C", str(worktree), "status", "--porcelain")
        if code != 0:
            return False, f"检查 worktree 失败: {err}"
        if not status.strip():
            return True, "无未提交修改"
        unsafe_paths = [
            path for path in (_status_path(line) for line in status.splitlines())
            if path and (
                is_sensitive_path(path)
                or Path(path).name.startswith(".tinyCode-")
                or Path(path).name == "settings.local.json"
            )
        ]
        if unsafe_paths:
            return False, "拒绝自动提交敏感配置: " + ", ".join(unsafe_paths)
        code, _, err = await self._git("-C", str(worktree), "add", "-A")
        if code != 0:
            return False, f"暂存 worktree 修改失败: {err}"
        code, _, err = await self._git(
            "-C", str(worktree), "diff", "--cached", "--check",
        )
        if code != 0:
            await self._git("-C", str(worktree), "reset")
            return False, f"变更完整性检查失败，已取消暂存: {err}"
        code, _, err = await self._git(
            "-C", str(worktree), "commit", "-m", f"TinyCode team: {member_name}",
        )
        if code != 0:
            return False, f"提交 worktree 修改失败: {err}"
        return True, "修改已提交"

    async def validate_worktree(
        self, worktree: Path, commands: list[str],
    ) -> tuple[bool, str]:
        """Run configured argv-safe validation commands in one member worktree."""
        for command in commands:
            try:
                argv = shlex.split(command)
            except ValueError as exc:
                return False, f"验证命令解析失败: {exc}"
            if not argv:
                return False, "验证命令不能为空"
            code, stdout, stderr = await self._run_command(worktree, argv)
            if code != 0:
                detail = (stderr or stdout).strip()[-2_000:]
                return False, f"验证失败 `{command}`: {detail or f'exit {code}'}"
        return True, "全部验证通过" if commands else "未配置验证命令"

    # -- internals -----------------------------------------------------------

    async def _git(self, *args: str) -> tuple[int, str, str]:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._repo_root),
                env=env,
                start_new_session=os.name == "posix",
            )
            _, stdout_data, stderr_data = await asyncio.wait_for(
                asyncio.gather(
                    proc.wait(),
                    _read_limited(proc.stdout, byte_limit=2_000_000),
                    _read_limited(proc.stderr, byte_limit=2_000_000),
                ),
                timeout=60.0,
            )
            stdout, _ = stdout_data
            stderr, _ = stderr_data
            return (proc.returncode or 0,
                    stdout.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"))
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            return -1, "", "git 命令执行超时"
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise
        except Exception as exc:
            return -1, "", str(exc)

    async def _run_command(
        self, cwd: Path, argv: list[str],
    ) -> tuple[int, str, str]:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=env,
                start_new_session=os.name == "posix",
            )
            _, stdout_data, stderr_data = await asyncio.wait_for(
                asyncio.gather(
                    proc.wait(),
                    _read_limited(proc.stdout, byte_limit=2_000_000),
                    _read_limited(proc.stderr, byte_limit=2_000_000),
                ),
                timeout=300.0,
            )
            stdout, _ = stdout_data
            stderr, _ = stderr_data
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            return -1, "", "验证命令执行超时（300s）"
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise
        except Exception as exc:
            return -1, "", f"{type(exc).__name__}: {exc}"

    async def _get_conflict_files(self) -> list[str]:
        code, out, _ = await self._git("diff", "--name-only", "--diff-filter=U")
        if code != 0:
            return []
        return [f for f in out.strip().split("\n") if f]

    async def _resolve_conflicts(self, files: list[str]) -> bool:
        for filepath in files:
            fp = (self._repo_root / filepath).resolve()
            try:
                fp.relative_to(self._repo_root.resolve())
            except ValueError:
                return False
            try:
                content = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                return False
            if len(content) > MAX_CONFLICT_FILE_CHARS:
                raise RuntimeError(
                    f"冲突文件过大，超过 {MAX_CONFLICT_FILE_CHARS} 字符: {filepath}"
                )

            # Ask LLM to resolve
            prompt = (
                f"以下是 git merge 冲突文件 '{filepath}'。请输出解决冲突后的完整文件内容。"
                f"只输出最终文件内容，不要加注释或解释。\n\n{content}"
            )
            resolved = ""
            begin_request = getattr(self._provider, "begin_request", None)
            if begin_request:
                begin_request()
            async for token in self._provider.chat_stream(
                [{"role": "user", "content": prompt}],
            ):
                if isinstance(token, str) and not token.startswith("<<"):
                    resolved += token
                    if len(resolved) > MAX_RESOLUTION_CHARS:
                        raise RuntimeError(
                            f"冲突解决结果超过 {MAX_RESOLUTION_CHARS} 字符"
                        )

            if not self._is_safe_resolution(resolved):
                return False
            try:
                fp.write_text(resolved, encoding="utf-8")
            except (OSError, UnicodeError):
                return False
        return True

    @staticmethod
    def _is_safe_resolution(content: str) -> bool:
        """Reject commentary wrappers and any remaining merge markers."""
        if not content.strip():
            return False
        stripped = content.strip()
        if re.fullmatch(r"```[^\n]*\n[\s\S]*\n```", stripped):
            return False
        marker = re.compile(r"^(<<<<<<<|=======|>>>>>>>)", re.MULTILINE)
        return marker.search(content) is None

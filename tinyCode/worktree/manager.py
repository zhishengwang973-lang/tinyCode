"""Git worktree manager — create, enter, exit, delete with full lifecycle."""

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from tinyCode.tools.run_command import _read_limited, _terminate_process
from tinyCode.worktree.models import WorktreeInfo
from tinyCode.worktree.validator import (
    dirname_to_name,
    legacy_name_to_dirname,
    name_to_branch,
    name_to_dirname,
    resolve_worktree_path,
    validate_name,
)

WORKTREES_DIR = ".tinyCode"  # relative to repo root
_LEGACY_SESSION_FILE = Path.home() / ".tinyCode" / "worktree_session.json"
# Kept as a patchable compatibility hook for integrations and tests.
SESSION_FILE = _LEGACY_SESSION_FILE


class GitWorktreeManager:
    """Manages git worktrees for agent isolation."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self._repo_root = (repo_root or self._find_repo_root()).resolve()
        self._available = (self._repo_root / ".git").exists()
        self._worktrees_dir = self._repo_root / WORKTREES_DIR / "worktrees"
        self._active: str = ""  # current worktree name ("" = main)

    # -- public API -----------------------------------------------------------

    async def create(
        self, name: str, branch: str = "", recover: bool = True,
    ) -> tuple[WorktreeInfo | None, str]:
        """Create a new worktree.

        If *recover* is True and the directory already exists, skips git worktree
        creation and reads HEAD directly (fast recovery).
        """
        ok, err = validate_name(name)
        if not ok:
            return None, err

        dir_name = name_to_dirname(name)
        target_path = self._worktrees_dir / dir_name
        legacy_path = self._worktrees_dir / legacy_name_to_dirname(name)
        if (
            recover
            and "/" in name
            and not target_path.exists()
            and legacy_path.exists()
        ):
            return None, (
                "检测到旧版扁平 Worktree 目录，但其原始名称存在歧义。"
                f"请使用扁平名称 '{legacy_path.name}' 进入，或先手动迁移。"
            )
        branch_name = branch or name_to_branch(name)

        # Fast recovery: directory exists
        if target_path.exists():
            if recover:
                code, top, err = await self._git(
                    "-C", str(target_path), "rev-parse", "--show-toplevel",
                )
                branch_code, actual_branch, branch_err = await self._git(
                    "-C", str(target_path), "branch", "--show-current",
                )
                if (
                    code != 0
                    or branch_code != 0
                    or Path(top.strip()).resolve() != target_path.resolve()
                    or actual_branch.strip() != branch_name
                ):
                    detail = err or branch_err or "目录不是预期的 Git worktree/分支"
                    return None, f"无法恢复已有目录: {detail}"
                head = self._read_head(target_path)
                info = WorktreeInfo(
                    name=name, path=str(target_path), branch=actual_branch.strip(),
                    head_commit=head, created_at="recovered",
                )
                return info, ""
            return None, f"目录已存在: {target_path}"

        # Create worktree via git
        target_path.parent.mkdir(parents=True, exist_ok=True)
        code, out, err_msg = await self._git(
            "worktree", "add", str(target_path), "-b", branch_name,
        )
        if code != 0:
            return None, f"git worktree add 失败: {err_msg}"

        head = self._read_head(target_path)
        info = WorktreeInfo(
            name=name, path=str(target_path), branch=branch_name,
            head_commit=head,
        )
        return info, ""

    async def enter(self, name: str) -> tuple[bool, str]:
        """Switch the working directory to a worktree."""
        if name:
            ok, err = validate_name(name)
            if not ok:
                return False, err

        previous_active = self._active
        previous_cwd = Path.cwd()
        if name:
            original_cwd = self._current_original_cwd()
            target_path = resolve_worktree_path(self._repo_root, name)
            if not target_path.exists():
                return False, f"工作目录不存在: {target_path}"
            valid_target, target_error = await self._validate_registered_worktree(
                target_path,
            )
            if not valid_target:
                return False, target_error
            try:
                # Mark it as recently used before switching.  A cleaner from
                # this or another TinyCode process must not treat an old,
                # clean worktree as abandoned while it is being entered.
                os.utime(target_path, None)
            except OSError as exc:
                return False, f"无法标记工作目录为活动状态: {exc}"
            destination = target_path
        else:
            # Exit back to main
            session = self.load_session()
            original_cwd = session.get("original_cwd", "") if session else ""
            requested = Path(original_cwd) if original_cwd else self._repo_root
            destination = requested if requested.is_dir() else self._repo_root

        try:
            os.chdir(str(destination))
        except OSError as exc:
            return False, f"无法切换工作目录: {exc}"

        self._active = name
        try:
            self._save_session(
                name,
                original_cwd=original_cwd if name else str(self._repo_root),
            )
        except (OSError, UnicodeError) as exc:
            self._active = previous_active
            try:
                os.chdir(previous_cwd)
            except OSError:
                pass
            return False, f"保存 Worktree 会话失败: {exc}"
        return True, ""

    async def exit(self, name: str, force: bool = False) -> tuple[bool, str]:
        """Exit a worktree and optionally remove it."""
        if not name:
            return False, "未指定工作目录名"
        ok, error = validate_name(name)
        if not ok:
            return False, error

        target_path = resolve_worktree_path(self._repo_root, name)

        if not target_path.exists():
            return False, f"工作目录不存在: {target_path}"

        # Change protection: check for uncommitted changes
        if not force:
            has_changes = await self._has_changes(target_path)
            if has_changes:
                return False, (
                    f"工作目录 '{name}' 有未提交的修改。"
                    f"请先提交/备份；--force 会永久删除这些修改。"
                )

        # Resolve the actual branch before removing the worktree; create()
        # allows callers to provide a custom branch name.
        branch_code, branch_out, branch_err = await self._git(
            "-C", str(target_path), "branch", "--show-current",
        )
        if branch_code != 0:
            return False, f"无法读取工作目录分支: {branch_err}"
        branch_name = branch_out.strip() or name_to_branch(name)

        if not force and not await self._branch_is_merged(branch_name):
            return False, (
                f"分支 '{branch_name}' 仍包含尚未合并到当前分支的提交。"
                "请先合并/备份；--force 才会永久删除该分支。"
            )

        # Switch back only when removing the active worktree.
        if name == self._active:
            switched, switch_error = await self.enter("")
            if not switched:
                return False, switch_error

        remove_code, _, remove_err = await self._git(
            "worktree", "remove", str(target_path), "--force",
        )
        if remove_code != 0:
            return False, f"git worktree remove 失败: {remove_err}"
        branch_flag = "-D" if force else "-d"
        branch_code, _, branch_err = await self._git("branch", branch_flag, branch_name)

        if name == self._active:
            self._active = ""
            self._save_session("")
        if branch_code != 0:
            return True, f"工作目录已删除，但分支 '{branch_name}' 删除失败: {branch_err}"
        return True, f"工作目录 '{name}' 及分支 '{branch_name}' 已删除"

    async def list_worktrees(self) -> list[WorktreeInfo]:
        """List all managed worktrees."""
        code, out, _ = await self._git("worktree", "list", "--porcelain")
        if code != 0:
            return []

        results: list[WorktreeInfo] = []
        active_path = (
            resolve_worktree_path(self._repo_root, self._active)
            if self._active else None
        )
        for block in out.strip().split("\n\n"):
            if not block:
                continue
            info = self._parse_porcelain(block)
            if info and self._is_managed_worktree_path(Path(info.path)):
                if active_path is not None:
                    info.is_active = Path(info.path).resolve() == active_path.resolve()
                results.append(info)
        if results:
            dirty_states = await asyncio.gather(*(
                self._has_changes(Path(info.path)) for info in results
            ))
            for info, dirty in zip(results, dirty_states):
                info.has_changes = dirty
        return results

    async def status(self, name: str = "") -> WorktreeInfo | None:
        """Get status of a worktree (or the active one)."""
        if not name:
            name = self._active
        if not name:
            return None
        ok, _ = validate_name(name)
        if not ok:
            return None

        target_path = resolve_worktree_path(self._repo_root, name)
        if not target_path.exists():
            return None

        head = self._read_head(target_path)
        has_changes = await self._has_changes(target_path)
        branch_code, branch_out, _ = await self._git(
            "-C", str(target_path), "branch", "--show-current",
        )
        branch = branch_out.strip() if branch_code == 0 else ""
        return WorktreeInfo(
            name=name, path=str(target_path),
            branch=branch or name_to_branch(name), head_commit=head,
            is_active=(name == self._active), has_changes=has_changes,
        )

    # -- cleanup --------------------------------------------------------------

    async def remove_stale(self, max_age_hours: int = 24) -> list[str]:
        """Remove worktrees older than *max_age_hours* with no changes."""
        if max_age_hours < 0:
            raise ValueError("max_age_hours 不能为负数")
        removed: list[str] = []
        cutoff = time.time() - max_age_hours * 3600
        worktrees = await self.list_worktrees()
        protected_names = self._active_session_names()
        for wt in worktrees:
            if wt.is_active or wt.name in protected_names:
                continue
            path = Path(wt.path)
            try:
                # Directory mtime is a conservative activity signal.  The
                # cleaner must never remove a newly-created worktree merely
                # because it is clean.
                if path.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            if await self._has_changes(path):
                continue  # fail-closed: don't remove dirty worktrees
            # Check branch pattern
            if not wt.branch.startswith("tinyCode/"):
                continue
            # Clean does not mean disposable: a branch may contain committed
            # work that has never been merged or pushed.  Automatic cleanup is
            # allowed only when its tip is already reachable from HEAD.
            if not await self._branch_is_merged(wt.branch):
                continue
            # Re-check after the awaited git inspections.  The user or a team
            # may have entered/touched this worktree since the first stat.
            if wt.name == self._active:
                continue
            try:
                if path.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            code, _, _ = await self._git("worktree", "remove", wt.path, "--force")
            if code != 0:
                continue
            branch_code, _, _ = await self._git("branch", "-d", wt.branch)
            if branch_code != 0:
                # The worktree is gone, but preserving the branch is the safe
                # failure mode and retains every commit.
                continue
            removed.append(wt.name)
        return removed

    # -- helpers --------------------------------------------------------------

    @property
    def active(self) -> str:
        return self._active

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def availability_error(self) -> str:
        return "" if self._available else "当前目录不在 Git 仓库中，Worktree 功能不可用"

    def load_session(self) -> dict[str, str] | None:
        """Load and validate the persisted worktree session."""
        session_path = self._session_path()
        if not session_path.exists():
            # Safely migrate the old global file only when it clearly points to
            # a worktree owned by this repository.
            if SESSION_FILE == _LEGACY_SESSION_FILE and _LEGACY_SESSION_FILE.exists():
                session_path = _LEGACY_SESSION_FILE
            else:
                return None
        try:
            value = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        active = value.get("active_worktree", "")
        original = value.get("original_cwd", "")
        if not isinstance(active, str) or not isinstance(original, str):
            return None
        stored_root = value.get("repo_root", "")
        if stored_root and (
            not isinstance(stored_root, str)
            or Path(stored_root).resolve() != self._repo_root
        ):
            return None
        if active:
            ok, _ = validate_name(active)
            if not ok or not resolve_worktree_path(self._repo_root, active).exists():
                return None
        if session_path == _LEGACY_SESSION_FILE and SESSION_FILE == _LEGACY_SESSION_FILE:
            try:
                original_path = Path(original).resolve()
                original_path.relative_to(self._repo_root)
            except (OSError, ValueError):
                return None
        return {"active_worktree": active, "original_cwd": original}

    def _session_path(self) -> Path:
        if SESSION_FILE != _LEGACY_SESSION_FILE:
            return SESSION_FILE
        identity = hashlib.sha256(
            str(self._repo_root).encode("utf-8")
        ).hexdigest()[:16]
        return Path.home() / ".tinyCode" / "worktree_sessions" / f"{identity}.json"

    def _active_session_names(self) -> set[str]:
        """Return active names persisted by this or another TinyCode process."""
        names: set[str] = set()
        directory = Path.home() / ".tinyCode" / "worktree_sessions"
        try:
            candidates = list(directory.glob("*.json"))
        except OSError:
            return names
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            stored_root = value.get("repo_root")
            active = value.get("active_worktree")
            if (
                isinstance(stored_root, str)
                and isinstance(active, str)
                and active
                and Path(stored_root).resolve() == self._repo_root
            ):
                names.add(active)
        return names

    def _is_managed_worktree_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._worktrees_dir.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _find_repo_root() -> Path:
        """Walk up from cwd to find the git repo root."""
        d = Path.cwd()
        while d != d.parent:
            git_marker = d / ".git"
            if git_marker.is_dir():
                return d
            if git_marker.is_file():
                try:
                    raw = git_marker.read_text(encoding="utf-8").strip()
                    git_dir = Path(raw.split(":", 1)[1].strip())
                    if not git_dir.is_absolute():
                        git_dir = (d / git_dir).resolve()
                    common_file = git_dir / "commondir"
                    if common_file.is_file():
                        common = Path(
                            common_file.read_text(encoding="utf-8").strip()
                        )
                        if not common.is_absolute():
                            common = (git_dir / common).resolve()
                        return common.parent
                except (IndexError, OSError, UnicodeError):
                    return d
                return d
            d = d.parent
        return Path.cwd()

    @staticmethod
    def _read_head(path: Path) -> str:
        try:
            head_file = path / ".git"
            if head_file.is_file():
                # worktree .git is a file pointing to the main repo
                git_dir = Path(head_file.read_text().strip().split(": ")[-1])
                head_file = git_dir / "HEAD"
            else:
                head_file = head_file / "HEAD"
            return head_file.read_text().strip()
        except (OSError, UnicodeError):
            return ""

    async def _has_changes(self, path: Path) -> bool:
        code, out, _ = await self._git("-C", str(path), "status", "--porcelain")
        # Fail closed: if status cannot be inspected, cleanup must preserve it.
        return code != 0 or bool(out.strip())

    async def _branch_is_merged(self, branch: str) -> bool:
        if not branch:
            return False
        code, _, _ = await self._git("merge-base", "--is-ancestor", branch, "HEAD")
        return code == 0

    async def _git(self, *args: str) -> tuple[int, str, str]:
        cmd = ["git", *args]
        proc: asyncio.subprocess.Process | None = None
        try:
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMPT"] = "0"
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
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
            return (-1, "", "git 命令执行超时")
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise
        except Exception as exc:
            return (-1, "", str(exc))

    def _parse_porcelain(self, block: str) -> WorktreeInfo | None:
        info: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("worktree "):
                info["path"] = line[9:]
            elif line.startswith("HEAD "):
                info["head"] = line[5:]
            elif line.startswith("branch "):
                info["branch"] = line[19:]  # refs/heads/xxx
        if "path" not in info:
            return None
        branch = info.get("branch", "").replace("refs/heads/", "")
        path = info["path"]
        # Flattening '/' to '-' is intentionally not reversed: doing so turned
        # legitimate hyphens into fake nested names and made lookups lossy.
        name = dirname_to_name(Path(path).name)
        return WorktreeInfo(
            name=name, path=path, branch=branch,
            head_commit=info.get("head", ""),
        )

    # -- session persistence --------------------------------------------------

    def _save_session(self, name: str, original_cwd: str = "") -> None:
        from tinyCode.storage.sessions import _atomic_write_text
        session = {
            "active_worktree": name,
            "original_cwd": original_cwd or str(self._repo_root),
            "repo_root": str(self._repo_root),
        }
        _atomic_write_text(self._session_path(), json.dumps(session, indent=2))

    def _current_original_cwd(self) -> str:
        session = self.load_session()
        if session and session.get("active_worktree") and session.get("original_cwd"):
            return session["original_cwd"]
        return str(Path.cwd())

    async def _validate_registered_worktree(
        self, target_path: Path,
    ) -> tuple[bool, str]:
        code, top, err = await self._git(
            "-C", str(target_path), "rev-parse", "--show-toplevel",
        )
        if code != 0 or Path(top.strip()).resolve() != target_path.resolve():
            return False, f"目录不是有效 Git worktree: {err or target_path}"
        code, common, err = await self._git(
            "-C", str(target_path), "rev-parse", "--git-common-dir",
        )
        if code != 0:
            return False, f"无法验证 Git worktree 归属: {err}"
        common_path = Path(common.strip())
        if not common_path.is_absolute():
            common_path = (target_path / common_path).resolve()
        expected = (self._repo_root / ".git").resolve()
        if common_path.resolve() != expected:
            return False, "目录属于另一个 Git 仓库，已拒绝进入"
        return True, ""

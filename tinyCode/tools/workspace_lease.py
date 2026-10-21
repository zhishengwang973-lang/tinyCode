"""Cross-process, path-scoped leases for workspace writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from tinyCode.time_utils import beijing_now_iso


DEFAULT_LEASE_TIMEOUT_SECONDS = 30.0
_POLL_SECONDS = 0.05
_GATE_NAME = "_gate.lock"
_WORKSPACE_NAME = "_workspace.lock"


class WorkspaceLeaseError(RuntimeError):
    """Raised when a cooperating writer cannot obtain a workspace lease."""


class WorkspaceWriteLease:
    """Acquire exact-file or whole-workspace leases across TinyCode processes.

    A short gate lease serializes the check-and-create phase.  Exact file
    leases can coexist; a whole-workspace lease conflicts with every file
    lease.  Lock ownership is recorded with a PID and random token so stale
    files from a dead process can be safely reclaimed.
    """

    def __init__(
        self,
        workspace: Path,
        paths: tuple[str, ...] | None,
        *,
        timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS,
    ) -> None:
        self._workspace = workspace.resolve()
        self._paths = _normalize_paths(paths)
        self._timeout = max(0.01, float(timeout_seconds))
        self._dir = self._workspace / ".tinyCode" / "locks" / "workspace-write"
        self._token = uuid.uuid4().hex
        self._held: list[Path] = []

    async def __aenter__(self) -> "WorkspaceWriteLease":
        deadline = time.monotonic() + self._timeout
        while True:
            acquired, reason = await asyncio.to_thread(self._try_acquire)
            if acquired:
                return self
            if time.monotonic() >= deadline:
                raise WorkspaceLeaseError(
                    "工作区写入冲突：" + (reason or "等待其他 TinyCode 写入超时")
                )
            await asyncio.sleep(_POLL_SECONDS)

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        await asyncio.to_thread(self.release)

    def release(self) -> None:
        for path in reversed(self._held):
            if _owner_token(path) != self._token:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._held = []

    def _try_acquire(self) -> tuple[bool, str]:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"无法创建工作区锁目录: {exc}"

        gate = self._dir / _GATE_NAME
        if not _create_lock(gate, self._token, ("gate",)):
            self._reclaim_if_stale(gate)
            return False, "其他写入正在协调锁"
        try:
            self._reclaim_stale_locks()
            requested = self._requested_lock_paths()
            existing = [
                path for path in self._dir.glob("*.lock")
                if path.name != _GATE_NAME
            ]
            conflict = self._find_conflict(existing, requested)
            if conflict:
                return False, f"{conflict} 正在被其他 TinyCode 写入"
            created: list[Path] = []
            for path in requested:
                if not _create_lock(path, self._token, self._paths):
                    for created_path in reversed(created):
                        created_path.unlink(missing_ok=True)
                    return False, "其他 TinyCode 刚取得了同一写入锁"
                created.append(path)
            self._held = created
            return True, ""
        finally:
            if _owner_token(gate) == self._token:
                try:
                    gate.unlink(missing_ok=True)
                except OSError:
                    pass

    def _requested_lock_paths(self) -> list[Path]:
        if self._paths is None:
            return [self._dir / _WORKSPACE_NAME]
        return [self._dir / _path_lock_name(path) for path in self._paths]

    def _find_conflict(self, existing: list[Path], requested: list[Path]) -> str:
        workspace_lock = self._dir / _WORKSPACE_NAME
        if workspace_lock in existing:
            return "整个工作区"
        if requested == [workspace_lock]:
            if existing:
                return "已有文件"
            return ""
        existing_set = set(existing)
        for path in requested:
            if path in existing_set:
                return _scope_label(path)
        return ""

    def _reclaim_stale_locks(self) -> None:
        for path in self._dir.glob("*.lock"):
            if path.name != _GATE_NAME:
                self._reclaim_if_stale(path)

    @staticmethod
    def _reclaim_if_stale(path: Path) -> None:
        owner = _read_owner(path)
        pid = owner.get("pid") if isinstance(owner, dict) else None
        if not isinstance(pid, int) or _pid_alive(pid):
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _normalize_paths(paths: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if paths is None:
        return None
    normalized: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            return None
        candidate = Path(raw)
        if not raw or candidate.is_absolute() or ".." in candidate.parts:
            return None
        normalized.add(candidate.as_posix())
    return tuple(sorted(normalized)) or None


def _path_lock_name(path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return f"file-{digest}.lock"


def _create_lock(path: Path, token: str, scopes: tuple[str, ...] | None) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    except OSError:
        return False
    try:
        payload = json.dumps({
            "pid": os.getpid(),
            "token": token,
            "scopes": list(scopes) if scopes is not None else ["<workspace>"],
            "created_at": beijing_now_iso(),
        }, ensure_ascii=False).encode("utf-8")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _read_owner(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _owner_token(path: Path) -> str:
    value = _read_owner(path).get("token", "")
    return value if isinstance(value, str) else ""


def _scope_label(path: Path) -> str:
    owner = _read_owner(path)
    scopes = owner.get("scopes") if isinstance(owner, dict) else None
    if isinstance(scopes, list) and scopes and all(isinstance(item, str) for item in scopes):
        return "、".join(scopes[:3])
    return "目标文件"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True

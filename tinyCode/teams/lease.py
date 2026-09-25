"""Cross-process lease preventing concurrent runs of the same Team."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from tinyCode.time_utils import beijing_now_iso


class TeamRunLease:
    def __init__(self, team_dir: Path) -> None:
        self._path = team_dir / "run.lock"
        self._token = uuid.uuid4().hex
        self._held = False

    def acquire(self) -> tuple[bool, str]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                fd = os.open(
                    self._path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                owner = self._read_owner()
                pid = owner.get("pid") if isinstance(owner, dict) else None
                if isinstance(pid, int) and self._pid_alive(pid):
                    return False, f"Team 已由进程 {pid} 运行，不能并发启动"
                try:
                    self._path.unlink()
                except OSError as exc:
                    return False, f"无法清理失效的 Team 运行锁: {exc}"
                continue
            except OSError as exc:
                return False, f"无法创建 Team 运行锁: {exc}"
            try:
                payload = json.dumps({
                    "pid": os.getpid(),
                    "token": self._token,
                    "started_at": beijing_now_iso(),
                }).encode("utf-8")
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._held = True
            return True, ""
        return False, "无法获取 Team 运行锁"

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        owner = self._read_owner()
        if isinstance(owner, dict) and owner.get("token") != self._token:
            return
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _read_owner(self) -> dict:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
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

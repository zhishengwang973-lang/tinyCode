"""Background cleaner — periodically removes expired worktree directories."""

import asyncio
import sys

from tinyCode.worktree.manager import GitWorktreeManager

CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes


class BackgroundCleaner:
    """Periodic background cleanup of stale worktrees."""

    def __init__(self, manager: GitWorktreeManager, max_age_hours: int = 24) -> None:
        self._manager = manager
        self._max_age_hours = max_age_hours
        self._task: asyncio.Task | None = None
        self.last_error = ""

    def start(self) -> bool:
        if not self._manager.is_available:
            return False
        if self._task is not None and not self._task.done():
            return True
        self._task = asyncio.ensure_future(self._loop())
        return True

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            await self._cleanup_once()

    async def _cleanup_once(self) -> list[str]:
        try:
            removed = await self._manager.remove_stale(self._max_age_hours)
            self.last_error = ""
            return removed
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self.last_error:
                print(f"Worktree 后台清理失败: {message}", file=sys.stderr)
            self.last_error = message
            return []

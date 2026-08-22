import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock

from tinyCode.worktree.cleaner import BackgroundCleaner
from tinyCode.worktree.manager import GitWorktreeManager


class BackgroundCleanerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleaner_does_not_start_outside_git_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = GitWorktreeManager(repo_root=Path(tmp))
            cleaner = BackgroundCleaner(manager)

            self.assertFalse(cleaner.start())
            self.assertIsNone(cleaner._task)

    async def test_cleanup_failure_is_recorded_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            manager = GitWorktreeManager(repo_root=root)
            manager.remove_stale = AsyncMock(side_effect=RuntimeError("git failed"))
            cleaner = BackgroundCleaner(manager)
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                removed = await cleaner._cleanup_once()

            self.assertEqual([], removed)
            self.assertIn("git failed", cleaner.last_error)
            self.assertIn("后台清理失败", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

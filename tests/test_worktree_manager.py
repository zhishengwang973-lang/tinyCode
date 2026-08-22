import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.worktree import manager as worktree_manager
from tinyCode.worktree.manager import GitWorktreeManager


class GitWorktreeManagerTests(unittest.IsolatedAsyncioTestCase):
    def test_non_git_directory_reports_worktree_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = GitWorktreeManager(repo_root=Path(tmp))

            self.assertFalse(manager.is_available)
            self.assertIn("不在 Git 仓库", manager.availability_error)

    def test_load_session_rejects_invalid_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_file = Path(tmp) / "session.json"
            session_file.write_text("[]", encoding="utf-8")
            manager = GitWorktreeManager(repo_root=Path(tmp))

            with patch.object(worktree_manager, "SESSION_FILE", session_file):
                self.assertIsNone(manager.load_session())

    async def test_enter_main_restores_repo_root_from_session_file(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            worktree_path = repo_root / ".tinyCode" / "worktrees" / "feature"
            session_file = Path(tmp) / "session.json"
            worktree_path.mkdir(parents=True)

            try:
                os.chdir(repo_root)
                manager = GitWorktreeManager(repo_root=repo_root)

                with patch.object(worktree_manager, "SESSION_FILE", session_file):
                    ok, msg = await manager.enter("feature")
                    self.assertTrue(ok, msg)
                    self.assertEqual(worktree_path.resolve(), Path.cwd().resolve())

                    ok, msg = await manager.enter("")

                self.assertTrue(ok, msg)
                self.assertEqual(repo_root.resolve(), Path.cwd().resolve())
            finally:
                os.chdir(original_cwd)

    async def test_list_worktrees_excludes_paths_that_only_share_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            manager = GitWorktreeManager(repo_root=repo_root)
            resolved_repo = manager.repo_root
            managed = resolved_repo / ".tinyCode" / "worktrees" / "feature"
            unmanaged = resolved_repo / ".tinyCode" / "worktrees_evil" / "feature"
            porcelain = "\n\n".join([
                "\n".join([
                    f"worktree {managed}",
                    "HEAD abc123",
                    "branch refs/heads/tinyCode/feature",
                ]),
                "\n".join([
                    f"worktree {unmanaged}",
                    "HEAD def456",
                    "branch refs/heads/tinyCode/evil",
                ]),
            ])

            async def fake_git(*args: str):
                return 0, porcelain, ""

            manager._git = fake_git

            worktrees = await manager.list_worktrees()

            self.assertEqual([str(managed)], [wt.path for wt in worktrees])

    async def test_remove_stale_keeps_recent_and_dirty_worktrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            recent = repo_root / ".tinyCode" / "worktrees" / "recent"
            dirty = repo_root / ".tinyCode" / "worktrees" / "dirty"
            stale = repo_root / ".tinyCode" / "worktrees" / "stale"
            for path in (recent, dirty, stale):
                path.mkdir(parents=True)
            old = time.time() - 48 * 3600
            os.utime(dirty, (old, old))
            os.utime(stale, (old, old))

            manager = GitWorktreeManager(repo_root=repo_root)
            manager.list_worktrees = unittest.mock.AsyncMock(return_value=[
                worktree_manager.WorktreeInfo("recent", str(recent), "tinyCode/recent"),
                worktree_manager.WorktreeInfo("dirty", str(dirty), "tinyCode/dirty"),
                worktree_manager.WorktreeInfo("stale", str(stale), "tinyCode/stale"),
            ])

            async def fake_has_changes(path: Path) -> bool:
                return path == dirty

            git_calls: list[tuple[str, ...]] = []

            async def fake_git(*args: str):
                git_calls.append(args)
                return 0, "", ""

            manager._has_changes = fake_has_changes
            manager._git = fake_git

            removed = await manager.remove_stale(max_age_hours=24)

            self.assertEqual(["stale"], removed)
            self.assertIn(("merge-base", "--is-ancestor", "tinyCode/stale", "HEAD"), git_calls)
            self.assertIn(("worktree", "remove", str(stale), "--force"), git_calls)
            self.assertIn(("branch", "-d", "tinyCode/stale"), git_calls)
            self.assertNotIn(str(recent), str(git_calls))
            self.assertNotIn(str(dirty), str(git_calls))

    async def test_remove_stale_preserves_clean_branch_with_unique_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            stale = repo_root / ".tinyCode" / "worktrees" / "stale"
            stale.mkdir(parents=True)
            old = time.time() - 48 * 3600
            os.utime(stale, (old, old))
            manager = GitWorktreeManager(repo_root=repo_root)
            manager.list_worktrees = unittest.mock.AsyncMock(return_value=[
                worktree_manager.WorktreeInfo("stale", str(stale), "tinyCode/stale"),
            ])
            manager._has_changes = unittest.mock.AsyncMock(return_value=False)
            manager._git = unittest.mock.AsyncMock(return_value=(1, "", "not merged"))

            removed = await manager.remove_stale(max_age_hours=24)

            self.assertEqual([], removed)
            self.assertNotIn("worktree", str(manager._git.await_args_list))

    async def test_remove_stale_never_removes_active_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            active = repo_root / ".tinyCode" / "worktrees" / "active"
            active.mkdir(parents=True)
            old = time.time() - 48 * 3600
            os.utime(active, (old, old))
            manager = GitWorktreeManager(repo_root=repo_root)
            manager.list_worktrees = unittest.mock.AsyncMock(return_value=[
                worktree_manager.WorktreeInfo(
                    "active", str(active), "tinyCode/active", is_active=True,
                ),
            ])
            manager._git = unittest.mock.AsyncMock(return_value=(0, "", ""))

            removed = await manager.remove_stale(max_age_hours=24)

            self.assertEqual([], removed)
            manager._git.assert_not_awaited()

    async def test_remove_stale_rechecks_activity_before_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            target = repo_root / ".tinyCode" / "worktrees" / "feature"
            target.mkdir(parents=True)
            old = time.time() - 48 * 3600
            os.utime(target, (old, old))
            manager = GitWorktreeManager(repo_root=repo_root)
            manager.list_worktrees = unittest.mock.AsyncMock(return_value=[
                worktree_manager.WorktreeInfo(
                    "feature", str(target), "tinyCode/feature",
                ),
            ])
            manager._has_changes = unittest.mock.AsyncMock(return_value=False)

            async def became_active(_: str) -> bool:
                os.utime(target, None)
                return True

            manager._branch_is_merged = became_active
            manager._git = unittest.mock.AsyncMock(return_value=(0, "", ""))

            removed = await manager.remove_stale(max_age_hours=24)

            self.assertEqual([], removed)
            manager._git.assert_not_awaited()

    async def test_enter_refreshes_worktree_activity_time(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            target = repo_root / ".tinyCode" / "worktrees" / "feature"
            session_file = Path(tmp) / "session.json"
            target.mkdir(parents=True)
            old = time.time() - 48 * 3600
            os.utime(target, (old, old))
            try:
                os.chdir(repo_root)
                manager = GitWorktreeManager(repo_root=repo_root)
                with patch.object(worktree_manager, "SESSION_FILE", session_file):
                    ok, message = await manager.enter("feature")

                self.assertTrue(ok, message)
                self.assertGreater(target.stat().st_mtime, old)
            finally:
                os.chdir(original_cwd)

    async def test_exit_warns_that_force_permanently_discards_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            target = repo_root / ".tinyCode" / "worktrees" / "feature"
            target.mkdir(parents=True)
            manager = GitWorktreeManager(repo_root=repo_root)
            manager._has_changes = unittest.mock.AsyncMock(return_value=True)

            ok, message = await manager.exit("feature")

            self.assertFalse(ok)
            self.assertIn("永久删除", message)

    async def test_enter_main_falls_back_when_original_directory_was_deleted(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            worktree_path = repo_root / ".tinyCode" / "worktrees" / "feature"
            session_file = Path(tmp) / "session.json"
            deleted_dir = Path(tmp) / "deleted"
            worktree_path.mkdir(parents=True)
            session_file.write_text(
                '{"active_worktree":"feature","original_cwd":"'
                + str(deleted_dir)
                + '"}',
                encoding="utf-8",
            )
            try:
                os.chdir(worktree_path)
                manager = GitWorktreeManager(repo_root=repo_root)
                manager._active = "feature"
                with patch.object(worktree_manager, "SESSION_FILE", session_file):
                    ok, message = await manager.enter("")

                self.assertTrue(ok, message)
                self.assertEqual(repo_root.resolve(), Path.cwd().resolve())
            finally:
                os.chdir(original_cwd)

    async def test_exit_uses_actual_custom_branch_and_reports_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            target = repo_root / ".tinyCode" / "worktrees" / "feature"
            target.mkdir(parents=True)
            manager = GitWorktreeManager(repo_root=repo_root)
            calls: list[tuple[str, ...]] = []

            async def fake_git(*args: str):
                calls.append(args)
                if args[-2:] == ("branch", "--show-current"):
                    return 0, "custom/branch\n", ""
                return 0, "", ""

            manager._git = fake_git
            ok, message = await manager.exit("feature", force=True)

            self.assertTrue(ok)
            self.assertIn("custom/branch", message)
            self.assertIn(("branch", "-D", "custom/branch"), calls)


if __name__ == "__main__":
    unittest.main()

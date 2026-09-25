import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, call

from tinyCode.teams.merger import GitMerger, MAX_CONFLICT_FILE_CHARS


class GitMergerValidationTests(unittest.TestCase):
    def test_rejects_markdown_wrapper(self):
        self.assertFalse(GitMerger._is_safe_resolution("```python\nprint('x')\n```"))

    def test_rejects_remaining_conflict_markers(self):
        self.assertFalse(
            GitMerger._is_safe_resolution("a\n<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> branch\n")
        )

    def test_accepts_plain_resolved_file(self):
        self.assertTrue(GitMerger._is_safe_resolution("def ok():\n    return True\n"))


class GitMergerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_clean_baseline_commit_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktree = Path(tmp) / "member"
            root.mkdir()
            for args in (
                ["git", "init", "-q", "-b", "main"],
                ["git", "config", "user.email", "tinycode@example.invalid"],
                ["git", "config", "user.name", "TinyCode Test"],
            ):
                subprocess.run(args, cwd=root, check=True)
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "base.txt"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "base"], cwd=root, check=True,
            )
            subprocess.run(
                ["git", "worktree", "add", "-q", "-b", "custom/member", str(worktree)],
                cwd=root,
                check=True,
            )
            merger = GitMerger(object(), root, allow_llm_conflicts=False)

            ok, baseline = await merger.inspect_worktree(worktree)
            self.assertTrue(ok, baseline)
            (worktree / "member.txt").write_text("done\n", encoding="utf-8")
            ok, message = await merger.prepare_worktree(worktree, "alice")
            self.assertTrue(ok, message)
            ok, message = await merger.merge(
                baseline["branch"], baseline["target_branch"],
            )

            self.assertTrue(ok, message)
            self.assertEqual(
                "done\n", (root / "member.txt").read_text(encoding="utf-8"),
            )

    async def test_conflict_resolution_is_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            merger = GitMerger(
                object(), Path(tmp), allow_llm_conflicts=False,
            )
            merger._git = AsyncMock(side_effect=[
                (0, "main\n", ""),
                (0, "", ""),
                (1, "", "conflict"),
                (0, "conflict.py\n", ""),
                (0, "", ""),
            ])
            merger._resolve_conflicts = AsyncMock()

            ok, message = await merger.merge("tinyCode/member")

            self.assertFalse(ok)
            self.assertIn("未启用", message)
            merger._resolve_conflicts.assert_not_awaited()
            self.assertIn(call("merge", "--abort"), merger._git.await_args_list)

    async def test_validation_commands_are_argv_safe_and_stop_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merger = GitMerger(object(), root)
            merger._run_command = AsyncMock(return_value=(1, "", "failed"))

            ok, message = await merger.validate_worktree(
                root, ['python3 -m unittest "tests/test one.py"'],
            )

            self.assertFalse(ok)
            self.assertIn("failed", message)
            merger._run_command.assert_awaited_once_with(
                root,
                ["python3", "-m", "unittest", "tests/test one.py"],
            )

    async def test_inspect_worktree_rejects_preexisting_dirty_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worker"
            worktree.mkdir()
            merger = GitMerger(object(), root)
            merger._git = AsyncMock(side_effect=[
                (0, str(worktree) + "\n", ""),
                (0, " M user-change.py\n", ""),
            ])

            ok, message = await merger.inspect_worktree(worktree)

            self.assertFalse(ok)
            self.assertIn("任务开始前", message)

    async def test_inspect_worktree_captures_actual_custom_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worker"
            worktree.mkdir()
            merger = GitMerger(object(), root)
            merger._git = AsyncMock(side_effect=[
                (0, str(worktree) + "\n", ""),
                (0, "", ""),
                (0, "custom/member\n", ""),
                (0, "abc123\n", ""),
                (0, "develop\n", ""),
                (0, "", ""),
            ])

            ok, baseline = await merger.inspect_worktree(worktree)

            self.assertTrue(ok)
            self.assertEqual("custom/member", baseline["branch"])
            self.assertEqual("develop", baseline["target_branch"])

    async def test_merge_reports_provider_failure_and_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            merger = GitMerger(object(), Path(tmp))
            merger._git = AsyncMock(side_effect=[
                (0, "main\n", ""),
                (0, "", ""),
                (1, "", "conflict"),
                (0, "conflict.py\n", ""),
                (0, "", ""),
            ])
            merger._resolve_conflicts = AsyncMock(
                side_effect=RuntimeError("provider down")
            )

            ok, message = await merger.merge("tinyCode/member")

            self.assertFalse(ok)
            self.assertIn("provider down", message)
            self.assertIn(call("merge", "--abort"), merger._git.await_args_list)

    async def test_conflict_resolution_rejects_oversized_file_before_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.py").write_text(
                "x" * (MAX_CONFLICT_FILE_CHARS + 1), encoding="utf-8",
            )
            provider = AsyncMock()
            merger = GitMerger(provider, root)

            with self.assertRaisesRegex(RuntimeError, "冲突文件过大"):
                await merger._resolve_conflicts(["large.py"])

            provider.chat_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()

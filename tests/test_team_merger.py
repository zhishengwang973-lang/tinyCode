import tempfile
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

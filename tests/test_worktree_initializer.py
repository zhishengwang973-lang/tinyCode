import tempfile
import unittest
from pathlib import Path

from tinyCode.worktree.initializer import WorktreeInitializer


class WorktreeInitializerTests(unittest.TestCase):
    def test_initialize_never_copies_credential_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            worktree_path = Path(tmp) / "worktree"
            repo_root.mkdir()
            worktree_path.mkdir()
            (repo_root / ".env.local").write_text("API_KEY=from-main\n", encoding="utf-8")
            (repo_root / "service.env").write_text("TOKEN=from-main\n", encoding="utf-8")
            (worktree_path / "service.env").write_text("TOKEN=existing\n", encoding="utf-8")

            logs = WorktreeInitializer(repo_root).initialize(worktree_path, symlink_dirs=[])

            self.assertFalse((worktree_path / ".env.local").exists())
            self.assertEqual(
                "TOKEN=existing\n",
                (worktree_path / "service.env").read_text(encoding="utf-8"),
            )
            self.assertNotIn("复制: .env.local", logs)
            self.assertNotIn("复制: service.env", logs)


if __name__ == "__main__":
    unittest.main()

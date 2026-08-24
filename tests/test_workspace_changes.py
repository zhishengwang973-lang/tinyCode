import tempfile
import unittest
from pathlib import Path

from tinyCode.tui.workspace_changes import WorkspaceSnapshot


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_reports_added_modified_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            changed = root / "src" / "changed.py"
            deleted = root / "old.txt"
            changed.parent.mkdir()
            changed.write_text("value = 1\n", encoding="utf-8")
            deleted.write_text("old\n", encoding="utf-8")
            snapshot = WorkspaceSnapshot.capture(root)

            changed.write_text("value = 2\n", encoding="utf-8")
            deleted.unlink()
            added = root / "docs" / "new.md"
            added.parent.mkdir()
            added.write_text("new\n", encoding="utf-8")

            changes = snapshot.compare()

            self.assertEqual(("docs/new.md",), changes.added)
            self.assertEqual(("src/changed.py",), changes.modified)
            self.assertEqual(("old.txt",), changes.deleted)

    def test_ignores_dependency_cache_and_internal_tool_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = WorkspaceSnapshot.capture(root)

            ignored = (
                root / ".git" / "index",
                root / "node_modules" / "pkg" / "index.js",
                root / "src" / "__pycache__" / "module.pyc",
                root / ".tinyCode" / "tool_results" / "large.txt",
                root / ".tinyCode" / "traces" / "task.jsonl",
            )
            for path in ignored:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated", encoding="utf-8")
            visible = root / ".tinyCode" / "notes" / "project.md"
            visible.parent.mkdir(parents=True)
            visible.write_text("note", encoding="utf-8")

            changes = snapshot.compare()

            self.assertEqual((".tinyCode/notes/project.md",), changes.added)

    def test_unchanged_workspace_has_no_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "same.txt").write_text("same\n", encoding="utf-8")

            changes = WorkspaceSnapshot.capture(root).compare()

            self.assertFalse(changes.any)

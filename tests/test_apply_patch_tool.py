import os
import tempfile
import unittest
from pathlib import Path

from tinyCode.tools.apply_patch import ApplyPatchTool, extract_patch_paths
from tinyCode.tools.context import use_workspace


class ApplyPatchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_update_and_delete_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app.py"
            old = root / "old.txt"
            app.write_text("def value():\n    return 1\n", encoding="utf-8")
            old.write_text("obsolete\n", encoding="utf-8")
            app.chmod(0o755)
            patch = """*** Begin Patch
*** Update File: app.py
@@
 def value():
-    return 1
+    return 2
*** Add File: docs/readme.md
+# Added
+
*** Delete File: old.txt
-obsolete
*** End Patch"""

            with use_workspace(root):
                result = await ApplyPatchTool().execute(patch)

            self.assertTrue(result.success, result.error)
            self.assertEqual("def value():\n    return 2\n", app.read_text(encoding="utf-8"))
            self.assertEqual("# Added\n\n", (root / "docs/readme.md").read_text())
            self.assertFalse(old.exists())
            self.assertEqual(0o755, app.stat().st_mode & 0o777)
            self.assertIn("新增 1", result.content)
            self.assertIn("修改 1", result.content)
            self.assertIn("删除 1", result.content)

    async def test_validation_failure_keeps_every_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app.py"
            app.write_text("value = 1\n", encoding="utf-8")
            patch = """*** Begin Patch
*** Add File: created.txt
+created
*** Update File: app.py
@@
-missing = 1
+missing = 2
*** End Patch"""

            with use_workspace(root):
                result = await ApplyPatchTool().execute(patch)

            self.assertFalse(result.success)
            self.assertIn("上下文未匹配", result.error)
            self.assertFalse((root / "created.txt").exists())
            self.assertEqual("value = 1\n", app.read_text(encoding="utf-8"))

    async def test_rejects_ambiguous_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "values.txt"
            target.write_text("same\nsame\n", encoding="utf-8")
            patch = """*** Begin Patch
*** Update File: values.txt
@@
-same
+changed
*** End Patch"""

            with use_workspace(root):
                result = await ApplyPatchTool().execute(patch)

            self.assertFalse(result.success)
            self.assertIn("不唯一", result.error)
            self.assertEqual("same\nsame\n", target.read_text())

    async def test_rejects_traversal_and_sensitive_paths(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "project"
            root.mkdir()
            outside = Path(parent) / "outside.txt"
            for path in ("../outside.txt", ".env"):
                with self.subTest(path=path):
                    patch = (
                        "*** Begin Patch\n"
                        f"*** Add File: {path}\n"
                        "+secret\n"
                        "*** End Patch"
                    )
                    with use_workspace(root):
                        result = await ApplyPatchTool().execute(patch)
                    self.assertFalse(result.success)
            self.assertFalse(outside.exists())
            self.assertFalse((root / ".env").exists())

    async def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "project"
            outside = Path(parent) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "link").symlink_to(outside, target_is_directory=True)
            patch = """*** Begin Patch
*** Add File: link/escaped.txt
+no
*** End Patch"""

            with use_workspace(root):
                result = await ApplyPatchTool().execute(patch)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertFalse((outside / "escaped.txt").exists())

    def test_extract_patch_paths_returns_all_declared_targets(self):
        paths = extract_patch_paths(
            "*** Begin Patch\n"
            "*** Update File: a.py\n"
            "*** Add File: docs/b.md\n"
            "*** End Patch"
        )

        self.assertEqual(["a.py", "docs/b.md"], paths)


if __name__ == "__main__":
    unittest.main()

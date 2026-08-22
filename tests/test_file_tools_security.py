import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tinyCode.tools.delete_file import DeleteFileTool
from tinyCode.tools.edit_file import EditFileTool
from tinyCode.tools.glob import GlobTool
from tinyCode.tools.grep import GrepTool
from tinyCode.tools.read_file import ReadFileTool
from tinyCode.tools.write_file import WriteFileTool
from tinyCode.tools.context import use_workspace


def run_async(coro):
    return asyncio.run(coro)


class FileToolPathSecurityTests(unittest.TestCase):
    def test_delete_file_removes_single_regular_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            target = Path(tmp) / "old.txt"
            target.write_text("unused", encoding="utf-8")

            try:
                os.chdir(tmp)
                result = run_async(DeleteFileTool().execute("old.txt"))
            finally:
                os.chdir(original_cwd)

            self.assertTrue(result.success)
            self.assertFalse(target.exists())

    def test_delete_file_rejects_parent_directory_traversal(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(DeleteFileTool().execute("../outside.txt"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertEqual("keep", outside.read_text(encoding="utf-8"))

    def test_delete_file_rejects_directory_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            directory = Path(tmp) / "src"
            directory.mkdir()

            try:
                os.chdir(tmp)
                result = run_async(DeleteFileTool().execute("src"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径不是普通文件", result.error)
            self.assertTrue(directory.exists())

    def test_delete_file_rejects_symlink_that_escapes_project(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "secret.txt"
            outside.write_text("keep", encoding="utf-8")
            link = project / "secret-link.txt"
            link.symlink_to(outside)
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(DeleteFileTool().execute("secret-link.txt"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertTrue(link.exists())
            self.assertEqual("keep", outside.read_text(encoding="utf-8"))

    def test_delete_file_rejects_non_string_paths_as_structured_failure(self):
        result = run_async(DeleteFileTool().execute(123))

        self.assertFalse(result.success)
        self.assertIn("path 必须是字符串", result.error)

    def test_write_file_rejects_parent_directory_traversal(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "outside.txt"
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(
                    WriteFileTool().execute("../outside.txt", "should not escape")
                )
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertFalse(outside.exists())

    def test_edit_file_rejects_parent_directory_traversal(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "outside.txt"
            outside.write_text("old", encoding="utf-8")
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(
                    EditFileTool().execute("../outside.txt", "old", "new")
                )
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertEqual("old", outside.read_text(encoding="utf-8"))

    def test_glob_rejects_parent_directory_traversal(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(GlobTool().execute("../*.txt"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertNotIn("outside.txt", result.content)

    def test_read_file_rejects_symlink_that_escapes_project(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            link = project / "secret-link.txt"
            link.symlink_to(outside)
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(ReadFileTool().execute("secret-link.txt"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("路径遍历", result.error)
            self.assertNotIn("secret", result.content)

    def test_read_file_rejects_local_provider_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            config = Path(tmp) / ".tinyCode.yaml"
            config.write_text("api_key: should-not-leak", encoding="utf-8")
            try:
                os.chdir(tmp)
                result = run_async(ReadFileTool().execute(".tinyCode.yaml"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("凭据", result.error)
            self.assertNotIn("should-not-leak", result.content)

    def test_write_edit_and_delete_reject_provider_or_environment_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            env_file = Path(tmp) / ".env"
            env_file.write_text("TOKEN=secret", encoding="utf-8")
            try:
                os.chdir(tmp)
                write_result = run_async(
                    WriteFileTool().execute(".tinyCode.yaml", "api_key: secret")
                )
                edit_result = run_async(
                    EditFileTool().execute(".env", "secret", "changed")
                )
                delete_result = run_async(DeleteFileTool().execute(".env"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(write_result.success)
            self.assertFalse(edit_result.success)
            self.assertFalse(delete_result.success)
            self.assertEqual("TOKEN=secret", env_file.read_text(encoding="utf-8"))

    def test_grep_skips_symlink_that_escapes_project(self):
        with tempfile.TemporaryDirectory() as parent:
            project = Path(parent) / "project"
            project.mkdir()
            outside = Path(parent) / "secret.txt"
            outside.write_text("needle-secret\n", encoding="utf-8")
            link = project / "secret-link.txt"
            link.symlink_to(outside)
            original_cwd = Path.cwd()

            try:
                os.chdir(project)
                result = run_async(GrepTool().execute("needle-secret"))
            finally:
                os.chdir(original_cwd)

            self.assertTrue(result.success)
            self.assertEqual("(无匹配)", result.content)

    def test_file_tools_reject_non_string_paths_as_structured_failures(self):
        read_result = run_async(ReadFileTool().execute(123))
        write_result = run_async(WriteFileTool().execute(123, "content"))
        edit_result = run_async(EditFileTool().execute(123, "old", "new"))

        self.assertFalse(read_result.success)
        self.assertFalse(write_result.success)
        self.assertFalse(edit_result.success)
        self.assertIn("path 必须是字符串", read_result.error)
        self.assertIn("path 必须是字符串", write_result.error)
        self.assertIn("path 必须是字符串", edit_result.error)

    def test_write_file_rejects_non_string_content_as_structured_failure(self):
        result = run_async(WriteFileTool().execute("new.txt", 123))

        self.assertFalse(result.success)
        self.assertIn("content 必须是字符串", result.error)

    def test_edit_file_rejects_non_string_replacement_values_as_structured_failures(self):
        old_result = run_async(EditFileTool().execute("file.txt", 123, "new"))
        new_result = run_async(EditFileTool().execute("file.txt", "old", 123))

        self.assertFalse(old_result.success)
        self.assertFalse(new_result.success)
        self.assertIn("old_string 必须是字符串", old_result.error)
        self.assertIn("new_string 必须是字符串", new_result.error)

    def test_edit_file_rejects_empty_old_string_without_inserting_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            target = Path(tmp) / "file.txt"
            target.write_text("", encoding="utf-8")

            try:
                os.chdir(tmp)
                result = run_async(EditFileTool().execute("file.txt", "", "inserted"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("old_string 不能为空", result.error)
            self.assertEqual("", target.read_text(encoding="utf-8"))

    def test_edit_file_reports_read_errors_as_structured_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            target = Path(tmp) / "binary.txt"
            target.write_bytes(b"\xff\xfe\xfd")

            try:
                os.chdir(tmp)
                result = run_async(EditFileTool().execute("binary.txt", "old", "new"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("读取文件失败", result.error)

    def test_edit_file_keeps_original_content_when_atomic_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            target = Path(tmp) / "file.txt"
            target.write_text("old value", encoding="utf-8")

            try:
                os.chdir(tmp)
                with patch("pathlib.Path.replace", side_effect=OSError("disk full")):
                    result = run_async(EditFileTool().execute("file.txt", "old", "new"))
            finally:
                os.chdir(original_cwd)

            self.assertFalse(result.success)
            self.assertIn("写入文件失败", result.error)
            self.assertEqual("old value", target.read_text(encoding="utf-8"))

    def test_edit_file_preserves_executable_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            target = Path(tmp) / "script.sh"
            target.write_text("echo old\n", encoding="utf-8")
            target.chmod(0o755)

            try:
                os.chdir(tmp)
                result = run_async(
                    EditFileTool().execute("script.sh", "old", "new")
                )
            finally:
                os.chdir(original_cwd)

            self.assertTrue(result.success)
            self.assertEqual(0o755, target.stat().st_mode & 0o777)


class TaskLocalWorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_tools_use_their_own_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            (first / "same.txt").write_text("first", encoding="utf-8")
            (second / "same.txt").write_text("second", encoding="utf-8")
            tool = ReadFileTool()

            async def read_from(root: Path):
                with use_workspace(root):
                    await asyncio.sleep(0)
                    return await tool.execute("same.txt")

            first_result, second_result = await asyncio.gather(
                read_from(first), read_from(second),
            )

            self.assertEqual("first", first_result.content)
            self.assertEqual("second", second_result.content)

    async def test_read_file_paginates_large_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("a" * 250_000, encoding="utf-8")

            with use_workspace(root):
                first = await ReadFileTool().execute("large.txt")
                second = await ReadFileTool().execute(
                    "large.txt", offset=200_000, limit=50_000,
                )

            self.assertTrue(first.success)
            self.assertLess(len(first.content), 201_000)
            self.assertIn("内容已分页", first.content)
            self.assertEqual("a" * 50_000, second.content)


if __name__ == "__main__":
    unittest.main()

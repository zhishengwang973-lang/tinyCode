import os
import tempfile
import unittest
from pathlib import Path

from tinyCode.tools.glob import GlobTool
from tinyCode.tools.grep import GrepTool


class SearchToolLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_grep_rejects_oversized_regex_before_scanning(self):
        result = await GrepTool().execute("x" * 1_001)

        self.assertFalse(result.success)
        self.assertIn("最大支持 1000", result.error)

    async def test_grep_rejects_nested_repetition_regex(self):
        result = await GrepTool().execute("(a+)+$")

        self.assertFalse(result.success)
        self.assertIn("灾难性回溯", result.error)

    async def test_glob_returns_at_most_fifty_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                for index in range(60):
                    Path(f"file_{index:02d}.txt").write_text("x", encoding="utf-8")

                result = await GlobTool().execute("*.txt")
            finally:
                os.chdir(original_cwd)

        self.assertTrue(result.success)
        returned_paths = [line for line in result.content.splitlines() if line.endswith(".txt")]
        self.assertEqual(50, len(returned_paths))
        self.assertIn("已截断到前 50 条", result.content)

    async def test_grep_returns_at_most_fifty_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                for index in range(60):
                    Path(f"file_{index:02d}.txt").write_text("needle\n", encoding="utf-8")

                result = await GrepTool().execute("needle")
            finally:
                os.chdir(original_cwd)

        self.assertTrue(result.success)
        returned_matches = [
            line for line in result.content.splitlines()
            if ".txt:" in line and "needle" in line
        ]
        self.assertEqual(50, len(returned_matches))
        self.assertIn("已截断到前 50 条", result.content)

    async def test_search_tools_reject_non_string_patterns_as_structured_failures(self):
        glob_result = await GlobTool().execute(123)
        grep_result = await GrepTool().execute(123)

        self.assertFalse(glob_result.success)
        self.assertFalse(grep_result.success)
        self.assertIn("pattern 必须是字符串", glob_result.error)
        self.assertIn("pattern 必须是字符串", grep_result.error)

    async def test_glob_hides_local_provider_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                Path("app.py").write_text("pass\n", encoding="utf-8")
                Path(".tinyCode.yaml").write_text(
                    "api_key: should-not-leak\n", encoding="utf-8",
                )
                result = await GlobTool().execute("*")
            finally:
                os.chdir(original_cwd)

        self.assertTrue(result.success)
        self.assertIn("app.py", result.content)
        self.assertNotIn(".tinyCode.yaml", result.content)


if __name__ == "__main__":
    unittest.main()

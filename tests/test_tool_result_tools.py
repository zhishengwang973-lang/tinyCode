import asyncio
import tempfile
import unittest
from pathlib import Path

from tinyCode.main import _create_tool_registry
from tinyCode.tools.tool_result_read import ToolResultReadTool
from tinyCode.tools.tool_result_search import ToolResultSearchTool


def run_async(coro):
    return asyncio.run(coro)


class ToolResultToolsTests(unittest.TestCase):
    def test_search_finds_matching_lines_in_stored_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_file = Path(tmp) / "large_result.txt"
            result_file.write_text(
                "alpha\nsession id: abc\nbeta\nsession state: active\n",
                encoding="utf-8",
            )

            result = run_async(
                ToolResultSearchTool(storage_dir=Path(tmp)).execute(
                    str(result_file), "session"
                )
            )

            self.assertTrue(result.success)
            self.assertIn("large_result.txt:2: session id: abc", result.content)
            self.assertIn("large_result.txt:4: session state: active", result.content)

    def test_read_returns_only_requested_line_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_file = Path(tmp) / "large_result.txt"
            result_file.write_text(
                "\n".join(f"line {index}" for index in range(1, 8)),
                encoding="utf-8",
            )

            result = run_async(
                ToolResultReadTool(storage_dir=Path(tmp)).execute(
                    str(result_file), start_line=3, limit=2
                )
            )

            self.assertTrue(result.success)
            self.assertIn("large_result.txt:3: line 3", result.content)
            self.assertIn("large_result.txt:4: line 4", result.content)
            self.assertNotIn("line 2", result.content)
            self.assertNotIn("line 5", result.content)

    def test_tools_reject_files_outside_tool_result_storage(self):
        with tempfile.TemporaryDirectory() as parent:
            storage = Path(parent) / "tool_results"
            storage.mkdir()
            outside = Path(parent) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")

            search_result = run_async(
                ToolResultSearchTool(storage_dir=storage).execute(str(outside), "secret")
            )
            read_result = run_async(
                ToolResultReadTool(storage_dir=storage).execute(str(outside))
            )

            self.assertFalse(search_result.success)
            self.assertIn("tool_results", search_result.error)
            self.assertFalse(read_result.success)
            self.assertIn("tool_results", read_result.error)

    def test_default_registry_exposes_tool_result_retrieval_tools(self):
        registry = _create_tool_registry()

        self.assertIsNotNone(registry.get("tool_result_search"))
        self.assertIsNotNone(registry.get("tool_result_read"))

    def test_registry_binds_both_helpers_to_truncator_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "project" / ".tinyCode" / "tool_results"
            registry = _create_tool_registry(storage)

            self.assertEqual(storage, registry.get("tool_result_search").storage_dir)
            self.assertEqual(storage, registry.get("tool_result_read").storage_dir)


if __name__ == "__main__":
    unittest.main()

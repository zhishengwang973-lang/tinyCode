import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tinyCode.conversation.truncator import (
    ToolResultTruncator,
    TruncateConfig,
    default_storage_dir,
)


class ToolResultTruncatorTests(unittest.TestCase):
    def test_default_limits_match_public_context_contract(self):
        config = TruncateConfig()

        self.assertEqual(16_000, config.per_result_threshold)
        self.assertEqual(64_000, config.total_round_threshold)
        self.assertEqual(2_000, config.preview_length)

    def test_default_storage_is_project_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)

            storage = default_storage_dir(project)

            self.assertEqual(
                project.resolve() / ".tinyCode" / "tool_results",
                storage,
            )

    def test_anthropic_tool_result_blocks_are_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = "x" * 20
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": content,
                        }
                    ],
                }
            ]
            truncator = ToolResultTruncator(
                TruncateConfig(
                    per_result_threshold=10,
                    total_round_threshold=100,
                    preview_length=5,
                    storage_dir=Path(tmp),
                )
            )

            new_messages, infos = truncator.process_round(messages)

            block = new_messages[0]["content"][0]
            self.assertEqual("tool_result", block["type"])
            self.assertIn("完整内容已保存到磁盘", block["content"])
            self.assertIn("tool_result_search", block["content"])
            self.assertIn("tool_result_read", block["content"])
            self.assertIn("xxxxx", block["content"])
            self.assertEqual(1, len(infos))
            self.assertEqual("toolu_1", infos[0]["tool_name"])
            self.assertEqual(20, infos[0]["original_chars"])
            self.assertEqual(content, Path(infos[0]["file_path"]).read_text(encoding="utf-8"))

    def test_truncated_results_from_same_tool_use_distinct_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = "a" * 20
            second = "b" * 20
            messages = [
                {"role": "tool", "name": "grep", "content": first},
                {"role": "tool", "name": "grep", "content": second},
            ]
            truncator = ToolResultTruncator(
                TruncateConfig(
                    per_result_threshold=10,
                    total_round_threshold=100,
                    preview_length=5,
                    storage_dir=Path(tmp),
                )
            )

            _, infos = truncator.process_round(messages)

            paths = [info["file_path"] for info in infos]
            self.assertEqual(2, len(paths))
            self.assertEqual(2, len(set(paths)))
            self.assertEqual(first, Path(paths[0]).read_text(encoding="utf-8"))
            self.assertEqual(second, Path(paths[1]).read_text(encoding="utf-8"))

    def test_same_result_is_not_written_again_on_every_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            messages = [{"role": "tool", "name": "grep", "content": "x" * 20}]
            truncator = ToolResultTruncator(TruncateConfig(
                per_result_threshold=10,
                total_round_threshold=100,
                preview_length=5,
                storage_dir=Path(tmp),
            ))

            _, first_infos = truncator.process_round(messages)
            _, second_infos = truncator.process_round(messages)

            self.assertEqual(first_infos[0]["file_path"], second_infos[0]["file_path"])
            self.assertEqual(1, len(list(Path(tmp).glob("*.txt"))))

    def test_cached_result_remains_available_without_new_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TruncateConfig(
                per_result_threshold=10,
                total_round_threshold=100,
                preview_length=5,
                storage_dir=Path(tmp),
            )
            truncator = ToolResultTruncator(config)
            _, first_infos = truncator.process_round([
                {"role": "tool", "name": "grep", "content": "x" * 20},
            ])

            _, later_infos = truncator.process_round([
                {"role": "user", "content": "continue"},
            ])

            self.assertEqual([], later_infos)
            self.assertTrue(truncator.has_available_results)

            reopened = ToolResultTruncator(TruncateConfig(
                per_result_threshold=10,
                total_round_threshold=100,
                preview_length=5,
                storage_dir=Path(tmp),
            ))
            self.assertTrue(reopened.has_available_results)

            Path(first_infos[0]["file_path"]).unlink()
            self.assertFalse(truncator.has_available_results)
            self.assertFalse(reopened.has_available_results)

    def test_total_budget_uses_preview_size_when_selecting_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            messages = [
                {"role": "tool", "name": f"read-{index}", "content": char * 10_000}
                for index, char in enumerate(("a", "b", "c"), start=1)
            ]
            truncator = ToolResultTruncator(TruncateConfig(
                per_result_threshold=50_000,
                total_round_threshold=15_000,
                preview_length=100,
                storage_dir=Path(tmp),
            ))

            new_messages, infos = truncator.process_round(messages)

            visible_total = sum(len(message["content"]) for message in new_messages)
            self.assertEqual(2, len(infos))
            self.assertLessEqual(visible_total, 15_000)
            self.assertEqual(3, len(new_messages))

    def test_unwritable_storage_falls_back_to_preview_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TruncateConfig(
                per_result_threshold=10,
                total_round_threshold=100,
                preview_length=5,
                storage_dir=Path(tmp) / "blocked",
            )
            with patch("pathlib.Path.mkdir", side_effect=PermissionError("read only")):
                truncator = ToolResultTruncator(config)

            messages = [{"role": "tool", "name": "grep", "content": "x" * 20}]
            new_messages, infos = truncator.process_round(messages)

            self.assertIn("完整内容未保存", new_messages[0]["content"])
            self.assertIn("read only", truncator.storage_error)
            self.assertEqual("", infos[0]["file_path"])


if __name__ == "__main__":
    unittest.main()

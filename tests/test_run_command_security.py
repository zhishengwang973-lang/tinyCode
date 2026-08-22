import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tinyCode.conversation.truncator import ToolResultTruncator, TruncateConfig
from tinyCode.tools.context import use_workspace
from tinyCode.tools.run_command import OUTPUT_LIMIT, RunCommandTool


class RunCommandSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_command_does_not_inherit_credential_environment(self):
        command = (
            f'{sys.executable} -c "import os; '
            'print(os.getenv(\'TINYCODE_TEST_SECRET\', \'missing\'))"'
        )
        with patch.dict(os.environ, {"TINYCODE_TEST_SECRET": "must-not-leak"}):
            result = await RunCommandTool().execute(command)

        self.assertTrue(result.success)
        self.assertIn("missing", result.content)
        self.assertNotIn("must-not-leak", result.content)

    async def test_run_command_cannot_read_provider_config_or_dump_environment(self):
        for command in ("cat .tinyCode.yaml", "env", "printenv HOME"):
            with self.subTest(command=command):
                result = await RunCommandTool().execute(command)
                self.assertFalse(result.success)
                self.assertIn("凭据", result.error)

    async def test_run_command_reuses_central_blacklist_for_eval(self):
        result = await RunCommandTool().execute("eval echo unsafe")

        self.assertFalse(result.success)
        self.assertIn("黑名单", result.error)

    async def test_destructive_command_aliases_and_shell_wrappers_are_blocked(self):
        for command in (
            "/bin/rm -rf ./important",
            "sh -c 'rm -rf ./important'",
            "git push -f origin main",
        ):
            with self.subTest(command=command):
                result = await RunCommandTool().execute(command)
                self.assertFalse(result.success)
                self.assertIn("黑名单", result.error)

    async def test_run_command_rejects_non_string_command_as_structured_failure(self):
        result = await RunCommandTool().execute(123)

        self.assertFalse(result.success)
        self.assertIn("command 必须是字符串", result.error)

    async def test_run_command_rejects_empty_command(self):
        result = await RunCommandTool().execute("   ")

        self.assertFalse(result.success)
        self.assertIn("command 不能为空", result.error)

    async def test_large_output_is_drained_but_memory_result_is_bounded(self):
        command = (
            f'{sys.executable} -c "import sys; '
            f'sys.stdout.write(\'x\'*{OUTPUT_LIMIT * 2})"'
        )

        result = await RunCommandTool().execute(command)

        self.assertTrue(result.success)
        self.assertLess(len(result.content), OUTPUT_LIMIT + 1_000)
        self.assertIn("输出已截断", result.content)

    async def test_large_command_result_is_persisted_in_project_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            storage = project / ".tinyCode" / "tool_results"
            truncator = ToolResultTruncator(TruncateConfig(
                storage_dir=storage,
            ))
            command = (
                f'{sys.executable} -c "import sys; '
                'sys.stdout.write(\'x\'*30000 + '
                '\'TINYCODE_DYNAMIC_RESULT_OK\' + \'y\'*30000)"'
            )

            with use_workspace(project):
                result = await RunCommandTool().execute(command)
            messages, infos = truncator.process_round([{
                "role": "tool",
                "name": "run_command",
                "content": result.to_message(),
            }])

            self.assertTrue(result.success, result.error)
            self.assertEqual(1, len(infos))
            stored = Path(infos[0]["file_path"])
            self.assertEqual(storage.resolve(), stored.parent)
            self.assertIn(
                "TINYCODE_DYNAMIC_RESULT_OK",
                stored.read_text(encoding="utf-8"),
            )
            self.assertIn("tool_result_search", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
import os
from unittest.mock import patch

from tinyCode.tools.run_command import RunCommandTool


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
        command = f'{sys.executable} -c "import sys; sys.stdout.write(\'x\'*200000)"'

        result = await RunCommandTool().execute(command)

        self.assertTrue(result.success)
        self.assertLess(len(result.content), 11_000)
        self.assertIn("输出已截断", result.content)


if __name__ == "__main__":
    unittest.main()

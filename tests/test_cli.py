import unittest

from tinyCode.cli import CLIOptions, parse_cli_args
from tinyCode.main import _CleanupStack


class CLIParserTests(unittest.TestCase):
    def test_parses_all_runtime_flags(self):
        options = parse_cli_args([
            "--mode", "strict", "--resume", "--trust-project-config",
        ])

        self.assertEqual(
            CLIOptions(
                mode="strict", resume=True, trust_project_config=True,
            ),
            options,
        )

    def test_missing_mode_value_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_cli_args(["--mode"])

    def test_unknown_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_cli_args(["--unknown"])


class CleanupStackTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_runs_in_reverse_and_continues_after_failure(self):
        calls = []
        cleanup = _CleanupStack()

        async def first():
            calls.append("first")

        async def second():
            calls.append("second")
            raise RuntimeError("cleanup failed")

        async def third():
            calls.append("third")

        cleanup.add("first", first)
        cleanup.add("second", second)
        cleanup.add("third", third)

        await cleanup.close()

        self.assertEqual(["third", "second", "first"], calls)


if __name__ == "__main__":
    unittest.main()

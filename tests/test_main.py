import io
import asyncio
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from tinyCode.cli import CLIOptions
from tinyCode.config.loader import ConfigError
from tinyCode.main import _CleanupStack, _run_application, main


class MainStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_config_returns_nonzero_exit_status(self):
        stderr = io.StringIO()
        with patch("tinyCode.main.load_config", side_effect=ConfigError("配置损坏")):
            with redirect_stderr(stderr):
                code = await _run_application(CLIOptions(), _CleanupStack())

        self.assertEqual(2, code)
        self.assertIn("配置损坏", stderr.getvalue())

    async def test_unexpected_top_level_failure_returns_readable_nonzero_status(self):
        stderr = io.StringIO()
        with patch(
            "tinyCode.main._run_application",
            side_effect=RuntimeError("optional component failed"),
        ):
            with redirect_stderr(stderr):
                code = await main(CLIOptions())

        self.assertEqual(1, code)
        self.assertIn("TinyCode 未能继续运行", stderr.getvalue())
        self.assertIn("optional component failed", stderr.getvalue())


class CleanupStackTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_cleanup_does_not_skip_remaining_resources(self):
        stack = _CleanupStack()
        calls: list[str] = []

        async def first() -> None:
            calls.append("first")

        async def cancelled() -> None:
            calls.append("cancelled")
            raise asyncio.CancelledError

        async def last() -> None:
            calls.append("last")

        stack.add("first", first)
        stack.add("cancelled", cancelled)
        stack.add("last", last)

        with self.assertRaises(asyncio.CancelledError):
            await stack.close()

        self.assertEqual(["last", "cancelled", "first"], calls)
        self.assertEqual([], stack._steps)


if __name__ == "__main__":
    unittest.main()

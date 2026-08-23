import unittest

from tinyCode.agent.progress import ProgressState, ProgressWatchdog
from tinyCode.providers.base import ToolCall
from tinyCode.tools.base import ToolResult


def observation(
    round_number: int,
    *,
    name: str = "read_file",
    params: dict | None = None,
    success: bool = True,
    content: str = "same result",
    error: str = "",
):
    call = ToolCall(
        id=f"call-{round_number}",
        name=name,
        input=params or {"file": "example.py"},
    )
    result = ToolResult(success=success, content=content, error=error)
    return [(call, result)]


class ProgressWatchdogTests(unittest.TestCase):
    def test_warns_then_intervenes_on_exact_repetition(self):
        watchdog = ProgressWatchdog()

        first = watchdog.observe(1, observation(1))
        second = watchdog.observe(2, observation(2))
        third = watchdog.observe(3, observation(3))

        self.assertEqual(ProgressState.PROGRESSING, first.state)
        self.assertEqual(ProgressState.SLOW, second.state)
        self.assertEqual(ProgressState.STALLED, third.state)
        self.assertTrue(third.requires_intervention)

    def test_same_error_three_times_is_hard_stuck(self):
        watchdog = ProgressWatchdog()

        for number in (1, 2):
            watchdog.observe(number, observation(
                number,
                success=False,
                content="",
                error="permission denied",
            ))
        result = watchdog.observe(3, observation(
            3,
            success=False,
            content="",
            error="permission denied",
        ))

        self.assertEqual(ProgressState.HARD_STUCK, result.state)
        self.assertIn("相同工具错误", result.reasons[0])

    def test_two_timeouts_are_hard_stuck(self):
        watchdog = ProgressWatchdog()
        watchdog.observe(1, observation(
            1, success=False, content="", error="command timed out",
        ))

        result = watchdog.observe(2, observation(
            2, success=False, content="", error="工具执行超时",
        ))

        self.assertEqual(ProgressState.HARD_STUCK, result.state)

    def test_detects_abab_oscillation(self):
        watchdog = ProgressWatchdog()
        states = [
            ("a.py", "A"),
            ("b.py", "B"),
            ("a.py", "A"),
            ("b.py", "B"),
        ]

        result = None
        for number, (path, content) in enumerate(states, 1):
            result = watchdog.observe(number, observation(
                number,
                params={"file": path},
                content=content,
            ))

        assert result is not None
        self.assertEqual(ProgressState.OSCILLATING, result.state)

    def test_successful_workspace_change_counts_as_progress(self):
        watchdog = ProgressWatchdog()

        for number in range(1, 7):
            result = watchdog.observe(number, observation(
                number,
                name="apply_patch",
                params={"patch": f"patch {number}"},
                content="Done!",
            ))

        self.assertEqual(ProgressState.PROGRESSING, result.state)
        self.assertTrue(watchdog.history[-1].workspace_changed)

    def test_repeating_same_successful_write_is_not_endless_progress(self):
        watchdog = ProgressWatchdog()

        for number in range(1, 5):
            result = watchdog.observe(number, observation(
                number,
                name="write_file",
                params={"file": "same.py", "content": "same"},
                content="written",
            ))

        self.assertEqual(ProgressState.STALLED, result.state)

    def test_decreasing_test_failures_count_as_progress(self):
        watchdog = ProgressWatchdog()
        watchdog.observe(1, observation(
            1,
            name="run_command",
            params={"command": "pytest -q"},
            success=False,
            content="3 failed, 8 passed",
            error="tests failed",
        ))

        result = watchdog.observe(2, observation(
            2,
            name="run_command",
            params={"command": "pytest -q"},
            success=False,
            content="1 failed, 10 passed",
            error="tests failed",
        ))

        self.assertEqual(ProgressState.PROGRESSING, result.state)
        self.assertTrue(watchdog.history[-1].test_improved)

    def test_reset_strategy_starts_a_fresh_window(self):
        watchdog = ProgressWatchdog()
        watchdog.observe(1, observation(1))
        watchdog.observe(2, observation(2))

        watchdog.reset_strategy()
        result = watchdog.observe(3, observation(3))

        self.assertEqual(ProgressState.PROGRESSING, result.state)
        self.assertEqual(1, len(watchdog.history))


if __name__ == "__main__":
    unittest.main()

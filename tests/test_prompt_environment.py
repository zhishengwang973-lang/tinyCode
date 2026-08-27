import unittest
from datetime import timedelta

from tinyCode.prompts.environment import (
    BEIJING_TIMEZONE, collect_current_time, collect_environment,
)


class PromptEnvironmentTests(unittest.TestCase):
    def test_environment_is_stable_and_time_uses_beijing_timezone(self):
        environment = collect_environment()
        current_time = collect_current_time()

        self.assertEqual(timedelta(hours=8), BEIJING_TIMEZONE.utcoffset(None))
        self.assertNotIn("当前时间", environment)
        self.assertIn("北京时间 (UTC+8)", current_time)
        self.assertNotIn(" UTC\n", current_time)


if __name__ == "__main__":
    unittest.main()

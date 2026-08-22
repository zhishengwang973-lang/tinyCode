import unittest
from datetime import timedelta

from tinyCode.prompts.environment import BEIJING_TIMEZONE, collect_environment


class PromptEnvironmentTests(unittest.TestCase):
    def test_environment_uses_beijing_time(self):
        environment = collect_environment()

        self.assertEqual(timedelta(hours=8), BEIJING_TIMEZONE.utcoffset(None))
        self.assertIn("北京时间 (UTC+8)", environment)
        self.assertNotIn(" UTC\n", environment)


if __name__ == "__main__":
    unittest.main()

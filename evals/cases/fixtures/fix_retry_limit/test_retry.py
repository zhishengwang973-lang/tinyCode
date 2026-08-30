import unittest
from retry import should_retry


class RetryTests(unittest.TestCase):
    def test_limit_is_exclusive(self):
        self.assertTrue(should_retry(0, 2))
        self.assertTrue(should_retry(1, 2))
        self.assertFalse(should_retry(2, 2))

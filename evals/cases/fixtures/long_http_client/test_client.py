import unittest
from client import HttpClient, RetryableError


class ClientTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        calls = []
        def sender(url):
            calls.append(url)
            if len(calls) < 3:
                raise RetryableError("temporary")
            return "ok"
        self.assertEqual("ok", HttpClient(sender, retries=2).request("/health"))
        self.assertEqual(3, len(calls))

    def test_raises_after_limit(self):
        def sender(url):
            raise RetryableError("temporary")
        with self.assertRaises(RetryableError):
            HttpClient(sender, retries=1).request("/health")

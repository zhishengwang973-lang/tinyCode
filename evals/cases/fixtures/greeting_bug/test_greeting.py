import unittest

from greeting import greet


class GreetingTests(unittest.TestCase):
    def test_greet_uses_chinese(self):
        self.assertEqual("你好，小明", greet("小明"))


if __name__ == "__main__":
    unittest.main()

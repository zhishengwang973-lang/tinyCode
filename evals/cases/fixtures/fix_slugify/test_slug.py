import unittest
from slug import slugify


class SlugTests(unittest.TestCase):
    def test_normalizes_whitespace(self):
        self.assertEqual("hello-world", slugify("  Hello   World  "))

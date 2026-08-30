import unittest
from formatter import format_name


class FormatterTests(unittest.TestCase):
    def test_words_are_normalized(self):
        self.assertEqual("Ada Lovelace", format_name("  ada lovelace  "))

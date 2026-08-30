import unittest
from parse import parse_int


class ParseTests(unittest.TestCase):
    def test_strips_input(self):
        self.assertEqual(42, parse_int(" 42 "))

    def test_blank_input_has_clear_error(self):
        with self.assertRaises(ValueError):
            parse_int(" ")

import unittest
from calc import add


class CalcTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(7, add(3, 4))

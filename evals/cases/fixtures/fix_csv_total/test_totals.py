import unittest
from totals import total_amount


class TotalTests(unittest.TestCase):
    def test_ignores_empty_values(self):
        self.assertEqual(12, total_amount("a,5\n\nb,\nc,7\n"))

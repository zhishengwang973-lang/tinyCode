import unittest
from report import render_report


class ReportTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual("暂无数据", render_report([]))

    def test_table_is_sorted(self):
        output = render_report([{"name": "Bob", "score": 2}, {"name": "Ada", "score": 5}])
        self.assertIn("| 名称 | 分数 |", output)
        self.assertLess(output.index("Ada"), output.index("Bob"))

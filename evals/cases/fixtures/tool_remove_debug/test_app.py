import io
import unittest
from contextlib import redirect_stdout
from app import double


class AppTests(unittest.TestCase):
    def test_no_debug_output(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(6, double(3))
        self.assertEqual("", stream.getvalue())

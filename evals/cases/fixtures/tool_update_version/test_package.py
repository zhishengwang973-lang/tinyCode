import unittest
import package


class PackageTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual("0.2.0", package.VERSION)
        self.assertEqual("demo", package.NAME)

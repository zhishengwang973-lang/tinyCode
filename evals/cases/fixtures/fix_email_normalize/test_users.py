import unittest
from users import normalize_email


class UserTests(unittest.TestCase):
    def test_only_domain_is_lowercase(self):
        self.assertEqual("Ada@example.com", normalize_email("  Ada@EXAMPLE.COM "))

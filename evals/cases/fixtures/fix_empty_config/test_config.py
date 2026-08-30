import os
import unittest
from unittest.mock import patch
from config import timeout


class ConfigTests(unittest.TestCase):
    def test_empty_timeout_uses_default(self):
        with patch.dict(os.environ, {"TIMEOUT": ""}, clear=False):
            self.assertEqual(30, timeout())

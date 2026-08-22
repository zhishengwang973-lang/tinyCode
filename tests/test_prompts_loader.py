import tempfile
import unittest
from pathlib import Path

from tinyCode.prompts.loader import load_injection


class PromptLoaderTests(unittest.TestCase):
    def test_load_injection_reads_named_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            injections_dir = Path(tmp) / "injections"
            injections_dir.mkdir()
            (injections_dir / "plan-mode.txt").write_text("plan only", encoding="utf-8")

            self.assertEqual("plan only", load_injection("plan-mode", injections_dir))

    def test_load_injection_rejects_path_traversal_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            injections_dir = root / "injections"
            injections_dir.mkdir()
            modules_dir = root / "modules"
            modules_dir.mkdir()
            (modules_dir / "secret.txt").write_text("outside", encoding="utf-8")

            self.assertEqual("", load_injection("../modules/secret", injections_dir))


if __name__ == "__main__":
    unittest.main()

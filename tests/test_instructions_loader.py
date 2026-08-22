import tempfile
import unittest
from pathlib import Path

from tinyCode.instructions.loader import InstructionsLoader, MAX_INSTRUCTION_CHARS


class InstructionsLoaderTests(unittest.TestCase):
    def test_project_include_cannot_escape_project_root_even_inside_home(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            parent = Path(tmp)
            project = parent / "project"
            project.mkdir()
            (parent / "secret.md").write_text("secret", encoding="utf-8")
            (project / "TINYCODE.md").write_text(
                "@include(../secret.md)\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "路径越界"):
                InstructionsLoader().load_project(project)

    def test_project_include_allows_files_inside_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            (project / "rules.md").write_text("project rules", encoding="utf-8")
            (project / "TINYCODE.md").write_text(
                "main\n@include(rules.md)\n",
                encoding="utf-8",
            )

            loaded = InstructionsLoader().load_project(project)

            self.assertIn("main", loaded)
            self.assertIn("project rules", loaded)

    def test_include_cycle_is_reported_without_recursive_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "TINYCODE.md").write_text(
                "@include(rules.md)\n", encoding="utf-8",
            )
            (project / "rules.md").write_text(
                "@include(TINYCODE.md)\n", encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "循环引用"):
                InstructionsLoader().load_project(project)

    def test_instruction_input_has_total_size_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "TINYCODE.md").write_text(
                "x" * (MAX_INSTRUCTION_CHARS + 1), encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "内容总量超过"):
                InstructionsLoader().load_project(project)


if __name__ == "__main__":
    unittest.main()

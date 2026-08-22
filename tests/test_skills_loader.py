import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tinyCode.skills.loader import SkillLoader


class SkillLoaderTests(unittest.TestCase):
    def test_tools_field_must_be_list_of_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_path = Path(tmp) / "bad.md"
            skill_path.write_text(
                "\n".join([
                    "---",
                    "name: bad",
                    "description: malformed tools",
                    "tools: read_file",
                    "---",
                    "Body",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                skill = SkillLoader().load_one(str(skill_path))

            self.assertIsNone(skill)
            self.assertIn("tools", stderr.getvalue())
            self.assertIn("列表", stderr.getvalue())

    def test_tools_field_accepts_list_of_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_path = Path(tmp) / "good.md"
            skill_path.write_text(
                "\n".join([
                    "---",
                    "name: good",
                    "description: valid tools",
                    "tools:",
                    "  - read_file",
                    "  - grep",
                    "---",
                    "Body",
                    "",
                ]),
                encoding="utf-8",
            )

            skill = SkillLoader().load_one(str(skill_path))

            self.assertIsNotNone(skill)
            self.assertEqual(["read_file", "grep"], skill.meta.tools)


if __name__ == "__main__":
    unittest.main()

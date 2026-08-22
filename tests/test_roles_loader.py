import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from tinyCode.subagent.roles import loader as role_loader
from tinyCode.subagent.roles.loader import RoleLoader


class RoleLoaderTests(unittest.TestCase):
    def test_malformed_frontmatter_is_skipped_without_stopping_valid_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir()
            (project_dir / "bad.md").write_text(
                "\n".join([
                    "---",
                    "- not",
                    "- a",
                    "- dict",
                    "---",
                    "Bad role",
                    "",
                ]),
                encoding="utf-8",
            )
            (project_dir / "good.md").write_text(
                "\n".join([
                    "---",
                    "name: good",
                    "description: valid role",
                    "---",
                    "Good role",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with (
                patch.object(role_loader, "BUILTIN_DIR", root / "missing_builtin"),
                patch.object(role_loader, "USER_DIR", root / "missing_user"),
                patch.object(role_loader, "PROJECT_DIR", project_dir),
                redirect_stderr(stderr),
            ):
                roles = RoleLoader().load_all()

            self.assertEqual(["good"], sorted(roles.keys()))
            self.assertIn("frontmatter", stderr.getvalue())

    def test_tool_allow_and_deny_fields_must_be_lists_of_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir()
            (project_dir / "bad_allow.md").write_text(
                "\n".join([
                    "---",
                    "name: bad_allow",
                    "tools_allow: read_file",
                    "---",
                    "Bad allow",
                    "",
                ]),
                encoding="utf-8",
            )
            (project_dir / "bad_deny.md").write_text(
                "\n".join([
                    "---",
                    "name: bad_deny",
                    "tools_deny:",
                    "  - read_file",
                    "  - 123",
                    "---",
                    "Bad deny",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with (
                patch.object(role_loader, "BUILTIN_DIR", root / "missing_builtin"),
                patch.object(role_loader, "USER_DIR", root / "missing_user"),
                patch.object(role_loader, "PROJECT_DIR", project_dir),
                redirect_stderr(stderr),
            ):
                roles = RoleLoader().load_all()

            self.assertEqual({}, roles)
            self.assertIn("tools_allow", stderr.getvalue())
            self.assertIn("tools_deny", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

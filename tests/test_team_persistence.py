import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.teams import persistence


class TeamPersistenceTests(unittest.TestCase):
    def test_load_team_def_rejects_name_that_escapes_user_team_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_dir = root / "teams"
            user_dir.mkdir()
            (root / "outside.json").write_text(
                json.dumps({"name": "outside", "members": []}),
                encoding="utf-8",
            )

            with patch.object(persistence, "USER_TEAMS_DIR", user_dir):
                self.assertIsNone(persistence.load_team_def("../outside"))

    def test_get_team_dir_rejects_name_that_escapes_project_team_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "teams"

            with patch.object(persistence, "PROJECT_TEAMS_DIR", project_dir):
                with self.assertRaises(ValueError):
                    persistence.get_team_dir("../outside")

            self.assertFalse((Path(tmp) / "outside").exists())

    def test_load_team_def_skips_invalid_members_and_keeps_valid_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp)
            (user_dir / "alpha.json").write_text(
                json.dumps(
                    {
                        "name": "alpha",
                        "members": [
                            {"name": "../bad", "role": "general"},
                            {"role": "missing-name"},
                            {"name": "alice", "role": "general"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(persistence, "USER_TEAMS_DIR", user_dir):
                team = persistence.load_team_def("alpha")

            self.assertIsNotNone(team)
            self.assertEqual(["alice"], [m.name for m in team.members])

    def test_load_team_def_rejects_invalid_top_level_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp)
            (user_dir / "alpha.json").write_text("[]", encoding="utf-8")
            with patch.object(persistence, "USER_TEAMS_DIR", user_dir):
                self.assertIsNone(persistence.load_team_def("alpha"))

    def test_load_team_def_rejects_invalid_round_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp)
            (user_dir / "alpha.json").write_text(
                json.dumps({"members": [], "max_rounds_per_member": "ten"}),
                encoding="utf-8",
            )
            with patch.object(persistence, "USER_TEAMS_DIR", user_dir):
                self.assertIsNone(persistence.load_team_def("alpha"))


if __name__ == "__main__":
    unittest.main()

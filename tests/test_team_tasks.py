import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.teams.models import TaskStatus
from tinyCode.teams import tasks as tasks_module
from tinyCode.teams.tasks import SharedTaskList


class SharedTaskListTests(unittest.TestCase):
    def test_malformed_file_reports_error_instead_of_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            (team_dir / "tasks.json").write_text("{bad", encoding="utf-8")

            tasks = SharedTaskList(team_dir)

            self.assertIn("JSONDecodeError", tasks.load_error)

    def test_load_skips_task_with_invalid_status_and_keeps_valid_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            (team_dir / "tasks.json").write_text(
                json.dumps(
                    {
                        "bad": {
                            "id": "bad",
                            "name": "broken",
                            "status": "unknown",
                        },
                        "good": {
                            "id": "good",
                            "name": "ready",
                            "status": "pending",
                            "created_at": "2026-01-01T00:00:00+00:00",
                        },
                    }
                ),
                encoding="utf-8",
            )

            tasks = SharedTaskList(team_dir)

            self.assertIsNone(tasks.get("bad"))
            self.assertEqual(TaskStatus.PENDING, tasks.get("good").status)

    def test_load_skips_task_with_unhashable_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            (team_dir / "tasks.json").write_text(
                json.dumps({"bad": {"id": "bad", "status": []}}),
                encoding="utf-8",
            )

            tasks = SharedTaskList(team_dir)

            self.assertIsNone(tasks.get("bad"))

    def test_ready_tasks_ignores_task_with_missing_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = SharedTaskList(Path(tmp))
            blocked = tasks.create("blocked", depends_on=["missing"])

            self.assertEqual([], tasks.ready_tasks())
            self.assertEqual(TaskStatus.PENDING, tasks.get(blocked.id).status)

    def test_member_created_tasks_are_tagged_with_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = SharedTaskList(Path(tmp))
            tasks.set_active_run("run-123")

            created = tasks.create("follow-up", "member discovered work")

            self.assertEqual("run-123", created.run_id)
            self.assertEqual([created.id], [task.id for task in tasks.list_for_run("run-123")])

    def test_task_run_and_preferred_member_survive_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            tasks = SharedTaskList(team_dir)
            created = tasks.create(
                "implement", "bounded scope", preferred_member="alice", run_id="run-a",
            )

            loaded = SharedTaskList(team_dir).get(created.id)

            self.assertEqual("run-a", loaded.run_id)
            self.assertEqual("alice", loaded.preferred_member)

    def test_update_ignores_invalid_field_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = SharedTaskList(Path(tmp))
            created = tasks.create("safe")

            updated = tasks.update(created.id, status="completed", name=42, id="changed")

            self.assertEqual(TaskStatus.PENDING, updated.status)
            self.assertEqual("safe", updated.name)
            self.assertEqual(created.id, updated.id)

    def test_reconcile_interrupted_marks_in_progress_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            tasks = SharedTaskList(team_dir)
            task = tasks.create("running")
            tasks.assign(task.id, "alice")

            restored = SharedTaskList(team_dir)
            count = restored.reconcile_interrupted()

            self.assertEqual(1, count)
            self.assertEqual(TaskStatus.FAILED, restored.get(task.id).status)
            self.assertIn("进程中断", restored.get(task.id).result)

    def test_large_result_is_bounded_and_full_text_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            tasks = SharedTaskList(team_dir)
            task = tasks.create("large")

            with patch.object(tasks_module, "MAX_TASK_RESULT_CHARS", 10):
                tasks.complete(task.id, "x" * 20)

            self.assertIn("完整结果", tasks.get(task.id).result)
            self.assertEqual(
                "x" * 20,
                (team_dir / "results" / f"{task.id}.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()

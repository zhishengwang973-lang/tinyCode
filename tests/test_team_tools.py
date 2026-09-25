import tempfile
import unittest
from pathlib import Path

from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.tasks import SharedTaskList
from tinyCode.teams.tools import create_team_tools


class TeamToolTests(unittest.IsolatedAsyncioTestCase):
    def _tools(self, root: Path):
        tasks = SharedTaskList(root)
        mailbox = Mailbox(root, "alice")
        return {
            tool.name: tool
            for tool in create_team_tools(
                root, tasks, mailbox, "alice", ["alice", "bob"],
            )
        }, tasks

    async def test_create_task_rejects_invalid_types_without_persisting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools, tasks = self._tools(Path(tmp))

            result = await tools["team_create_task"].execute(
                name=42, description="inspect",
            )

            self.assertFalse(result.success)
            self.assertEqual([], tasks.list_all())

    async def test_update_task_rejects_unhashable_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools, tasks = self._tools(Path(tmp))
            task = tasks.create("inspect", "inspect code")

            result = await tools["team_update_task"].execute(
                task_id=task.id, status=[],
            )

            self.assertFalse(result.success)
            self.assertEqual("pending", tasks.get(task.id).status.value)

    async def test_send_message_rejects_unknown_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools, _tasks = self._tools(root)

            result = await tools["team_send_message"].execute(
                to="charlie", content="hello",
            )

            self.assertFalse(result.success)
            self.assertIn("团队成员不存在", result.error)
            self.assertFalse((root / "mailboxes" / "charlie.jsonl").exists())

    async def test_member_cannot_update_another_members_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools, tasks = self._tools(Path(tmp))
            task = tasks.create("bob-task")
            tasks.assign(task.id, "bob")

            result = await tools["team_update_task"].execute(
                task_id=task.id, status="completed",
            )

            self.assertFalse(result.success)
            self.assertEqual("in_progress", tasks.get(task.id).status.value)

    async def test_member_can_send_status_to_lead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools, _ = self._tools(root)

            result = await tools["team_send_message"].execute(
                to="lead", content="blocked on API",
            )

            self.assertTrue(result.success)
            self.assertEqual(
                ["blocked on API"],
                [message.content for message in Mailbox(root, "lead").read_new()],
            )


if __name__ == "__main__":
    unittest.main()

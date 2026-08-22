import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tinyCode.teams.models import MemberDef, TeamDef
from tinyCode.teams.lead import LeadAgent
from tinyCode.teams.tasks import SharedTaskList
from tinyCode.tools.registry import ToolRegistry


class TeamOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_team_reports_unknown_team(self):
        from tinyCode.teams.orchestrator import run_team

        with patch("tinyCode.teams.orchestrator.load_team_def", return_value=None):
            result = await run_team("missing", "do work")

        self.assertEqual("Team 'missing' 不存在", result)

    async def test_run_team_rejects_team_without_members(self):
        from tinyCode.teams.orchestrator import run_team

        team_def = TeamDef(name="alpha", members=[])

        with patch("tinyCode.teams.orchestrator.load_team_def", return_value=team_def):
            result = await run_team("alpha", "do work")

        self.assertEqual("Team 'alpha' 没有可用成员", result)

    async def test_run_team_requires_runtime_dependencies(self):
        from tinyCode.teams.orchestrator import run_team

        team_def = TeamDef(
            name="alpha",
            members=[MemberDef(name="alice", role="coder")],
        )

        with patch("tinyCode.teams.orchestrator.load_team_def", return_value=team_def):
            result = await run_team("alpha", "ship feature")

        self.assertEqual("Team 运行需要 provider、tool_registry 和 tool_executor", result)

    async def test_run_team_requires_member_approval(self):
        from tinyCode.teams.orchestrator import run_team

        team_def = TeamDef(
            name="alpha",
            members=[MemberDef(name="alice", needs_approval=True)],
        )
        with patch("tinyCode.teams.orchestrator.load_team_def", return_value=team_def):
            result = await run_team(
                "alpha", "do work", provider=object(),
                tool_registry=object(), tool_executor=object(),
            )
        self.assertIn("需要显式批准", result)

    async def test_run_team_rejects_unimplemented_terminal_backend(self):
        from tinyCode.teams.orchestrator import run_team

        team_def = TeamDef(
            name="alpha",
            members=[MemberDef(name="alice", backend="terminal")],
        )
        with patch("tinyCode.teams.orchestrator.load_team_def", return_value=team_def):
            result = await run_team(
                "alpha", "do work", provider=object(),
                tool_registry=object(), tool_executor=object(),
            )
        self.assertIn("backend='coro'", result)

    async def test_run_team_builds_lead_agent_and_executes_goal(self):
        from tinyCode.teams.orchestrator import run_team

        team_def = TeamDef(
            name="alpha",
            members=[MemberDef(name="alice", role="coder")],
        )
        lead = AsyncMock()
        lead.execute.return_value = "done"
        team_dir = Path("/tmp/team-alpha")
        provider = object()
        tool_registry = ToolRegistry()
        tool_executor = object()

        with patch("tinyCode.teams.orchestrator.load_team_def", return_value=team_def), \
             patch("tinyCode.teams.orchestrator.get_team_dir", return_value=team_dir), \
             patch("tinyCode.teams.orchestrator.TeamMember") as member_cls, \
             patch("tinyCode.teams.orchestrator.SharedTaskList") as tasks_cls, \
             patch("tinyCode.teams.orchestrator.GitMerger") as merger_cls, \
             patch("tinyCode.teams.orchestrator.LeadAgent", return_value=lead):
            result = await run_team(
                "alpha",
                "ship feature",
                provider=provider,
                tool_registry=tool_registry,
                tool_executor=tool_executor,
            )

        self.assertEqual("done", result)
        member_cls.assert_called_once()
        member_kwargs = member_cls.call_args.kwargs
        self.assertEqual(team_def.members[0], member_kwargs["member_def"])
        self.assertEqual(team_dir, member_kwargs["team_dir"])
        self.assertIs(provider, member_kwargs["provider"])
        self.assertIs(tool_executor, member_kwargs["tool_executor"])
        self.assertEqual(team_def.max_rounds_per_member, member_kwargs["max_rounds"])
        self.assertEqual(Path.cwd().resolve(), member_kwargs["workspace"])
        self.assertIsNot(member_kwargs["tool_registry"], tool_registry)
        self.assertIsNotNone(member_kwargs["tool_registry"].get("team_create_task"))
        tasks_cls.assert_called_once_with(team_dir)
        merger_cls.assert_called_once()
        lead.execute.assert_awaited_once_with("ship feature")

    async def test_lead_ignores_stale_in_progress_tasks_from_previous_run(self):
        class Member:
            def __init__(self):
                self.defn = MemberDef(name="alice")

            async def run(self, task: str) -> str:
                return "done"

        class Merger:
            async def merge(self, branch: str):
                return True, "ok"

        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            tasks = SharedTaskList(team_dir)
            stale = tasks.create("stale", "from old run")
            tasks.assign(stale.id, "ghost")
            lead = LeadAgent(
                team_def=TeamDef(name="alpha", members=[MemberDef(name="alice")]),
                team_dir=team_dir,
                members={"alice": Member()},
                task_list=tasks,
                merger=Merger(),
            )

            result = await lead.execute("new goal")

            self.assertIn("1/1 完成", result)
            self.assertEqual(2, len(tasks.list_all()))


if __name__ == "__main__":
    unittest.main()

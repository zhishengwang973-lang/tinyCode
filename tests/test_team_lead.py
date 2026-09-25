import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.teams.lead import LeadAgent
from tinyCode.teams.models import MemberDef, TeamDef
from tinyCode.teams.tasks import SharedTaskList


class PlanningProvider:
    def begin_request(self):
        pass

    async def chat_stream(self, messages):
        yield (
            '[{"name":"core","description":"implement core","member":"alice",'
            '"depends_on":[]},{"name":"tests","description":"write edge tests",'
            '"member":"bob","depends_on":[]}]'
        )


class Member:
    def __init__(self, name):
        self.defn = MemberDef(name=name)
        self.received = []

    async def run(self, task):
        self.received.append(task)
        return f"done: {task}"


class Merger:
    async def merge(self, branch):
        return True, "ok"


class TeamLeadTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_plan_is_distinct_and_honors_member_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            members = {"alice": Member("alice"), "bob": Member("bob")}
            team_def = TeamDef(
                name="alpha",
                members=[MemberDef(name="alice"), MemberDef(name="bob")],
            )
            tasks = SharedTaskList(Path(tmp))
            lead = LeadAgent(
                team_def, Path(tmp), members, tasks, Merger(),
                provider=PlanningProvider(),
            )

            result = await lead.execute("ship feature")

            self.assertIn("2/2 完成", result)
            self.assertIn("Team 统计: Turn 0 · 模型请求 1 次 · Token 不可用", result)
            self.assertEqual(["implement core"], members["alice"].received)
            self.assertEqual(["write edge tests"], members["bob"].received)
            current = [task for task in tasks.list_all() if task.run_id]
            self.assertEqual({"alice", "bob"}, {task.preferred_member for task in current})

    def test_plan_parser_rejects_forward_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            members = {"alice": Member("alice")}
            lead = LeadAgent(
                TeamDef(name="a", members=[MemberDef(name="alice")]),
                Path(tmp), members, SharedTaskList(Path(tmp)), Merger(),
            )
            raw = (
                '[{"name":"bad","description":"x","member":"alice",'
                '"depends_on":[1]}]'
            )
            self.assertEqual([], lead._parse_plan(raw))

    async def test_oversized_model_plan_falls_back_without_crashing(self):
        class OversizedPlanningProvider:
            async def chat_stream(self, messages):
                yield "x" * 20

        with tempfile.TemporaryDirectory() as tmp:
            members = {"alice": Member("alice")}
            lead = LeadAgent(
                TeamDef(name="a", members=[MemberDef(name="alice")]),
                Path(tmp), members, SharedTaskList(Path(tmp)), Merger(),
                provider=OversizedPlanningProvider(),
            )

            with patch("tinyCode.teams.lead.MAX_TEAM_PLAN_CHARS", 10):
                result = await lead.execute("ship safely")

            self.assertIn("1/1 完成", result)
            self.assertEqual(1, len(members["alice"].received))

    async def test_merge_skips_member_without_tasks_and_uses_actual_branch(self):
        class OneMemberPlanProvider:
            async def chat_stream(self, messages):
                yield (
                    '[{"name":"core","description":"implement",'
                    '"member":"alice","depends_on":[]}]'
                )

        class WorktreeMember(Member):
            def __init__(self, name, workspace):
                super().__init__(name)
                self.defn = MemberDef(name=name, worktree=name)
                self.workspace = workspace

        class RecordingMerger:
            def __init__(self):
                self.merged = []

            async def worktree_branch(self, worktree):
                return True, "custom/alice"

            async def prepare_worktree(self, worktree, member_name):
                return True, "ok"

            async def merge(self, branch):
                self.merged.append(branch)
                return True, "ok"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            members = {
                "alice": WorktreeMember("alice", root / "alice"),
                "bob": WorktreeMember("bob", root / "bob"),
            }
            merger = RecordingMerger()
            lead = LeadAgent(
                TeamDef(
                    name="alpha",
                    members=[members["alice"].defn, members["bob"].defn],
                ),
                root,
                members,
                SharedTaskList(root),
                merger,
                provider=OneMemberPlanProvider(),
            )

            result = await lead.execute("ship")

            self.assertEqual(["custom/alice"], merger.merged)
            self.assertIn("bob: 本轮未分配任务，已跳过合并", result)


if __name__ == "__main__":
    unittest.main()

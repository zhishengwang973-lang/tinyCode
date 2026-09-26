import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.config.models import TeamAutomationConfig
from tinyCode.teams.auto import AutoTeamService
from tinyCode.worktree.manager import GitWorktreeManager


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


class AutoTeamRoutingTests(unittest.TestCase):
    def _service(self, mode: str = "auto") -> AutoTeamService:
        return AutoTeamService(
            TeamAutomationConfig(mode=mode),
            repo_root=Path.cwd(),
            worktree_manager=object(),
            provider=object(),
            tool_registry=object(),
            tool_executor=object(),
        )

    def test_simple_programming_answer_stays_single_agent(self):
        service = self._service()
        self.assertIsNone(service.propose("给我写一个快速排序算法"))
        self.assertIsNone(service.propose("解释 Python 装饰器是什么"))

    def test_complex_parallel_delivery_proposes_team(self):
        proposal = self._service().propose(
            "全面重构整个项目的缓存模块，实现并发安全，补齐测试、文档和回归验证"
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(["implementer", "verifier", "reviewer"], [
            member.name for member in proposal.members
        ])

    def test_explicit_team_and_opt_out_are_respected(self):
        self.assertIsNotNone(self._service().propose("请用多个 Agent 并行检查三个模块"))
        self.assertIsNone(self._service("team").propose("不要使用 Team，修复这个问题"))


class AutoTeamReviewFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _git(self.root, "init", "-b", "main")
        _git(self.root, "config", "user.email", "tests@example.com")
        _git(self.root, "config", "user.name", "TinyCode Tests")
        (self.root / "base.txt").write_text("base\n", encoding="utf-8")
        _git(self.root, "add", "base.txt")
        _git(self.root, "commit", "-m", "initial")
        self.manager = GitWorktreeManager(self.root)
        self.service = AutoTeamService(
            TeamAutomationConfig(max_members=2, cleanup_after_apply=True),
            repo_root=self.root,
            worktree_manager=self.manager,
            provider=object(),
            tool_registry=object(),
            tool_executor=object(),
        )

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_run_keeps_main_unchanged_until_review_is_applied(self):
        proposal = self.service.propose(
            "请用多个 Agent 并行实现功能并补齐测试"
        )
        self.assertIsNotNone(proposal)

        async def fake_run_team(_name, _goal, **kwargs):
            definition = kwargs["definition"]
            for member in definition.members:
                info = await self.manager.status(member.worktree)
                self.assertIsNotNone(info)
                path = Path(info.path)
                (path / f"{member.name}.txt").write_text(
                    member.name + "\n", encoding="utf-8",
                )
                _git(path, "add", "-A")
                _git(path, "commit", "-m", member.name)
            return "## Team 执行完成"

        original_head = _git(self.root, "rev-parse", "HEAD")
        with patch("tinyCode.teams.auto.run_team", side_effect=fake_run_team):
            result = await self.service.run(proposal)

        self.assertTrue(result.review_ready)
        self.assertEqual(original_head, _git(self.root, "rev-parse", "HEAD"))
        self.assertFalse((self.root / "implementer.txt").exists())

        applied, message = await self.service.apply(result.run_id)
        self.assertTrue(applied, message)
        self.assertTrue((self.root / "implementer.txt").exists())
        self.assertTrue((self.root / "verifier.txt").exists())
        self.assertEqual([], await self.manager.list_worktrees())

    async def test_discard_preserves_main_and_marks_record(self):
        proposal = self.service.propose("请用多个 Agent 并行实现功能和测试")

        async def fake_run_team(_name, _goal, **kwargs):
            member = kwargs["definition"].members[0]
            info = await self.manager.status(member.worktree)
            path = Path(info.path)
            (path / "change.txt").write_text("change\n", encoding="utf-8")
            _git(path, "add", "-A")
            _git(path, "commit", "-m", "change")
            return "done"

        with patch("tinyCode.teams.auto.run_team", side_effect=fake_run_team):
            result = await self.service.run(proposal)
        discarded, message = await self.service.discard(result.run_id)
        self.assertTrue(discarded, message)
        self.assertFalse((self.root / "change.txt").exists())
        self.assertEqual("discarded", self.service.list_reviews()[0].status)

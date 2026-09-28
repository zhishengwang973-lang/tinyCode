import tempfile
import unittest
from pathlib import Path

from tinyCode.goals import GoalService, GoalStatus, GoalStore
from tinyCode.goals.tools import GoalCompleteTool, GoalStatusTool


class GoalServiceTests(unittest.IsolatedAsyncioTestCase):
    def _service(self, root: Path) -> GoalService:
        service = GoalService(GoalStore(root), default_max_turns=2)
        service.bind_session("session123")
        return service

    async def test_goal_is_persisted_per_session_and_requires_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = self._service(root)
            goal = service.start("修复失败测试，并通过测试命令验证结果")
            self.assertTrue((root / ".tinyCode" / "goals" / "session123.json").is_file())
            self.assertIn("进行中", service.render())

            complete = GoalCompleteTool(service)
            failed = await complete.execute(evidence="完成")
            self.assertFalse(failed.success)
            succeeded = await complete.execute(evidence="pytest tests/test_example.py 通过")
            self.assertTrue(succeeded.success)
            self.assertEqual(GoalStatus.COMPLETED, goal.status)

            restored = self._service(root)
            self.assertIsNotNone(restored.current)
            self.assertEqual(GoalStatus.COMPLETED, restored.current.status)
            self.assertIn("pytest", restored.render())

    async def test_only_productive_active_goal_turn_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.start("完成并验证迁移")
            self.assertFalse(service.should_continue(
                tool_calls=0, terminal_reason="no_tool_call", has_pending_user_input=False,
            ))
            self.assertFalse(service.should_continue(
                tool_calls=1, terminal_reason="cancelled", has_pending_user_input=False,
            ))
            self.assertFalse(service.should_continue(
                tool_calls=1, terminal_reason="no_tool_call", has_pending_user_input=True,
            ))
            self.assertTrue(service.should_continue(
                tool_calls=1, terminal_reason="no_tool_call", has_pending_user_input=False,
            ))

    async def test_budget_limit_requires_explicit_extension_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.start("完成迁移")
            service.record_turn(model_requests=1, tool_calls=1, reason="no_tool_call")
            goal = service.record_turn(model_requests=1, tool_calls=1, reason="no_tool_call")
            self.assertEqual(GoalStatus.BUDGET_LIMITED, goal.status)
            with self.assertRaisesRegex(ValueError, "新增回合数"):
                service.resume()
            service.resume(additional_turns=3)
            self.assertEqual(GoalStatus.ACTIVE, service.current.status)
            self.assertEqual(5, service.current.max_turns)

    async def test_completed_goal_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.start("完成迁移")
            self.assertEqual("Goal 已标记完成", service.complete("测试日志显示所有用例通过"))
            with self.assertRaisesRegex(ValueError, "已完成"):
                service.resume()

    async def test_final_response_closes_active_goal_when_model_omits_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.start("总结项目能力")
            goal = service.complete_from_final_response("该项目提供 Agent、工具与会话能力。")

            self.assertIsNotNone(goal)
            self.assertEqual(GoalStatus.COMPLETED, goal.status)
            self.assertIn("自动归档", goal.completion_evidence)
            self.assertFalse(service.active)

    async def test_empty_final_response_does_not_close_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.start("总结项目能力")

            self.assertIsNone(service.complete_from_final_response("  "))
            self.assertTrue(service.active)

    async def test_final_response_wins_over_budget_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = GoalService(GoalStore(Path(tmp)), default_max_turns=1)
            service.bind_session("budget-final-session")
            service.start("总结项目能力")
            service.record_turn(model_requests=1, tool_calls=0, reason="no_tool_call")

            goal = service.complete_from_final_response("最终项目总结已交付。")
            self.assertIsNotNone(goal)
            self.assertEqual(GoalStatus.COMPLETED, goal.status)

    async def test_status_tool_is_safe_without_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            result = await GoalStatusTool(service).execute()
            self.assertTrue(result.success)
            self.assertIn("没有 Goal", result.content)

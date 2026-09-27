import tempfile
import unittest
from pathlib import Path

from tinyCode.agent.task_mode import TaskMode
from tinyCode.config.models import TaskPlanningConfig
from tinyCode.tasking.models import ExecutorKind, TaskNode, TaskPlan
from tinyCode.tasking.planner import TaskPlanningService
from tinyCode.tasking.store import TaskPlanStore


class FakePlannerProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_usage = {}
        self.requests = 0

    def begin_request(self) -> None:
        self.requests += 1

    async def chat_stream(self, messages):
        del messages
        yield self.response


class TaskPlanningTests(unittest.IsolatedAsyncioTestCase):
    def _service(self, root: Path, response: str) -> TaskPlanningService:
        return TaskPlanningService(
            TaskPlanningConfig(enabled=True, max_tasks=6, min_task_chars=1),
            FakePlannerProvider(response),
            TaskPlanStore(root),
        )

    async def test_model_plan_is_persisted_and_dependencies_are_enforced(self):
        response = """[
          {"id":"inspect","title":"检查","description":"读取现状","depends_on":[],"read_scope":["src/"],"write_scope":[],"acceptance":["确认范围"],"executor":"main"},
          {"id":"implement","title":"实现","description":"修改代码","depends_on":["inspect"],"read_scope":[],"write_scope":["src/app.py"],"acceptance":["功能完成"],"executor":"main"},
          {"id":"verify","title":"验证","description":"运行测试","depends_on":["implement"],"read_scope":[],"write_scope":[],"acceptance":["测试通过"],"executor":"main"}
        ]"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = self._service(root, response)
            result = await service.create_if_needed("请跨模块实现功能并补测试和文档", TaskMode.MODIFY)

            self.assertIsNotNone(result.plan)
            self.assertEqual("model", result.source)
            self.assertTrue((root / ".tinyCode" / "task_plans" / f"{result.plan.id}.json").is_file())
            self.assertIn("依赖尚未完成", service.update_task("implement", "in_progress"))
            self.assertIn("已更新为 in_progress", service.update_task("inspect", "in_progress"))
            self.assertIn("已更新为 completed", service.update_task("inspect", "completed"))
            self.assertIn("已更新为 in_progress", service.update_task("implement", "in_progress"))
            self.assertIn("已更新为 completed", service.update_task("implement", "completed"))
            self.assertIn("已更新为 in_progress", service.update_task("verify", "in_progress"))
            self.assertIn("已更新为 completed", service.update_task("verify", "completed"))
            completed = service.clear_active(outcome="completed")
            self.assertEqual("completed", completed.status.value)

    async def test_invalid_model_output_uses_safe_serial_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), "not-json")
            result = await service.create_if_needed("对整个项目进行重构、测试和文档验证", TaskMode.MODIFY)
            self.assertIsNotNone(result.plan)
            self.assertEqual("fallback", result.source)
            self.assertEqual(["scope", "execute", "verify"], [task.id for task in result.plan.tasks])
            self.assertEqual("in_progress", result.plan.tasks[0].status.value)
            updated = service.observe_successful_workspace_write("apply_patch")
            self.assertIsNotNone(updated)
            self.assertEqual("completed", updated.tasks[0].status.value)
            self.assertEqual("in_progress", updated.tasks[1].status.value)

    def test_overlapping_writes_require_a_dependency(self):
        plan = TaskPlan.create(
            "goal",
            [
                TaskNode("one", "one", "one", write_scope=["src/app.py"]),
                TaskNode("two", "two", "two", write_scope=["src/"]),
            ],
            mode="modify", source="test",
        )
        with self.assertRaisesRegex(ValueError, "写入范围重叠"):
            TaskPlanningService._validate_plan(plan)

    def test_interrupted_active_plan_is_marked_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskPlanStore(Path(tmp))
            plan = TaskPlan.create(
                "goal", [
                    TaskNode("one", "one", "one"),
                    TaskNode("two", "two", "two", depends_on=["one"]),
                ], mode="inspect", source="test",
            )
            store.save(plan)
            self.assertEqual(1, store.mark_interrupted_active_plans())
            restored = store.load(plan.id)

        self.assertEqual("interrupted", restored.status.value)
        self.assertIn("核验", restored.error)

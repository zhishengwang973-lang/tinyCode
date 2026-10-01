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

    async def test_manual_draft_requires_approval_before_nodes_can_advance(self):
        response = """[
          {"id":"inspect","title":"检查","description":"读取现状","depends_on":[],"read_scope":["src/"],"write_scope":[],"acceptance":["确认范围"],"executor":"main"},
          {"id":"implement","title":"实现","description":"修改代码","depends_on":["inspect"],"read_scope":[],"write_scope":["src/app.py"],"acceptance":["功能完成"],"executor":"main"}
        ]"""
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), response)
            result = await service.create_draft("实现配置迁移并补测试", TaskMode.MODIFY)

            self.assertIsNotNone(result.plan)
            self.assertEqual("draft", result.plan.status.value)
            self.assertTrue(all(task.status.value == "pending" for task in result.plan.tasks))
            self.assertIn("尚未批准", service.update_task("inspect", "in_progress"))

            plan, message = service.approve_draft()
            self.assertIsNotNone(plan)
            self.assertIn("开始执行", message)
            self.assertEqual("active", plan.status.value)
            self.assertEqual("in_progress", plan.tasks[0].status.value)

    async def test_revision_keeps_draft_identity_and_discard_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), "not-json")
            draft = await service.create_draft("重构服务层并补测试", TaskMode.MODIFY)
            self.assertIsNotNone(draft.plan)
            original_id = draft.plan.id

            revised = await service.revise_draft("增加迁移回滚步骤")
            self.assertIsNotNone(revised.plan)
            self.assertEqual(original_id, revised.plan.id)
            self.assertEqual("draft", revised.plan.status.value)
            self.assertEqual("计划草案已丢弃", service.discard_draft())
            restored = TaskPlanStore(Path(tmp)).load(original_id)

        self.assertIsNotNone(restored)
        self.assertEqual("discarded", restored.status.value)

    async def test_latest_draft_is_restored_but_not_interrupted_at_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = self._service(root, "not-json")
            draft = await service.create_draft("实现可靠的配置迁移", TaskMode.MODIFY)
            self.assertIsNotNone(draft.plan)

            restored_service = self._service(root, "not-json")

        self.assertIsNotNone(restored_service.active_plan)
        self.assertEqual(draft.plan.id, restored_service.active_plan.id)
        self.assertEqual("draft", restored_service.active_plan.status.value)

    async def test_auto_planning_never_replaces_pending_manual_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), "not-json")
            draft = await service.create_draft("重构认证模块并补测试", TaskMode.MODIFY)
            skipped = await service.create_if_needed(
                "跨模块重构整个项目并补测试、文档和完整验证", TaskMode.MODIFY,
            )

        self.assertIsNotNone(draft.plan)
        self.assertIsNone(skipped.plan)
        self.assertEqual("pending_manual_draft", skipped.source)
        self.assertEqual(draft.plan.id, service.active_plan.id)

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

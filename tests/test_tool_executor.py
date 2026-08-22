import unittest

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.executor import ToolExecutor


class RecordingTool(BaseTool):
    def __init__(self) -> None:
        self.executed = False

    @property
    def name(self) -> str:
        return "recording_tool"

    @property
    def description(self) -> str:
        return "records whether it ran"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WRITE

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    async def execute(self, **kwargs) -> ToolResult:
        self.executed = True
        return ToolResult(success=True, content="ran")


class BadReturnTool(RecordingTool):
    async def execute(self, **kwargs):
        self.executed = True
        return "not a tool result"


class MalformedResultTool(RecordingTool):
    async def execute(self, **kwargs):
        self.executed = True
        return ToolResult(success=True, content=["not text"], error="")


class ToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_returns_structured_failure_when_tool_is_missing(self):
        result = await ToolExecutor().execute(None, {})

        self.assertFalse(result.success)
        self.assertIn("未知工具", result.error)

    async def test_execute_rejects_non_object_params_before_tool_runs(self):
        tool = RecordingTool()

        result = await ToolExecutor().execute(tool, [])

        self.assertFalse(result.success)
        self.assertIn("工具参数必须是对象", result.error)
        self.assertFalse(tool.executed)

    async def test_execute_rejects_non_tool_result_return_value(self):
        tool = BadReturnTool()

        result = await ToolExecutor().execute(tool, {})

        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertIn("工具返回值必须是 ToolResult", result.error)
        self.assertTrue(tool.executed)

    async def test_execute_rejects_invalid_tool_result_fields(self):
        result = await ToolExecutor().execute(MalformedResultTool(), {})

        self.assertFalse(result.success)
        self.assertIn("字段类型无效", result.error)


if __name__ == "__main__":
    unittest.main()

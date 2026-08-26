import unittest

import tinyCode.tools as tools_package
from tinyCode.main import _create_tool_registry
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.delete_file import DeleteFileTool
from tinyCode.tools.apply_patch import ApplyPatchTool
from tinyCode.tools.request_user_input import RequestUserInputTool
from tinyCode.tools.web_fetch import WebFetchTool
from tinyCode.tools.web_search import WebSearchTool
from tinyCode.tools.registry import ToolRegistry


class DummyTool(BaseTool):
    def __init__(self, name: str, parameters=None, description="dummy") -> None:
        self._name = name
        self._parameters = parameters if parameters is not None else []
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="ok")


class ToolRegistryTests(unittest.TestCase):
    def test_tools_package_exports_tool_category_contract(self):
        self.assertIs(ToolCategory, tools_package.ToolCategory)

    def test_tools_package_exports_delete_file_tool(self):
        self.assertIs(DeleteFileTool, tools_package.DeleteFileTool)

    def test_main_registry_exposes_delete_file_tool(self):
        registry = _create_tool_registry()

        self.assertIsInstance(registry.get("delete_file"), DeleteFileTool)

    def test_main_registry_exposes_new_builtin_tools(self):
        registry = _create_tool_registry()

        self.assertIsInstance(registry.get("apply_patch"), ApplyPatchTool)
        self.assertIsInstance(registry.get("request_user_input"), RequestUserInputTool)
        self.assertIsInstance(registry.get("web_search"), WebSearchTool)
        self.assertIsInstance(registry.get("web_fetch"), WebFetchTool)

    def test_array_parameter_schema_includes_item_type_and_default(self):
        registry = _create_tool_registry()

        schema = registry.get("request_user_input").to_openai_schema()
        options = schema["function"]["parameters"]["properties"]["options"]

        self.assertEqual({"type": "string"}, options["items"])
        self.assertEqual([], options["default"])

    def test_duplicate_tool_name_is_rejected(self):
        registry = ToolRegistry()
        registry.register(DummyTool("read_file"))

        with self.assertRaisesRegex(ValueError, "read_file"):
            registry.register(DummyTool("read_file"))

    def test_provider_schemas_use_canonical_name_order(self):
        registry = ToolRegistry()
        registry.register(DummyTool("zeta"))
        registry.register(DummyTool("alpha"))
        registry.register(DummyTool("middle"))

        openai_names = [
            item["function"]["name"] for item in registry.to_openai_format()
        ]
        anthropic_names = [item["name"] for item in registry.to_anthropic_format()]

        self.assertEqual(["alpha", "middle", "zeta"], openai_names)
        self.assertEqual(openai_names, anthropic_names)

    def test_empty_tool_name_is_rejected(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "工具名必须是非空字符串"):
            registry.register(DummyTool(""))

    def test_non_string_tool_name_is_rejected(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "工具名必须是非空字符串"):
            registry.register(DummyTool(123))

    def test_provider_incompatible_tool_name_is_rejected(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "A-Za-z0-9"):
            registry.register(DummyTool("server/read"))

        with self.assertRaisesRegex(ValueError, "64"):
            registry.register(DummyTool("x" * 65))

    def test_non_string_tool_description_is_rejected(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "工具描述必须是字符串"):
            registry.register(DummyTool("bad_description", description=123))

    def test_non_list_tool_parameters_are_rejected(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "工具参数必须是列表"):
            registry.register(DummyTool("bad_params", parameters="path"))

    def test_parameter_schema_string_fields_are_validated(self):
        registry = ToolRegistry()
        bad_params = [ToolParameter(123, "string", "path")]

        with self.assertRaisesRegex(ValueError, "工具参数名必须是非空字符串"):
            registry.register(DummyTool("bad_param_name", parameters=bad_params))

    def test_parameter_required_flag_is_boolean(self):
        registry = ToolRegistry()
        bad_params = [ToolParameter("path", "string", "path", required="yes")]

        with self.assertRaisesRegex(ValueError, "工具参数 required 必须是布尔值"):
            registry.register(DummyTool("bad_required", parameters=bad_params))

    def test_duplicate_parameter_name_is_rejected(self):
        registry = ToolRegistry()
        params = [
            ToolParameter("path", "string", "first"),
            ToolParameter("path", "string", "second"),
        ]

        with self.assertRaisesRegex(ValueError, "工具参数名重复"):
            registry.register(DummyTool("duplicate_params", parameters=params))

    def test_invalid_json_schema_parameter_type_is_rejected(self):
        registry = ToolRegistry()
        params = [ToolParameter("path", "pathlib.Path", "path")]

        with self.assertRaisesRegex(ValueError, "有效 JSON Schema 类型"):
            registry.register(DummyTool("bad_param_type", parameters=params))


if __name__ == "__main__":
    unittest.main()

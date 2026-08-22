import unittest

from tinyCode.mcp.adapter import (
    MCPPromptAdapter, MCPResourceAdapter, MCPToolAdapter,
    provider_safe_tool_name,
)
from tinyCode.tools.base import ToolCategory


class FakeMCPClient:
    server_name = "fake"

    async def call_tool(self, name: str, arguments: dict) -> str:
        raise RuntimeError("remote tool failed")


class FakePromptClient:
    server_name = "fake"

    def __init__(self) -> None:
        self.resource_calls: list[str] = []
        self.prompt_calls: list[tuple[str, dict]] = []
        self.prompt_result: dict = {"messages": []}

    async def list_resources(self) -> list[dict]:
        return [{"uri": "file:///README.md", "name": "README"}]

    async def read_resource(self, uri: str) -> dict:
        self.resource_calls.append(uri)
        return {"text": "resource text"}

    async def list_prompts(self) -> list[dict]:
        return [{"name": "review", "description": "Review code"}]

    async def get_prompt(self, name: str, arguments: dict) -> dict:
        self.prompt_calls.append((name, arguments))
        return self.prompt_result


class MCPToolAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_error_returns_structured_failure(self):
        adapter = MCPToolAdapter(
            FakeMCPClient(),
            {
                "name": "remote_read",
                "description": "remote read",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "path"}},
                    "required": ["path"],
                },
            },
        )

        result = await adapter.execute(path="README.md")

        self.assertFalse(result.success)
        self.assertEqual("remote tool failed", result.error)

    async def test_tool_schema_generation_skips_malformed_input_schema_fields(self):
        adapter = MCPToolAdapter(
            FakeMCPClient(),
            {
                "name": "remote_read",
                "description": "remote read",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to read"},
                        "limit": {"type": 7, "description": ["bad"]},
                        "custom": {"type": "pathlib.Path"},
                        "broken": "not an object",
                    },
                    "required": "path",
                },
            },
        )

        schema = adapter.to_openai_schema()

        parameters = schema["function"]["parameters"]
        self.assertEqual(
            {
                "path": {"type": "string", "description": "Path to read"},
                "limit": {"type": "string", "description": ""},
                "custom": {"type": "string", "description": ""},
            },
            parameters["properties"],
        )
        self.assertEqual([], parameters["required"])

    async def test_execute_resolves_current_pool_client(self):
        class WorkingClient:
            server_name = "fake"

            async def call_tool(self, name, arguments):
                return f"{name}:{arguments['path']}"

        replacement = WorkingClient()

        async def resolver():
            return replacement

        adapter = MCPToolAdapter(
            FakeMCPClient(),
            {"name": "remote_read", "inputSchema": {}},
            resolver,
        )

        result = await adapter.execute(path="README.md")

        self.assertTrue(result.success)
        self.assertEqual("remote_read:README.md", result.content)

    def test_provider_safe_name_is_stable_and_protocol_compatible(self):
        first = provider_safe_tool_name("server/with spaces", "工具/" + "x" * 100)
        second = provider_safe_tool_name("server/with spaces", "工具/" + "x" * 100)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 64)
        self.assertRegex(first, r"^[A-Za-z0-9_-]+$")


class MCPPromptAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_resource_and_prompt_adapters_are_read_only(self):
        client = FakePromptClient()

        self.assertEqual(ToolCategory.READ, MCPResourceAdapter(client).category)
        self.assertEqual(ToolCategory.READ, MCPPromptAdapter(client).category)

    async def test_prompt_arguments_must_be_json_object(self):
        client = FakePromptClient()
        adapter = MCPPromptAdapter(client)

        result = await adapter.execute(name="review", arguments='["not", "object"]')

        self.assertFalse(result.success)
        self.assertIn("JSON 对象", result.error)
        self.assertEqual([], client.prompt_calls)

    async def test_prompt_result_skips_malformed_messages(self):
        client = FakePromptClient()
        client.prompt_result = {
            "messages": [
                "bad-message",
                {"role": "user", "content": {"text": "Review this file"}},
                {"role": "assistant", "content": 123},
            ]
        }
        adapter = MCPPromptAdapter(client)

        result = await adapter.execute(name="review")

        self.assertTrue(result.success)
        self.assertEqual("[user]: Review this file", result.content)

    async def test_resource_uri_must_be_string_before_remote_call(self):
        client = FakePromptClient()
        adapter = MCPResourceAdapter(client)

        result = await adapter.execute(uri=123)

        self.assertFalse(result.success)
        self.assertIn("uri 必须是字符串", result.error)
        self.assertEqual([], client.resource_calls)

    async def test_prompt_name_and_arguments_must_be_strings_before_remote_call(self):
        client = FakePromptClient()
        adapter = MCPPromptAdapter(client)

        bad_name = await adapter.execute(name=123)
        bad_arguments = await adapter.execute(name="review", arguments={"topic": "code"})

        self.assertFalse(bad_name.success)
        self.assertFalse(bad_arguments.success)
        self.assertIn("name 必须是字符串", bad_name.error)
        self.assertIn("arguments 必须是字符串", bad_arguments.error)
        self.assertEqual([], client.prompt_calls)


if __name__ == "__main__":
    unittest.main()

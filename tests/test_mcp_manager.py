import unittest

from tinyCode.mcp.manager import MCPManager
from tinyCode.tools.registry import ToolRegistry


class FakeMCPPool:
    def __init__(self, clients: dict) -> None:
        self._clients = clients

    async def connect_all(self) -> dict:
        return self._clients

    async def get_client(self, name):
        return self._clients.get(name)


class FakeMCPClient:
    server_name = "fake"

    async def list_tools(self) -> list[dict]:
        return [
            {"description": "missing name"},
            {
                "name": "read",
                "description": "Read data",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]


class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_tool_definition_does_not_block_valid_adapters(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager._pool = FakeMCPPool({"fake": FakeMCPClient()})

        count = await manager.discover_and_register()

        self.assertEqual(3, count)
        self.assertIsNotNone(registry.get("mcp_fake_tool_read"))
        self.assertIsNotNone(registry.get("mcp_fake_resource"))
        self.assertIsNotNone(registry.get("mcp_fake_prompt"))


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest

from tinyCode.mcp.config import MCPServerConfig
from tinyCode.mcp.pool import MCPPool


class ReconnectingClient:
    def __init__(self) -> None:
        self.is_connected = False
        self.connect_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        await asyncio.sleep(0.01)
        self.is_connected = True


class MCPPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_callers_share_one_reconnect_attempt(self):
        pool = MCPPool([
            MCPServerConfig(
                name="docs",
                transport="http",
                url="https://example.invalid",
            )
        ])
        client = ReconnectingClient()
        pool._clients["docs"] = client

        first, second = await asyncio.gather(
            pool.get_client("docs"),
            pool.get_client("docs"),
        )

        self.assertIs(client, first)
        self.assertIs(client, second)
        self.assertEqual(1, client.connect_calls)


if __name__ == "__main__":
    unittest.main()

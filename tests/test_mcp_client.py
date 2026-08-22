import asyncio
import unittest

from tinyCode.mcp.client import MCPClient
from tinyCode.mcp.protocol import JSONRPCRequest, JSONRPCResponse
from tinyCode.mcp.transport.base import BaseTransport


class FakeTransport(BaseTransport):
    def __init__(self, responses: list[JSONRPCResponse]) -> None:
        self.responses = responses
        self.requests: list[JSONRPCRequest] = []
        self.notifications: list[str] = []
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        self.notifications.append(method)


class MCPClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_timeout_also_bounds_custom_transport(self):
        class HangingTransport(FakeTransport):
            async def send_request(self, request):
                await asyncio.Event().wait()

        client = MCPClient(HangingTransport([]), "hanging", timeout=0.01)

        with self.assertRaises(asyncio.TimeoutError):
            await client.list_tools()

    async def test_call_tool_raises_when_mcp_result_is_error(self):
        transport = FakeTransport([
            JSONRPCResponse(
                id=1,
                result={
                    "isError": True,
                    "content": [{"type": "text", "text": "tool failed"}],
                },
            )
        ])
        client = MCPClient(transport, "fake")

        with self.assertRaisesRegex(RuntimeError, "tool failed"):
            await client.call_tool("dangerous", {})

    async def test_call_tool_extracts_text_content(self):
        transport = FakeTransport([
            JSONRPCResponse(
                id=1,
                result={
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                },
            )
        ])
        client = MCPClient(transport, "fake")

        text = await client.call_tool("read", {"path": "README.md"})

        self.assertEqual("first\nsecond", text)
        self.assertEqual("tools/call", transport.requests[0].method)

    async def test_call_tool_skips_malformed_resource_content(self):
        transport = FakeTransport([
            JSONRPCResponse(
                id=1,
                result={
                    "content": [
                        {"type": "text", "text": "visible"},
                        {"type": "resource", "resource": "not an object"},
                    ],
                },
            )
        ])
        client = MCPClient(transport, "fake")

        text = await client.call_tool("read", {})

        self.assertEqual("visible", text)

    async def test_discovery_methods_return_only_dict_items_from_list_fields(self):
        transport = FakeTransport([
            JSONRPCResponse(
                id=1,
                result={"tools": [{"name": "read"}, "bad-tool", 7]},
            ),
            JSONRPCResponse(id=2, result={"resources": "not-a-list"}),
            JSONRPCResponse(id=3, result=[]),
        ])
        client = MCPClient(transport, "fake")

        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

        self.assertEqual([{"name": "read"}], tools)
        self.assertEqual([], resources)
        self.assertEqual([], prompts)

    async def test_call_and_read_reject_non_object_results(self):
        transport = FakeTransport([
            JSONRPCResponse(id=1, result=[]),
            JSONRPCResponse(id=2, result="bad"),
            JSONRPCResponse(id=3, result={"contents": "bad"}),
        ])
        client = MCPClient(transport, "fake")

        with self.assertRaisesRegex(RuntimeError, "无效 JSON-RPC result"):
            await client.call_tool("read", {})
        with self.assertRaisesRegex(RuntimeError, "无效 JSON-RPC result"):
            await client.get_prompt("template")
        with self.assertRaisesRegex(RuntimeError, "无效 contents"):
            await client.read_resource("file:///bad")

    async def test_call_tool_ignores_non_list_content(self):
        transport = FakeTransport([
            JSONRPCResponse(id=1, result={"content": "not-a-list"}),
        ])
        client = MCPClient(transport, "fake")

        self.assertEqual("", await client.call_tool("read", {}))


if __name__ == "__main__":
    unittest.main()

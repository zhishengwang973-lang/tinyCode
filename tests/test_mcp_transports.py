import asyncio
import json
import sys
import unittest
from unittest.mock import patch

from tinyCode.mcp.protocol import JSONRPCRequest
from tinyCode.mcp.transport.http import HttpTransport
from tinyCode.mcp.transport.stdio import StdioTransport


class MCPTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_http_url_does_not_duplicate_mcp_suffix(self):
        transport = HttpTransport("https://example.test/mcp")

        self.assertEqual("https://example.test/mcp", transport._url)

    async def test_http_sse_returns_on_matching_event_without_waiting_for_eof(self):
        class NeverEndingResponse:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                yield 'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
                await asyncio.Event().wait()

        class FakeClient:
            def stream(self, *args, **kwargs):
                return NeverEndingResponse()

        transport = HttpTransport("https://example.test/mcp", timeout=0.2)
        transport._client = FakeClient()

        response = await transport.send_request(JSONRPCRequest(method="ping"))

        self.assertEqual({"ok": True}, response.result)

    async def test_http_sse_heartbeat_without_response_obeys_total_timeout(self):
        class HeartbeatResponse:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                while True:
                    yield ": heartbeat"
                    await asyncio.sleep(0)

        class FakeClient:
            def stream(self, *args, **kwargs):
                return HeartbeatResponse()

        transport = HttpTransport("https://example.test/mcp", timeout=0.02)
        transport._client = FakeClient()

        with self.assertRaises(asyncio.TimeoutError):
            await transport.send_request(JSONRPCRequest(method="ping"))

    async def test_stdio_drains_large_stderr_while_waiting_for_response(self):
        script = (
            "import json,sys; "
            "req=json.loads(sys.stdin.readline()); "
            "sys.stderr.write('x'*200000); sys.stderr.flush(); "
            "print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'ok':True}}), flush=True)"
        )
        transport = StdioTransport(sys.executable, ["-c", script], timeout=2.0)
        await transport.connect()
        try:
            response = await transport.send_request(JSONRPCRequest(method="ping"))
        finally:
            await transport.disconnect()

        self.assertEqual({"ok": True}, response.result)

    async def test_stdio_eof_fails_pending_request_immediately(self):
        script = "import sys; sys.stdin.readline()"
        transport = StdioTransport(sys.executable, ["-c", script], timeout=10.0)
        await transport.connect()
        try:
            with self.assertRaisesRegex(ConnectionError, "closed stdout"):
                await asyncio.wait_for(
                    transport.send_request(JSONRPCRequest(method="ping")),
                    timeout=2.0,
                )
        finally:
            await transport.disconnect()

    async def test_stdio_rejects_oversized_request_before_write(self):
        transport = StdioTransport(sys.executable, ["-c", "import time; time.sleep(2)"])
        await transport.connect()
        try:
            with patch("tinyCode.mcp.transport.stdio.MAX_FRAME_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    await transport.send_request(JSONRPCRequest(
                        method="ping", params={"value": "large"},
                    ))
        finally:
            await transport.disconnect()


if __name__ == "__main__":
    unittest.main()

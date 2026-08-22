"""MCP client — handshake, discovery, tool/resource/prompt calls."""

import asyncio
from typing import Any

from tinyCode.mcp.protocol import (
    JSONRPCRequest,
    MCP_PROTOCOL_VERSION,
)
from tinyCode.mcp.transport.base import BaseTransport


MAX_EXTRACTED_TEXT_CHARS = 1_000_000


class MCPClient:
    """An MCP client that speaks JSON-RPC 2.0 over a transport.

    Lifecycle: connect → initialize → initialized → discover → call.
    """

    def __init__(self, transport: BaseTransport, server_name: str, timeout: float = 30.0) -> None:
        self._transport = transport
        self.server_name = server_name
        self._timeout = timeout
        self._server_info: dict = {}
        self._connected = False

    # -- lifecycle ------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self._transport.is_connected

    async def connect(self) -> None:
        """Establish connection and run MCP handshake."""
        try:
            await self._transport.connect()

            # 1. Initialize
            init_resp = await self._send("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "TinyCode", "version": "0.1.0"},
            })
            if init_resp.error:
                raise ConnectionError(f"Initialize failed: {init_resp.error}")
            self._server_info = init_resp.result or {}

            # 2. Send initialized notification
            await self._transport.send_notification("notifications/initialized")
            self._connected = True
        except BaseException:
            self._connected = False
            await self._transport.disconnect()
            raise

    async def disconnect(self) -> None:
        self._connected = False
        await self._transport.disconnect()

    # -- discovery ------------------------------------------------------------

    async def list_tools(self) -> list[dict]:
        resp = await self._send("tools/list")
        if resp.error:
            raise RuntimeError(f"tools/list failed: {resp.error}")
        return self._extract_discovery_items(resp.result, "tools")

    async def list_resources(self) -> list[dict]:
        resp = await self._send("resources/list")
        if resp.error:
            raise RuntimeError(f"resources/list failed: {resp.error}")
        return self._extract_discovery_items(resp.result, "resources")

    async def list_prompts(self) -> list[dict]:
        resp = await self._send("prompts/list")
        if resp.error:
            raise RuntimeError(f"prompts/list failed: {resp.error}")
        return self._extract_discovery_items(resp.result, "prompts")

    # -- tool call ------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the text content."""
        resp = await self._send("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if resp.error:
            raise RuntimeError(f"tools/call '{name}' failed: {resp.error}")
        result = self._require_result_object(resp.result, f"tools/call '{name}'")
        text = self._extract_text(result)
        if result.get("isError"):
            raise RuntimeError(text or f"tools/call '{name}' failed")
        return text

    # -- resource read --------------------------------------------------------

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI."""
        resp = await self._send("resources/read", {"uri": uri})
        if resp.error:
            raise RuntimeError(f"resources/read '{uri}' failed: {resp.error}")
        result = self._require_result_object(resp.result, f"resources/read '{uri}'")
        contents = result.get("contents", [])
        if not isinstance(contents, list):
            raise RuntimeError(f"resources/read '{uri}' 返回了无效 contents")
        return {
            "uri": uri,
            "text": self._extract_text_from_contents(contents),
            "contents": contents,
        }

    # -- prompt get -----------------------------------------------------------

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get a prompt template."""
        params: dict = {"name": name}
        if arguments:
            params["arguments"] = arguments
        resp = await self._send("prompts/get", params)
        if resp.error:
            raise RuntimeError(f"prompts/get '{name}' failed: {resp.error}")
        return self._require_result_object(resp.result, f"prompts/get '{name}'")

    # -- helpers --------------------------------------------------------------

    async def _send(self, method: str, params: dict | None = None):
        # Keep the client contract bounded even for custom transports that do
        # not enforce their own timeout correctly.
        return await asyncio.wait_for(
            self._transport.send_request(
                JSONRPCRequest(method=method, params=params)
            ),
            timeout=self._timeout,
        )

    @staticmethod
    def _require_result_object(result: Any, operation: str) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError(f"{operation} 返回了无效 JSON-RPC result（应为对象）")
        return result

    @staticmethod
    def _extract_discovery_items(result: Any, field: str) -> list[dict]:
        if not isinstance(result, dict):
            return []
        items = result.get(field, [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _extract_text(result: dict) -> str:
        content = result.get("content", [])
        if not isinstance(content, list):
            return ""
        return MCPClient._extract_text_from_contents(content)

    @staticmethod
    def _extract_text_from_contents(contents: list) -> str:
        parts: list[str] = []
        total = 0
        truncated = False
        for item in contents:
            if isinstance(item, dict):
                text = ""
                if item.get("type") == "text":
                    value = item.get("text", "")
                    text = value if isinstance(value, str) else ""
                elif item.get("type") == "resource":
                    resource = item.get("resource", {})
                    if isinstance(resource, dict):
                        value = resource.get("text", "")
                        text = value if isinstance(value, str) else ""
                if text:
                    remaining = MAX_EXTRACTED_TEXT_CHARS - total
                    if remaining <= 0:
                        truncated = True
                        break
                    parts.append(text[:remaining])
                    total += min(len(text), remaining)
                    if len(text) > remaining:
                        truncated = True
                        break
        if truncated:
            parts.append("[MCP 文本结果已截断]")
        return "\n".join(parts)

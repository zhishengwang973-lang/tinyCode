"""Streamable HTTP transport for MCP."""

import asyncio

import httpx

from tinyCode.mcp.protocol import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    decode_message,
    encode_message,
)
from tinyCode.mcp.transport.base import BaseTransport


MAX_RESPONSE_BYTES = 5_000_000


class HttpTransport(BaseTransport):
    """Communicates with an MCP server via Streamable HTTP (POST /mcp)."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        normalized = url.rstrip("/")
        self._url = normalized if normalized.endswith("/mcp") else normalized + "/mcp"
        self._headers = headers or {}
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._client is not None:
            await self.disconnect()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={**self._headers, "Content-Type": "application/json"},
        )
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Transport disconnected"))
        self._pending.clear()

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        if not self._client:
            raise ConnectionError("Transport not connected")
        client = self._client

        req_id = self._next_id
        self._next_id += 1
        request.id = req_id
        payload_bytes = encode_message(request).encode("utf-8")
        if len(payload_bytes) > MAX_RESPONSE_BYTES:
            raise ValueError("MCP HTTP request exceeds size limit")

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future

        async def exchange() -> JSONRPCResponse:
            """Keep connection setup, body streaming and response matching bounded."""
            try:
                async with client.stream(
                    "POST", self._url, content=payload_bytes,
                ) as resp:
                    if resp.status_code == 200:
                        # Response may be SSE or plain JSON.
                        content_type = resp.headers.get("content-type", "")
                        if "text/event-stream" in content_type:
                            received = 0
                            async for line in resp.aiter_lines():
                                received += len(line.encode("utf-8")) + 1
                                if received > MAX_RESPONSE_BYTES:
                                    raise ConnectionError(
                                        "MCP HTTP response exceeds size limit"
                                    )
                                if line.startswith("data:"):
                                    data_str = line[len("data:"):].lstrip()
                                    msg = decode_message(data_str)
                                    if isinstance(msg, JSONRPCResponse) and msg.id == req_id:
                                        future.set_result(msg)
                                        # Streamable HTTP servers may keep an
                                        # SSE connection open for later events.
                                        # The matching response completes this
                                        # request; do not wait for EOF.
                                        break
                        else:
                            payload = bytearray()
                            async for chunk in resp.aiter_bytes():
                                if len(payload) + len(chunk) > MAX_RESPONSE_BYTES:
                                    raise ConnectionError(
                                        "MCP HTTP response exceeds size limit"
                                    )
                                payload.extend(chunk)
                            text = payload.decode("utf-8", errors="replace")
                            msg = decode_message(text.strip())
                            if isinstance(msg, JSONRPCResponse) and msg.id == req_id:
                                future.set_result(msg)
                        if not future.done():
                            future.set_exception(ConnectionError(
                                "MCP server returned no matching JSON-RPC response"
                            ))
                    else:
                        payload = bytearray()
                        async for chunk in resp.aiter_bytes():
                            if len(payload) + len(chunk) > MAX_RESPONSE_BYTES:
                                raise ConnectionError(
                                    "MCP HTTP response exceeds size limit"
                                )
                            payload.extend(chunk)
                        text = payload.decode("utf-8", errors="replace")
                        future.set_exception(
                            ConnectionError(f"HTTP {resp.status_code}: {text[:500]}")
                        )
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            return await future

        try:
            # httpx's read timeout is an inactivity timeout. A broken server
            # can send heartbeats forever without returning our JSON-RPC id;
            # this outer deadline bounds the complete operation instead.
            return await asyncio.wait_for(exchange(), timeout=self._timeout)
        finally:
            self._pending.pop(req_id, None)

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        if not self._client:
            return
        notif = JSONRPCNotification(method=method, params=params)
        payload_bytes = encode_message(notif).encode("utf-8")
        if len(payload_bytes) > MAX_RESPONSE_BYTES:
            raise ValueError("MCP HTTP notification exceeds size limit")
        try:
            async with self._client.stream(
                "POST", self._url, content=payload_bytes,
            ):
                pass
        except Exception:
            pass  # Notifications are fire-and-forget

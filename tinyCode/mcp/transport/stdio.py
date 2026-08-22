"""Stdio transport — spawns a subprocess for MCP communication."""

import asyncio
import os

from tinyCode.mcp.protocol import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    decode_message,
    encode_message,
)
from tinyCode.mcp.transport.base import BaseTransport
from tinyCode.tools.run_command import _terminate_process


MAX_FRAME_BYTES = 5_000_000


class StdioTransport(BaseTransport):
    """Communicates with an MCP server via stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._command = command
        self._args = args or []
        self._env = env
        self._timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail = bytearray()

    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def connect(self) -> None:
        if self.is_connected:
            return
        if self._proc is not None:
            await self.disconnect()
        self._stderr_tail.clear()
        merged_env = os.environ.copy()
        if self._env:
            merged_env.update(self._env)

        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
            limit=MAX_FRAME_BYTES + 1,
            start_new_session=os.name == "posix",
        )
        self._reader_task = asyncio.ensure_future(self._read_loop())
        self._stderr_task = asyncio.ensure_future(self._drain_stderr())

    async def disconnect(self) -> None:
        proc = self._proc
        self._proc = None
        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                await _terminate_process(proc)

        background = [
            task for task in (self._reader_task, self._stderr_task)
            if task is not None
        ]
        self._reader_task = None
        self._stderr_task = None
        if (
            os.name == "posix"
            and proc is not None
            and any(not task.done() for task in background)
        ):
            # The direct server can exit while a descendant keeps inherited
            # stdout/stderr descriptors open. Clean the process group before
            # cancelling our drain tasks so those descendants do not leak.
            await _terminate_process(proc)
        for task in background:
            if not task.done():
                task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        self._fail_pending(ConnectionError("Transport disconnected"))

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        if not self._proc or self._proc.stdin is None:
            raise ConnectionError("Transport not connected")

        req_id = self._next_id
        self._next_id += 1
        request.id = req_id

        line = encode_message(request) + "\n"
        encoded = line.encode("utf-8")
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("MCP request frame exceeds size limit")

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future

        async def send_and_wait() -> JSONRPCResponse:
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(encoded)
            await self._proc.stdin.drain()
            return await future

        try:
            # Include a blocked stdin drain in the request deadline. Waiting
            # only on the response future lets an unresponsive child that
            # stopped reading stdin hang this call indefinitely.
            return await asyncio.wait_for(send_and_wait(), timeout=self._timeout)
        finally:
            self._pending.pop(req_id, None)

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        if not self._proc or self._proc.stdin is None:
            return
        notif = JSONRPCNotification(method=method, params=params)
        line = encode_message(notif) + "\n"
        encoded = line.encode("utf-8")
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("MCP notification frame exceeds size limit")
        self._proc.stdin.write(encoded)
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=self._timeout)

    async def _read_loop(self) -> None:
        """Read lines from stdout and dispatch responses."""
        proc = self._proc
        assert proc and proc.stdout
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break  # EOF
                if len(line) > MAX_FRAME_BYTES:
                    raise ConnectionError("MCP response frame exceeds size limit")
                msg = decode_message(line.decode("utf-8").strip())
                if (
                    isinstance(msg, JSONRPCResponse)
                    and isinstance(msg.id, int)
                    and msg.id in self._pending
                ):
                    fut = self._pending[msg.id]
                    if not fut.done():
                        fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(ConnectionError(f"MCP stdout reader failed: {exc}"))
        else:
            detail = self._stderr_tail.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            self._fail_pending(ConnectionError(f"MCP server closed stdout{suffix}"))

    async def _drain_stderr(self) -> None:
        """Continuously drain stderr so a verbose server cannot deadlock."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    return
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > 8192:
                    del self._stderr_tail[:-8192]
        except asyncio.CancelledError:
            raise

    def _fail_pending(self, exc: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

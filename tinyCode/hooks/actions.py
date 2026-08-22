"""Action executors — shell, prompt_inject, http, sub_agent."""

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable

import httpx

from tinyCode.hooks.models import Action, ActionType
from tinyCode.hooks.templates import TemplateEngine
from tinyCode.tools.run_command import _read_limited, _terminate_process


class ActionExecutor:
    """Execute hook actions with error isolation."""

    def __init__(self, template_engine: TemplateEngine | None = None) -> None:
        self._templates = template_engine or TemplateEngine()
        self._sub_agent_handler: Callable[[str], Awaitable[str]] | None = None

    def set_sub_agent_handler(
        self, handler: Callable[[str], Awaitable[str]] | None,
    ) -> None:
        self._sub_agent_handler = handler

    async def execute(
        self,
        action: Action,
        context: dict,
        timeout: float = 30.0,
    ) -> str | None:
        """Execute a single action.  Returns the result text, or None.

        Errors are logged to stderr but never raised.
        """
        try:
            if action.type == ActionType.SHELL:
                return await self._exec_shell(action, context, timeout)
            elif action.type == ActionType.PROMPT_INJECT:
                return self._exec_prompt_inject(action, context)
            elif action.type == ActionType.HTTP:
                return await self._exec_http(action, context, timeout)
            elif action.type == ActionType.SUB_AGENT:
                return await asyncio.wait_for(
                    self._exec_sub_agent(action, context), timeout=timeout,
                )
        except asyncio.TimeoutError:
            print(
                f"Hook action [{action.type.value}]: 执行超时（{timeout:g}s）",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"Hook action [{action.type.value}]: 执行失败 — {exc}", file=sys.stderr)
        return None

    # -- internals -----------------------------------------------------------

    async def _exec_shell(self, action: Action, context: dict, timeout: float) -> str:
        cmd = self._templates.render(action.command, context)
        proc: asyncio.subprocess.Process | None = None
        try:
            if os.name == "posix":
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            _, stdout_data, stderr_data = await asyncio.wait_for(
                asyncio.gather(
                    proc.wait(),
                    _read_limited(proc.stdout, byte_limit=8000),
                    _read_limited(proc.stderr, byte_limit=8000),
                ),
                timeout=timeout,
            )
            stdout, _ = stdout_data
            stderr, _ = stderr_data
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            result = out
            if err:
                result += f"\n[stderr]\n{err}"
            return result[:2000]
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            return f"(超时 {timeout}s)"
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise

    def _exec_prompt_inject(self, action: Action, context: dict) -> str:
        return self._templates.render(action.text, context)

    async def _exec_http(self, action: Action, context: dict, timeout: float) -> str:
        url = self._templates.render(action.url, context)
        body = self._templates.render(action.body, context) if action.body else None
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            async with client.stream(
                action.method, url, content=body, headers=action.headers,
            ) as resp:
                payload = bytearray()
                async for chunk in resp.aiter_bytes():
                    remaining = 8_000 - len(payload)
                    if remaining <= 0:
                        break
                    payload.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        break
                return payload.decode("utf-8", errors="replace")[:2000]

    async def _exec_sub_agent(self, action: Action, context: dict) -> str:
        if self._sub_agent_handler is None:
            raise RuntimeError("Hook sub_agent handler 未配置")
        task = self._templates.render(action.task, context)
        if not task.strip():
            raise ValueError("Hook sub_agent task 不能为空")
        return await self._sub_agent_handler(task)

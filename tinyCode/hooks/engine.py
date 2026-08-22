"""Hook engine — fires events at lifecycle points in the Agent Loop."""

import asyncio
import sys
from typing import Any

from tinyCode.hooks.actions import ActionExecutor
from tinyCode.hooks.conditions import ConditionEvaluator
from tinyCode.hooks.models import HookEvent, Rule
from tinyCode.hooks.templates import TemplateEngine


class HookEngine:
    """Registers hook rules and fires events at lifecycle nodes."""

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules
        self._conditions = ConditionEvaluator()
        self._templates = TemplateEngine()
        self._actions = ActionExecutor(self._templates)
        self._fired_once: set[str] = set()  # rule names that fired (for once:true)
        self._background_tasks: set[asyncio.Task] = set()

    # -- public API -----------------------------------------------------------

    async def fire(
        self,
        event: HookEvent,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Fire all matching rules for *event*.

        For TOOL_PRE_EXEC, returns the first rejection reason (or None if all
        allowed).  The caller should feed the rejection back to the LLM as a
        tool result.

        For non-intercept prompt injections, returns all rendered texts joined
        in rule order. Other action results are intentionally not injected.
        """
        ctx = dict(context or {})
        ctx["_event"] = event.value

        prompt_results: list[str] = []
        for rule in self._rules:
            if rule.event != event:
                continue

            # once-only
            if rule.control.once and rule.name in self._fired_once:
                continue

            # Evaluate condition
            if not self._conditions.evaluate(rule.condition, ctx):
                continue

            # Mark fired
            if rule.control.once:
                self._fired_once.add(rule.name)

            # Execute actions
            if rule.control.async_:
                for action in rule.actions:
                    self._spawn(
                        self._actions.execute(action, ctx, timeout=rule.control.timeout)
                    )
            else:
                for action in rule.actions:
                    result = await self._actions.execute(
                        action, ctx, timeout=rule.control.timeout,
                    )
                    # For intercept events, collect rejection from prompt_inject
                    if rule.is_intercept and action.type.value == "prompt_inject" and result:
                        return result
                    if action.type.value == "prompt_inject" and result:
                        prompt_results.append(result)

        return "\n\n".join(prompt_results) or None

    def set_sub_agent_handler(self, handler) -> None:
        self._actions.set_sub_agent_handler(handler)

    def fire_sync(
        self,
        event: HookEvent,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Synchronous helper — runs fire() inside an event loop if needed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in event loop — schedule and wait
                self._spawn(self.fire(event, context))
                # Can't block here — just fire and forget
                return None
            return asyncio.run(self.fire(event, context))
        except RuntimeError:
            return asyncio.run(self.fire(event, context))

    def reset_once(self) -> None:
        """Clear once-fired tracking (e.g., on new session)."""
        self._fired_once.clear()

    async def shutdown(self, *, cancel: bool = True) -> None:
        """Join asynchronous hook actions, optionally cancelling them first."""
        tasks = list(self._background_tasks)
        self._background_tasks.clear()
        if cancel:
            for task in tasks:
                if not task.done():
                    task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _spawn(self, awaitable) -> asyncio.Task:
        task = asyncio.ensure_future(awaitable)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

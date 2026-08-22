import asyncio
import unittest

from tinyCode.hooks.engine import HookEngine
from tinyCode.hooks.models import Action, ActionType, Control, HookEvent, Rule


class HookEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_pre_exec_returns_first_intercept_reason(self):
        engine = HookEngine([
            Rule(
                event=HookEvent.TOOL_PRE_EXEC,
                actions=[Action(type=ActionType.PROMPT_INJECT, text="first block")],
                name="first",
            ),
            Rule(
                event=HookEvent.TOOL_PRE_EXEC,
                actions=[Action(type=ActionType.PROMPT_INJECT, text="second block")],
                name="second",
            ),
        ])

        reason = await engine.fire(HookEvent.TOOL_PRE_EXEC, {"tool_name": "write_file"})

        self.assertEqual("first block", reason)

    async def test_non_intercept_prompt_injections_are_aggregated(self):
        engine = HookEngine([
            Rule(
                event=HookEvent.MESSAGE_PRE_SEND,
                actions=[Action(type=ActionType.PROMPT_INJECT, text="first")],
            ),
            Rule(
                event=HookEvent.MESSAGE_PRE_SEND,
                actions=[Action(type=ActionType.PROMPT_INJECT, text="second")],
            ),
        ])

        result = await engine.fire(HookEvent.MESSAGE_PRE_SEND, {})

        self.assertEqual("first\n\nsecond", result)

    async def test_sub_agent_action_invokes_configured_handler(self):
        calls = []

        async def handler(task):
            calls.append(task)
            return "queued"

        engine = HookEngine([
            Rule(
                event=HookEvent.SYSTEM_ERROR,
                actions=[Action(type=ActionType.SUB_AGENT, task="inspect {{error}}")],
            ),
        ])
        engine.set_sub_agent_handler(handler)

        await engine.fire(HookEvent.SYSTEM_ERROR, {"error": "boom"})

        self.assertEqual(["inspect boom"], calls)

    async def test_sub_agent_action_obeys_hook_timeout(self):
        cancelled = asyncio.Event()

        async def handler(task):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        engine = HookEngine([
            Rule(
                event=HookEvent.SYSTEM_ERROR,
                actions=[Action(type=ActionType.SUB_AGENT, task="inspect")],
                control=Control(timeout=0.01),
            ),
        ])
        engine.set_sub_agent_handler(handler)

        await engine.fire(HookEvent.SYSTEM_ERROR, {"error": "boom"})

        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest

from tinyCode.agent.events import (
    AgentDoneEvent,
    ErrorEvent,
    HITLRequestEvent,
    RoundLimitDecision,
    RoundLimitDecisionAction,
    RoundLimitReachedEvent,
    TextDeltaEvent,
)
from tinyCode.agent.runtime import TurnRuntime, TurnState
from tinyCode.security.models import HITLDecision


class FakeLoop:
    def __init__(self, events=None, error: Exception | None = None) -> None:
        self.events = list(events or [])
        self.error = error
        self.cancelled = False

    async def run(self, history):
        if self.error:
            raise self.error
        for event in self.events:
            yield event

    def cancel(self) -> None:
        self.cancelled = True


class WaitingLoop(FakeLoop):
    def __init__(self) -> None:
        super().__init__()
        self.future: asyncio.Future | None = None

    async def run(self, history):
        self.future = asyncio.get_running_loop().create_future()
        yield HITLRequestEvent("run_command", {}, "confirm", self.future)
        decision = await self.future
        yield TextDeltaEvent(decision.value)
        yield AgentDoneEvent("no_tool_call")


class StallingThenSuccessLoop(FakeLoop):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.started = asyncio.Event()

    async def run(self, history):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        yield AgentDoneEvent("no_tool_call")


class WaitingForRoundLimitLoop(FakeLoop):
    def __init__(self) -> None:
        super().__init__()
        self.future: asyncio.Future | None = None

    async def run(self, history):
        self.future = asyncio.get_running_loop().create_future()
        yield RoundLimitReachedEvent(1, 1, 10, 100, False, self.future)
        decision = await self.future
        reason = (
            "round_budget_stopped"
            if decision.action == RoundLimitDecisionAction.STOP
            else "no_tool_call"
        )
        yield AgentDoneEvent(reason)


class TurnRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_owner_and_normal_completion(self):
        runtime = TurnRuntime(FakeLoop([
            TextDeltaEvent("ok"),
            AgentDoneEvent("no_tool_call"),
        ]))
        self.assertTrue(runtime.reserve())
        self.assertFalse(runtime.reserve())
        self.assertTrue(runtime.claim())

        events = [event async for event in runtime.run(object())]

        self.assertEqual(2, len(events))
        self.assertFalse(runtime.active)
        self.assertEqual(TurnState.COMPLETED, runtime.snapshot().last_outcome)

    async def test_approval_is_owned_and_resolved_by_runtime(self):
        loop = WaitingLoop()
        runtime = TurnRuntime(loop)

        async def consume():
            self.assertTrue(runtime.reserve())
            self.assertTrue(runtime.claim())
            return [event async for event in runtime.run(object())]

        task = asyncio.create_task(consume())
        for _ in range(100):
            if runtime.waiting_for_approval:
                break
            await asyncio.sleep(0)

        self.assertTrue(runtime.waiting_for_approval)
        self.assertTrue(runtime.resolve_approval(HITLDecision.ALLOW_ONCE))
        events = await task

        self.assertEqual("allow_once", events[1].text)
        self.assertFalse(runtime.waiting_for_approval)
        self.assertEqual(TurnState.COMPLETED, runtime.snapshot().last_outcome)

    async def test_cancel_interrupts_owner_and_cleans_approval(self):
        loop = WaitingLoop()
        runtime = TurnRuntime(loop)

        async def consume():
            runtime.reserve()
            runtime.claim()
            return [event async for event in runtime.run(object())]

        task = asyncio.create_task(consume())
        for _ in range(100):
            if runtime.waiting_for_approval:
                break
            await asyncio.sleep(0)

        self.assertTrue(runtime.cancel())
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(loop.cancelled)
        self.assertFalse(runtime.active)
        self.assertFalse(runtime.waiting_for_approval)
        self.assertEqual(TurnState.CANCELLED, runtime.snapshot().last_outcome)

    async def test_round_limit_decision_is_owned_and_resolved_by_runtime(self):
        loop = WaitingForRoundLimitLoop()
        runtime = TurnRuntime(loop)

        async def consume():
            runtime.reserve()
            runtime.claim()
            return [event async for event in runtime.run(object())]

        task = asyncio.create_task(consume())
        for _ in range(100):
            if runtime.waiting_for_round_limit:
                break
            await asyncio.sleep(0)

        self.assertTrue(runtime.waiting_for_round_limit)
        self.assertTrue(runtime.resolve_round_limit(RoundLimitDecision(
            RoundLimitDecisionAction.EXTEND,
        )))
        await task

        self.assertFalse(runtime.waiting_for_round_limit)
        self.assertEqual(TurnState.COMPLETED, runtime.snapshot().last_outcome)

    async def test_stopping_at_soft_budget_is_paused_not_completed(self):
        loop = WaitingForRoundLimitLoop()
        runtime = TurnRuntime(loop)

        async def consume():
            runtime.reserve()
            runtime.claim()
            return [event async for event in runtime.run(object())]

        task = asyncio.create_task(consume())
        for _ in range(100):
            if runtime.waiting_for_round_limit:
                break
            await asyncio.sleep(0)
        runtime.resolve_round_limit(RoundLimitDecision(
            RoundLimitDecisionAction.STOP,
        ))
        await task

        self.assertEqual(TurnState.PAUSED, runtime.snapshot().last_outcome)

    async def test_hard_round_limit_has_distinct_terminal_state(self):
        runtime = TurnRuntime(FakeLoop([AgentDoneEvent("hard_max_rounds")]))
        runtime.reserve()
        runtime.claim()

        await self._collect(runtime)

        self.assertEqual(TurnState.LIMIT_REACHED, runtime.snapshot().last_outcome)

    @staticmethod
    async def _collect(runtime: TurnRuntime):
        return [event async for event in runtime.run(object())]

    async def test_exception_becomes_error_event_and_next_turn_can_start(self):
        runtime = TurnRuntime(FakeLoop(error=RuntimeError("disconnected")))
        runtime.reserve()
        runtime.claim()

        events = [event async for event in runtime.run(object())]

        self.assertEqual(1, len(events))
        self.assertIsInstance(events[0], ErrorEvent)
        self.assertIn("disconnected", events[0].message)
        self.assertEqual(TurnState.FAILED, runtime.snapshot().last_outcome)
        self.assertTrue(runtime.reserve())

    async def test_missing_terminal_event_is_reported(self):
        runtime = TurnRuntime(FakeLoop([TextDeltaEvent("partial")]))
        runtime.reserve()
        runtime.claim()

        events = [event async for event in runtime.run(object())]

        self.assertIsInstance(events[-1], ErrorEvent)
        self.assertIn("未收到终止事件", events[-1].message)
        self.assertEqual(TurnState.FAILED, runtime.snapshot().last_outcome)

    async def test_cancel_stalled_stream_then_start_next_turn(self):
        loop = StallingThenSuccessLoop()
        runtime = TurnRuntime(loop)

        async def consume():
            runtime.reserve()
            runtime.claim()
            return [event async for event in runtime.run(object())]

        first = asyncio.create_task(consume())
        await loop.started.wait()
        self.assertTrue(runtime.cancel())
        with self.assertRaises(asyncio.CancelledError):
            await first

        self.assertFalse(runtime.active)
        second_events = await consume()

        self.assertEqual(2, loop.calls)
        self.assertIsInstance(second_events[-1], AgentDoneEvent)
        self.assertEqual(TurnState.COMPLETED, runtime.snapshot().last_outcome)


if __name__ == "__main__":
    unittest.main()

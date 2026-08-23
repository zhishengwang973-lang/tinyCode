"""Single-owner runtime for one foreground conversation turn."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import TYPE_CHECKING

from tinyCode.agent.events import (
    AgentDoneEvent,
    AgentEvent,
    ErrorEvent,
    HITLRequestEvent,
    RoundLimitDecision,
    RoundLimitDecisionAction,
    RoundLimitReachedEvent,
)
from tinyCode.security.models import HITLDecision

if TYPE_CHECKING:
    from tinyCode.agent.loop import AgentLoop
    from tinyCode.conversation.history import ConversationHistory


class TurnState(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_ROUND_LIMIT = "waiting_round_limit"
    COMPLETED = "completed"
    PAUSED = "paused"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TurnSnapshot:
    turn_id: int
    state: TurnState
    active: bool
    started_at: float | None
    last_outcome: TurnState | None
    last_error: str


class TurnRuntime:
    """Owns turn lifecycle, cancellation, and the active approval future."""

    _ACTIVE_STATES = {
        TurnState.PREPARING,
        TurnState.RUNNING,
        TurnState.WAITING_APPROVAL,
        TurnState.WAITING_ROUND_LIMIT,
    }

    def __init__(self, agent_loop: "AgentLoop") -> None:
        self._agent_loop = agent_loop
        self._state = TurnState.IDLE
        self._turn_id = 0
        self._started_at: float | None = None
        self._last_outcome: TurnState | None = None
        self._last_error = ""
        self._owner_task: asyncio.Task | None = None
        self._approval_future: asyncio.Future | None = None
        self._round_limit_future: asyncio.Future | None = None

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state in self._ACTIVE_STATES or self._owner_task is not None

    @property
    def waiting_for_approval(self) -> bool:
        return (
            self._state == TurnState.WAITING_APPROVAL
            and self._approval_future is not None
            and not self._approval_future.done()
        )

    @property
    def approval_future(self) -> asyncio.Future | None:
        """Read-only compatibility hook for UI/tests; runtime owns mutation."""
        return self._approval_future

    @property
    def waiting_for_round_limit(self) -> bool:
        return (
            self._state == TurnState.WAITING_ROUND_LIMIT
            and self._round_limit_future is not None
            and not self._round_limit_future.done()
        )

    def snapshot(self) -> TurnSnapshot:
        return TurnSnapshot(
            turn_id=self._turn_id,
            state=self._state,
            active=self.active,
            started_at=self._started_at,
            last_outcome=self._last_outcome,
            last_error=self._last_error,
        )

    def reserve(self) -> bool:
        """Synchronously reserve the foreground slot before task scheduling."""
        if self.active:
            return False
        self._turn_id += 1
        self._state = TurnState.PREPARING
        self._started_at = monotonic()
        self._last_error = ""
        self._owner_task = None
        self._approval_future = None
        self._round_limit_future = None
        return True

    def claim(self) -> bool:
        """Attach the currently running asyncio task to a reservation."""
        task = asyncio.current_task()
        if task is None or self._state != TurnState.PREPARING:
            return False
        if self._owner_task is not None and self._owner_task is not task:
            return False
        self._owner_task = task
        return True

    async def run(
        self,
        history: "ConversationHistory",
    ) -> AsyncIterator[AgentEvent]:
        """Consume one AgentLoop stream and enforce an explicit terminal state."""
        task = asyncio.current_task()
        if self._state != TurnState.PREPARING or self._owner_task is not task:
            raise RuntimeError("当前任务未持有对话轮次")

        outcome: TurnState | None = None
        self._state = TurnState.RUNNING
        try:
            async for event in self._agent_loop.run(history):
                if isinstance(event, HITLRequestEvent):
                    if not isinstance(event.future, asyncio.Future):
                        raise RuntimeError("HITLRequestEvent.future 必须是 asyncio.Future")
                    self._approval_future = event.future
                    self._state = TurnState.WAITING_APPROVAL
                elif isinstance(event, RoundLimitReachedEvent):
                    if not isinstance(event.future, asyncio.Future):
                        raise RuntimeError(
                            "RoundLimitReachedEvent.future 必须是 asyncio.Future"
                        )
                    self._round_limit_future = event.future
                    self._state = TurnState.WAITING_ROUND_LIMIT
                elif isinstance(event, AgentDoneEvent):
                    if event.reason == "cancelled":
                        outcome = TurnState.CANCELLED
                    elif event.reason == "round_budget_stopped":
                        outcome = TurnState.PAUSED
                    elif event.reason == "hard_max_rounds":
                        outcome = TurnState.LIMIT_REACHED
                    else:
                        outcome = TurnState.COMPLETED
                elif isinstance(event, ErrorEvent):
                    outcome = TurnState.FAILED
                    self._last_error = event.message

                if isinstance(event, (AgentDoneEvent, ErrorEvent)):
                    assert outcome is not None
                    self._finish(outcome)
                    yield event
                    return

                yield event

            if outcome is None:
                outcome = TurnState.FAILED
                self._last_error = "Agent 事件流意外结束，未收到终止事件"
                self._finish(outcome)
                yield ErrorEvent(message=self._last_error)
                return
        except asyncio.CancelledError:
            outcome = TurnState.CANCELLED
            self._agent_loop.cancel()
            self._deny_pending_approval()
            raise
        except Exception as exc:
            outcome = TurnState.FAILED
            self._last_error = f"对话运行时失败: {type(exc).__name__}: {exc}"
            self._finish(outcome)
            yield ErrorEvent(message=self._last_error)
            return
        finally:
            if self._owner_task is task or self.active:
                self._finish(outcome or TurnState.FAILED)

    def resolve_approval(self, decision: HITLDecision) -> bool:
        future = self._approval_future
        if not self.waiting_for_approval or future is None:
            return False
        future.set_result(decision)
        self._approval_future = None
        self._state = TurnState.RUNNING
        return True

    def resolve_round_limit(self, decision: RoundLimitDecision) -> bool:
        future = self._round_limit_future
        if not self.waiting_for_round_limit or future is None:
            return False
        future.set_result(decision)
        self._round_limit_future = None
        self._state = TurnState.RUNNING
        return True

    def cancel(self, *, interrupt: bool = True) -> bool:
        if not self.active:
            return False
        self._agent_loop.cancel()
        self._deny_pending_approval()
        self._stop_pending_round_limit()
        self._state = TurnState.CANCELLED
        owner = self._owner_task
        if interrupt and owner is not None and owner is not asyncio.current_task():
            owner.cancel()
        elif owner is None:
            self._finish(TurnState.CANCELLED)
        return True

    def fail_preparation(self, message: str) -> None:
        if self._state == TurnState.PREPARING:
            self._last_error = message
            self._finish(TurnState.FAILED)

    def release(self) -> None:
        """Release a reservation if execution ended before the event stream."""
        if self._state == TurnState.PREPARING:
            self._finish(TurnState.FAILED)
        elif self._state == TurnState.CANCELLED:
            self._finish(TurnState.CANCELLED)

    def _deny_pending_approval(self) -> None:
        future = self._approval_future
        if future is not None and not future.done():
            future.set_result(HITLDecision.DENY)
        self._approval_future = None

    def _stop_pending_round_limit(self) -> None:
        future = self._round_limit_future
        if future is not None and not future.done():
            future.set_result(RoundLimitDecision(RoundLimitDecisionAction.STOP))
        self._round_limit_future = None

    def _finish(self, outcome: TurnState) -> None:
        self._deny_pending_approval()
        self._stop_pending_round_limit()
        self._last_outcome = outcome
        self._state = TurnState.IDLE
        self._started_at = None
        self._owner_task = None

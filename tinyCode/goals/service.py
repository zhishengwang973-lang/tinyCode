"""Goal lifecycle, evidence accounting, and continuation decisions."""

from __future__ import annotations

from tinyCode.goals.models import Goal, GoalStatus
from tinyCode.goals.store import GoalStore
from tinyCode.time_utils import beijing_now_iso


class GoalService:
    """Owns one persisted Goal for the currently selected conversation."""

    def __init__(self, store: GoalStore, *, default_max_turns: int) -> None:
        self._store = store
        self._default_max_turns = max(1, default_max_turns)
        self._session_id = ""
        self._goal: Goal | None = None

    @property
    def current(self) -> Goal | None:
        return self._goal

    @property
    def active(self) -> bool:
        return self._goal is not None and self._goal.status is GoalStatus.ACTIVE

    def bind_session(self, session_id: str) -> Goal | None:
        self._session_id = session_id
        self._goal = self._store.load(session_id) if session_id else None
        return self._goal

    def start(self, objective: str, *, max_turns: int | None = None) -> Goal:
        if not self._session_id:
            raise RuntimeError("当前会话尚未初始化")
        objective = objective.strip()
        if not objective:
            raise ValueError("Goal 目标不能为空")
        goal = Goal.create(
            self._session_id, objective,
            max_turns=max_turns or self._default_max_turns,
        )
        self._goal = goal
        self._save()
        return goal

    def pause(self, reason: str = "用户暂停") -> Goal | None:
        if not self.active:
            return None
        assert self._goal is not None
        self._goal.status = GoalStatus.PAUSED
        self._goal.last_reason = reason
        self._save()
        return self._goal

    def resume(self, *, additional_turns: int = 0) -> Goal | None:
        goal = self._goal
        if goal is None:
            return None
        if goal.status is GoalStatus.COMPLETED:
            raise ValueError("Goal 已完成；如需新目标，请使用 /goal <目标>")
        if additional_turns < 0:
            raise ValueError("新增预算不能小于 0")
        if goal.status is GoalStatus.BUDGET_LIMITED and additional_turns <= 0:
            raise ValueError("预算已用尽，请使用 /goal resume <新增回合数>")
        if additional_turns:
            goal.max_turns += additional_turns
        goal.status = GoalStatus.ACTIVE
        goal.last_reason = "用户恢复 Goal"
        self._save()
        return goal

    def clear(self) -> Goal | None:
        goal = self._goal
        if goal is None:
            return None
        self._store.clear(goal.session_id)
        self._goal = None
        return goal

    def complete(self, evidence: str) -> str:
        goal = self._goal
        if not self.active or goal is None:
            return "当前没有进行中的 Goal"
        evidence = evidence.strip()
        if len(evidence) < 8:
            return "完成 Goal 必须提供可复核的证据摘要（至少 8 个字符）"
        goal.status = GoalStatus.COMPLETED
        goal.completion_evidence = evidence[:4_000]
        goal.last_reason = "模型已提交完成证据"
        self._save()
        return "Goal 已标记完成"

    def complete_from_final_response(self, response: str) -> Goal | None:
        """Close a Goal when the agent has delivered a normal final answer.

        ``AgentDoneEvent(no_tool_call)`` is the runtime's terminal answer
        boundary. Models occasionally omit ``goal_complete``, especially for
        text-only research or explanation Goals. Keeping such a Goal active is
        misleading and can invite a useless continuation, so the final answer
        itself is retained as the delivery evidence.
        """
        goal = self._goal
        response = response.strip()
        if (
            goal is None
            or not response
            or goal.status not in {GoalStatus.ACTIVE, GoalStatus.BUDGET_LIMITED}
        ):
            return None
        goal.status = GoalStatus.COMPLETED
        goal.completion_evidence = (
            "模型未显式调用 goal_complete；以下最终交付已自动归档：\n"
            + response[:4_000]
        )
        goal.last_reason = "模型输出最终回答后自动收束"
        self._save()
        return goal

    def record_turn(
        self, *, model_requests: int, tool_calls: int, reason: str,
    ) -> Goal | None:
        goal = self._goal
        if goal is None:
            return None
        goal.completed_turns += 1
        goal.model_requests += max(0, model_requests)
        goal.tool_calls += max(0, tool_calls)
        goal.last_reason = reason
        if goal.status is GoalStatus.ACTIVE and reason in {
            "cancelled", "stalled", "round_budget_stopped", "hard_max_rounds",
        }:
            goal.status = GoalStatus.PAUSED
            goal.last_reason = "本次执行已暂停；使用 /goal resume 继续"
        elif goal.status is GoalStatus.ACTIVE and goal.completed_turns >= goal.max_turns:
            goal.status = GoalStatus.BUDGET_LIMITED
            goal.last_reason = "已达到 Goal 执行预算；可使用 /goal resume <新增回合数> 继续"
        self._save()
        return goal

    def should_continue(
        self, *, tool_calls: int, terminal_reason: str, has_pending_user_input: bool,
    ) -> bool:
        """Conservative event-boundary continuation policy.

        A no-tool turn may be a final answer or a failed plan.  Continuing it
        blindly creates a spinner loop, so only a productive tool turn can
        trigger the next automatic Goal turn.
        """
        return bool(
            self.active
            and terminal_reason == "no_tool_call"
            and tool_calls > 0
            and not has_pending_user_input
        )

    def render(self) -> str:
        return self._goal.render() if self._goal is not None else "当前会话没有 Goal"

    def prompt_context(self) -> str:
        return self._goal.prompt_context() if self.active and self._goal else ""

    def _save(self) -> None:
        assert self._goal is not None
        self._goal.updated_at = beijing_now_iso()
        self._store.save(self._goal)

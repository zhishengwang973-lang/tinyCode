"""Multi-signal progress watchdog for ReAct tasks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum

from tinyCode.providers.base import ToolCall
from tinyCode.tools.base import ToolResult


_WORKSPACE_TOOLS = {"write_file", "edit_file", "apply_patch", "delete_file"}
_TEST_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:pytest|py\.test|unittest|npm\s+test|pnpm\s+test|"
    r"yarn\s+test|cargo\s+test|go\s+test)(?:\s|$)",
    re.IGNORECASE,
)
_FAILED_RE = re.compile(r"\b(\d+)\s+failed\b", re.IGNORECASE)
_UNITTEST_FAILURE_RE = re.compile(r"failures=(\d+)", re.IGNORECASE)
_UNITTEST_ERROR_RE = re.compile(r"errors=(\d+)", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"超时|timed?\s*out|timeout", re.IGNORECASE)


class ProgressState(Enum):
    PROGRESSING = "progressing"
    SLOW = "slow"
    STALLED = "stalled"
    OSCILLATING = "oscillating"
    HARD_STUCK = "hard_stuck"


@dataclass(frozen=True)
class RoundProgressSnapshot:
    round_number: int
    action_signature: str
    result_signature: str
    state_signature: str
    error_signature: str
    workspace_changed: bool
    new_information: bool
    test_failures: int | None
    test_improved: bool
    timed_out: bool

    @property
    def effective_progress(self) -> bool:
        return self.workspace_changed or self.new_information or self.test_improved


@dataclass(frozen=True)
class ProgressAssessment:
    state: ProgressState
    reasons: tuple[str, ...] = ()
    recovery_prompt: str = ""

    @property
    def requires_intervention(self) -> bool:
        return self.state in {
            ProgressState.STALLED,
            ProgressState.OSCILLATING,
            ProgressState.HARD_STUCK,
        }


class ProgressWatchdog:
    """Classify task progress from observable round-level evidence.

    The model's prose is deliberately excluded: only tool calls, tool results,
    successful workspace mutations, test outcomes, and timeout/error signals
    count as evidence.
    """

    def __init__(self, *, history_size: int = 6) -> None:
        self._history: deque[RoundProgressSnapshot] = deque(maxlen=history_size)
        self._seen_results: set[str] = set()
        self._seen_mutations: set[str] = set()
        self._last_test_failures: int | None = None

    @property
    def history(self) -> tuple[RoundProgressSnapshot, ...]:
        return tuple(self._history)

    def reset_strategy(self) -> None:
        """Start a fresh observation window after an explicit strategy change."""
        self._history.clear()
        self._seen_results.clear()
        self._seen_mutations.clear()
        self._last_test_failures = None

    def observe(
        self,
        round_number: int,
        calls_and_results: list[tuple[ToolCall, ToolResult]],
    ) -> ProgressAssessment:
        snapshot = self._snapshot(round_number, calls_and_results)
        self._history.append(snapshot)
        return self._assess()

    def _snapshot(
        self,
        round_number: int,
        calls_and_results: list[tuple[ToolCall, ToolResult]],
    ) -> RoundProgressSnapshot:
        calls_payload = [
            (call.name, self._digest_json(call.input))
            for call, _result in calls_and_results
        ]
        results_payload = [
            (
                call.name,
                result.success,
                self._digest_text(result.content),
                self._digest_text(result.error),
            )
            for call, result in calls_and_results
        ]
        action_signature = self._digest_json(calls_payload)
        result_signature = self._digest_json(results_payload)
        state_signature = self._digest_json((calls_payload, results_payload))
        errors = sorted(
            self._digest_text(result.error.strip())
            for _call, result in calls_and_results
            if not result.success and result.error.strip()
        )
        error_signature = self._digest_json(errors) if errors else ""
        mutation_signatures = {
            self._digest_json((call.name, call.input))
            for call, result in calls_and_results
            if call.name in _WORKSPACE_TOOLS and result.success
        }
        workspace_changed = bool(mutation_signatures - self._seen_mutations)
        self._seen_mutations.update(mutation_signatures)
        new_information = bool(result_signature) and result_signature not in self._seen_results
        self._seen_results.add(result_signature)
        test_failures = self._test_failure_count(calls_and_results)
        test_improved = (
            test_failures is not None
            and self._last_test_failures is not None
            and test_failures < self._last_test_failures
        )
        if test_failures is not None:
            self._last_test_failures = test_failures
        timed_out = any(
            bool(_TIMEOUT_RE.search(result.error))
            for _call, result in calls_and_results
            if not result.success
        )
        return RoundProgressSnapshot(
            round_number=round_number,
            action_signature=action_signature,
            result_signature=result_signature,
            state_signature=state_signature,
            error_signature=error_signature,
            workspace_changed=workspace_changed,
            new_information=new_information,
            test_failures=test_failures,
            test_improved=test_improved,
            timed_out=timed_out,
        )

    def _assess(self) -> ProgressAssessment:
        items = list(self._history)
        if len(items) >= 2 and all(item.timed_out for item in items[-2:]):
            return self._assessment(
                ProgressState.HARD_STUCK,
                "连续 2 轮工具执行超时",
            )
        if (
            len(items) >= 3
            and items[-1].error_signature
            and len({item.error_signature for item in items[-3:]}) == 1
        ):
            return self._assessment(
                ProgressState.HARD_STUCK,
                "连续 3 轮出现相同工具错误",
            )
        if (
            len(items) >= 4
            and items[-4].state_signature == items[-2].state_signature
            and items[-3].state_signature == items[-1].state_signature
            and items[-1].state_signature != items[-2].state_signature
        ):
            return self._assessment(
                ProgressState.OSCILLATING,
                "最近 4 轮状态呈 A-B-A-B 往返震荡",
            )
        if (
            len(items) >= 3
            and len({item.action_signature for item in items[-3:]}) == 1
            and len({item.result_signature for item in items[-3:]}) == 1
            and not any(item.workspace_changed for item in items[-3:])
        ):
            return self._assessment(
                ProgressState.STALLED,
                "连续 3 轮工具调用和结果完全相同",
                "期间工作区没有发生有效修改",
            )
        if (
            len(items) >= 5
            and not any(item.effective_progress for item in items[-5:])
        ):
            return self._assessment(
                ProgressState.STALLED,
                "连续 5 轮没有新的工具信息、文件修改或测试改善",
            )
        if (
            len(items) >= 2
            and items[-1].state_signature == items[-2].state_signature
            and not items[-1].workspace_changed
        ):
            return self._assessment(
                ProgressState.SLOW,
                "连续 2 轮重复了相同工具调用和结果",
            )
        return ProgressAssessment(ProgressState.PROGRESSING)

    @staticmethod
    def _assessment(state: ProgressState, *reasons: str) -> ProgressAssessment:
        prompts = {
            ProgressState.SLOW: (
                "检测到执行开始重复。下一轮不要再次使用相同调用；先分析上次结果，"
                "说明失败根因，并选择能产生新证据的替代步骤。"
            ),
            ProgressState.STALLED: (
                "检测到任务没有有效进展。停止重复当前步骤，重新检查目标和已有证据，"
                "先给出一个不同的短计划，再执行其中最能验证根因的一步。"
            ),
            ProgressState.OSCILLATING: (
                "检测到方案在两个状态之间反复切换。不要再次撤销到之前状态；"
                "比较两种方案的失败证据，选择第三种策略或请求用户决策。"
            ),
            ProgressState.HARD_STUCK: (
                "检测到重复错误或超时。停止自动重试，定位根因；如果缺少权限、"
                "依赖或用户信息，应明确说明阻塞条件并请求帮助。"
            ),
        }
        return ProgressAssessment(state, tuple(reasons), prompts.get(state, ""))

    @staticmethod
    def _digest_json(value: object) -> str:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str,
        ).encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _digest_text(value: str) -> str:
        return hashlib.sha256(
            value.encode("utf-8", errors="replace")
        ).hexdigest()

    @staticmethod
    def _test_failure_count(
        calls_and_results: list[tuple[ToolCall, ToolResult]],
    ) -> int | None:
        counts: list[int] = []
        for call, result in calls_and_results:
            command = call.input.get("command") if call.name == "run_command" else None
            if not isinstance(command, str) or not _TEST_COMMAND_RE.search(command):
                continue
            text = result.to_message()
            failed = _FAILED_RE.search(text)
            if failed:
                counts.append(int(failed.group(1)))
                continue
            failures = _UNITTEST_FAILURE_RE.search(text)
            errors = _UNITTEST_ERROR_RE.search(text)
            if failures or errors:
                counts.append(
                    (int(failures.group(1)) if failures else 0)
                    + (int(errors.group(1)) if errors else 0)
                )
                continue
            if result.success or re.search(r"\bOK\b|\bpassed\b", text):
                counts.append(0)
        return min(counts) if counts else None

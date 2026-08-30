"""Typed, intentionally small schema for TinyCode evaluation cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileContainsAssertion:
    path: str
    text: str


@dataclass(frozen=True)
class CommandAssertion:
    command: str
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class EvalAssertions:
    tests: tuple[CommandAssertion, ...] = ()
    file_contains: tuple[FileContainsAssertion, ...] = ()
    final_text_contains: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    tools_not_used: tuple[str, ...] = ()
    no_errors: bool = True


@dataclass(frozen=True)
class EvalBudget:
    max_rounds: int = 12
    max_model_requests: int = 18
    max_tokens: int = 80_000
    max_duration_seconds: float = 300.0


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    source_path: Path
    fixture: Path | None = None
    assertions: EvalAssertions = field(default_factory=EvalAssertions)
    budgets: EvalBudget = field(default_factory=EvalBudget)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    points: float
    maximum: float


@dataclass(frozen=True)
class JudgeResult:
    available: bool
    tool_process: float = 0.0
    instruction_following: float = 0.0
    code_quality: float = 0.0
    rationale: str = ""
    raw_response: str = ""
    error: str = ""

    @property
    def points(self) -> float:
        return self.tool_process + self.instruction_following + self.code_quality


@dataclass(frozen=True)
class EvalReport:
    case: str
    executor: str
    judge: str
    score: float
    deterministic_score: float
    judge_score: float
    checks: tuple[CheckResult, ...]
    judge_result: JudgeResult
    final_text: str
    status: str
    duration_seconds: float
    rounds: int
    model_requests: int
    tokens: int
    tool_calls: int
    tool_success_rate: float | None
    tools: tuple[str, ...]
    errors: tuple[str, ...]
    workspace_changes: dict[str, list[str]]
    workspace_diff: str = ""
    workspace_diff_truncated: bool = False
    trace_path: str = ""
    workspace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

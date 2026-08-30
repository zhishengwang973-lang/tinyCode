"""Safe loading and validation of YAML evaluation cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tinyCode.evals.models import (
    CommandAssertion,
    EvalAssertions,
    EvalBudget,
    EvalCase,
    FileContainsAssertion,
)


class EvalConfigError(ValueError):
    """An evaluation case is malformed and can be fixed by its author."""


def load_case(path: Path) -> EvalCase:
    """Load an evaluation case without resolving paths outside its directory."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EvalConfigError(f"评测用例解析失败 ({path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise EvalConfigError("评测用例顶层必须是对象（mapping）")

    name = _string(raw, "name")
    prompt = _string(raw, "prompt")
    fixture = _optional_relative_path(raw.get("fixture"), path.parent, "fixture")
    assertions = _assertions(raw.get("assertions", {}))
    budgets = _budgets(raw.get("budgets", {}))
    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in tags_raw
    ):
        raise EvalConfigError("tags 必须是非空字符串列表")
    return EvalCase(
        name=name,
        prompt=prompt,
        source_path=path.resolve(),
        fixture=fixture,
        assertions=assertions,
        budgets=budgets,
        tags=tuple(item.strip() for item in tags_raw),
    )


def _assertions(raw: object) -> EvalAssertions:
    if not isinstance(raw, dict):
        raise EvalConfigError("assertions 必须是对象（mapping）")
    tests_raw = raw.get("tests", [])
    if not isinstance(tests_raw, list):
        raise EvalConfigError("assertions.tests 必须是列表")
    tests: list[CommandAssertion] = []
    for index, item in enumerate(tests_raw):
        if isinstance(item, str) and item.strip():
            tests.append(CommandAssertion(item.strip()))
            continue
        if not isinstance(item, dict):
            raise EvalConfigError(f"assertions.tests[{index}] 必须是命令字符串或对象")
        command = _string(item, "command", f"assertions.tests[{index}]")
        timeout = _number(item.get("timeout_seconds", 30), f"assertions.tests[{index}].timeout_seconds", 1, 600)
        tests.append(CommandAssertion(command, timeout))

    contains_raw = raw.get("file_contains", [])
    if not isinstance(contains_raw, list):
        raise EvalConfigError("assertions.file_contains 必须是列表")
    contains: list[FileContainsAssertion] = []
    for index, item in enumerate(contains_raw):
        if not isinstance(item, dict):
            raise EvalConfigError(f"assertions.file_contains[{index}] 必须是对象")
        value_path = _safe_relative_path(
            _string(item, "path", f"assertions.file_contains[{index}]"),
            f"assertions.file_contains[{index}].path",
        )
        contains.append(FileContainsAssertion(
            value_path,
            _string(item, "text", f"assertions.file_contains[{index}]"),
        ))

    return EvalAssertions(
        tests=tuple(tests),
        file_contains=tuple(contains),
        final_text_contains=_strings(raw.get("final_text_contains", []), "assertions.final_text_contains"),
        tools_used=_strings(raw.get("tools_used", []), "assertions.tools_used"),
        tools_not_used=_strings(raw.get("tools_not_used", []), "assertions.tools_not_used"),
        no_errors=_bool(raw.get("no_errors", True), "assertions.no_errors"),
    )


def _budgets(raw: object) -> EvalBudget:
    if not isinstance(raw, dict):
        raise EvalConfigError("budgets 必须是对象（mapping）")
    return EvalBudget(
        max_rounds=int(_number(raw.get("max_rounds", 12), "budgets.max_rounds", 1, 100)),
        max_model_requests=int(_number(raw.get("max_model_requests", 18), "budgets.max_model_requests", 1, 500)),
        max_tokens=int(_number(raw.get("max_tokens", 80_000), "budgets.max_tokens", 1, 10_000_000)),
        max_duration_seconds=_number(raw.get("max_duration_seconds", 300), "budgets.max_duration_seconds", 1, 3600),
    )


def _string(raw: dict[str, Any], key: str, prefix: str = "") -> str:
    value = raw.get(key)
    name = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        raise EvalConfigError(f"{name} 必须是非空字符串")
    return value.strip()


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EvalConfigError(f"{name} 必须是非空字符串列表")
    return tuple(item.strip() for item in value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise EvalConfigError(f"{name} 必须是 true 或 false")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalConfigError(f"{name} 必须是数字")
    number = float(value)
    if not minimum <= number <= maximum:
        raise EvalConfigError(f"{name} 必须在 {minimum:g} 到 {maximum:g} 之间")
    return number


def _optional_relative_path(value: object, parent: Path, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvalConfigError(f"{name} 必须是相对路径字符串")
    relative = _safe_relative_path(value.strip(), name)
    resolved = (parent / relative).resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise EvalConfigError(f"{name} 不允许离开用例目录") from exc
    if not resolved.is_dir():
        raise EvalConfigError(f"{name} 目录不存在: {value}")
    return resolved


def _safe_relative_path(value: str, name: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvalConfigError(f"{name} 必须是工作区内的相对路径")
    return candidate.as_posix()

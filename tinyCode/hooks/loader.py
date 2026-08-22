"""Hook loader — YAML parsing with centralized validation."""

import math
import sys
from pathlib import Path

import yaml

from tinyCode.hooks.models import (
    Action,
    ActionType,
    Condition,
    ConditionRule,
    Control,
    HookEvent,
    MatchMode,
    Operator,
    Rule,
)

# Valid event names (auto-generated from enum)
_VALID_EVENTS = {e.value for e in HookEvent}

# Events that support the "intercept" pattern
_INTERCEPT_EVENTS = {HookEvent.TOOL_PRE_EXEC.value}

# Action types that need "command" field
_NEEDS_COMMAND = {ActionType.SHELL}
_NEEDS_TEXT = {ActionType.PROMPT_INJECT}
_NEEDS_URL = {ActionType.HTTP}
_NEEDS_TASK = {ActionType.SUB_AGENT}


def _string_field(data: dict, field: str, default: str = "") -> str:
    value = data.get(field, default)
    if not isinstance(value, str):
        raise ValueError(f"action.{field} 必须是字符串")
    return value


def _bool_field(data: dict, field: str, default: bool = False) -> bool:
    value = data.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"control.{field} 必须是布尔值")
    return value


def _timeout_field(data: dict, field: str, default: float = 30.0) -> float:
    value = data.get(field, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"control.{field} 必须是数字")
    if not math.isfinite(float(value)) or not 0 < value <= 3600:
        raise ValueError(f"control.{field} 必须是 0 到 3600 之间的有限数字")
    return float(value)


def _condition_string_field(data: dict, field: str, default: str = "") -> str:
    value = data.get(field, default)
    if not isinstance(value, str):
        raise ValueError(f"condition.{field} 必须是字符串")
    return value


def load_hooks(
    project_path: Path | None = None,
    global_path: Path | None = None,
    *,
    include_project: bool = True,
) -> list[Rule]:
    """Load hook rules from project + global YAML files.

    Merges both files (project overrides nothing — both sets run).
    """
    if project_path is None:
        project_path = Path.cwd() / ".tinyCode-hooks.yaml"
    if global_path is None:
        global_path = Path.home() / ".tinyCode" / "hooks.yaml"

    rules: list[Rule] = []
    paths = [global_path]
    if include_project:
        paths.append(project_path)
    for path in paths:
        rules.extend(_load_file(path))
    return rules


def _load_file(path: Path) -> list[Rule]:
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        print(f"Hook [{path}]: 读取或解析失败 — {exc}", file=sys.stderr)
        return []
    if not isinstance(raw, dict) or "hooks" not in raw:
        return []
    if not isinstance(raw["hooks"], list):
        print(f"Hook [{path}]: 'hooks' 必须是列表，跳过", file=sys.stderr)
        return []

    rules: list[Rule] = []
    for i, entry in enumerate(raw["hooks"]):
        if not isinstance(entry, dict):
            print(f"Hook [{path}]: 规则 #{i} 不是字典，跳过", file=sys.stderr)
            continue
        try:
            rule = _parse_rule(entry, str(path), i)
            if rule:
                rules.append(rule)
        except ValueError as exc:
            print(f"Hook [{path}]: 规则 #{i} 无效 — {exc}", file=sys.stderr)
    return rules


def _parse_rule(entry: dict, source: str, index: int) -> Rule | None:
    # --- event (required) ---
    event_str = entry.get("event", "")
    if not event_str:
        raise ValueError("缺少 'event' 字段")
    if event_str not in _VALID_EVENTS:
        raise ValueError(f"无效事件: {event_str}（有效值: {', '.join(sorted(_VALID_EVENTS))}）")
    event = HookEvent(event_str)

    # --- condition (optional) ---
    condition = None
    cond_raw = entry.get("condition")
    if cond_raw:
        if not isinstance(cond_raw, dict):
            raise ValueError("'condition' 必须是字典")
        match_str = _condition_string_field(cond_raw, "match", "ALL").upper()
        if match_str not in ("ALL", "ANY"):
            raise ValueError(f"condition.match 必须是 ALL 或 ANY: {match_str}")
        cond_rules: list[ConditionRule] = []
        for cr in cond_raw.get("rules", []):
            if not isinstance(cr, dict):
                raise ValueError("condition.rules 的每个元素必须是字典")
            op_str = _condition_string_field(cr, "operator", "exact").lower()
            try:
                op = Operator(op_str)
            except ValueError:
                raise ValueError(f"无效 operator: {op_str}")
            cond_rules.append(ConditionRule(
                field=_condition_string_field(cr, "field"),
                operator=op,
                value=str(cr.get("value", "")),
            ))
        condition = Condition(match=MatchMode(match_str), rules=cond_rules)

    # --- actions (required, at least one) ---
    actions: list[Action] = []
    for a in entry.get("actions", []):
        if not isinstance(a, dict):
            raise ValueError("actions 的每个元素必须是字典")
        type_str = a.get("type", "")
        try:
            atype = ActionType(type_str)
        except ValueError:
            raise ValueError(f"无效 action type: {type_str}")
        headers = a.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError("action.headers 必须是字典")
        action = Action(
            type=atype,
            command=_string_field(a, "command"),
            text=_string_field(a, "text"),
            url=_string_field(a, "url"),
            method=_string_field(a, "method", "POST"),
            headers=headers,
            body=_string_field(a, "body"),
            task=_string_field(a, "task"),
        )
        # Validate required fields
        if atype in _NEEDS_COMMAND and not action.command:
            raise ValueError(f"action type '{type_str}' 缺少 'command' 字段")
        if atype in _NEEDS_TEXT and not action.text:
            raise ValueError(f"action type '{type_str}' 缺少 'text' 字段")
        if atype in _NEEDS_URL and not action.url:
            raise ValueError(f"action type '{type_str}' 缺少 'url' 字段")
        if atype in _NEEDS_TASK and not action.task:
            raise ValueError(f"action type '{type_str}' 缺少 'task' 字段")
        actions.append(action)

    if not actions:
        raise ValueError("至少需要一个 action")

    # --- control ---
    ctrl_raw = entry.get("control", {})
    if not isinstance(ctrl_raw, dict):
        raise ValueError("'control' 必须是字典")
    control = Control(
        once=_bool_field(ctrl_raw, "once"),
        async_=_bool_field(ctrl_raw, "async"),
        timeout=_timeout_field(ctrl_raw, "timeout"),
    )

    # --- validation: intercept rules cannot be async ---
    if event_str in _INTERCEPT_EVENTS and control.async_:
        raise ValueError(f"拦截事件 '{event_str}' 不允许 async=true")
    if control.async_ and any(action.type == ActionType.PROMPT_INJECT for action in actions):
        raise ValueError("prompt_inject 不允许 async=true")

    rule = Rule(
        event=event,
        condition=condition,
        actions=actions,
        control=control,
        name=entry.get("name", f"rule-{index}"),
    )
    return rule

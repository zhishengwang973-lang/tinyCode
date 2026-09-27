"""High-precision, clause-scoped intent constraints.

Rules here never guess a broad semantic intent.  They only extract explicit
user constraints such as "don't use Team"; ambiguous combinations are left as
``AUTO`` so the semantic router or conservative baseline can decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class TeamPolicy(str, Enum):
    AUTO = "auto"
    DENY = "deny"
    FORCE = "force"


class SubagentPolicy(str, Enum):
    AUTO = "auto"
    DENY = "deny"
    READ_ONLY_ALLOWED = "read_only_allowed"


class PlanningPolicy(str, Enum):
    AUTO = "auto"
    DENY = "deny"
    FORCE = "force"


class ModePolicy(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    INSPECT = "inspect"
    MODIFY = "modify"


@dataclass(frozen=True)
class IntentConstraints:
    """Only explicit, scoped constraints; never a forced semantic guess."""

    team: TeamPolicy = TeamPolicy.AUTO
    subagent: SubagentPolicy = SubagentPolicy.AUTO
    planning: PlanningPolicy = PlanningPolicy.AUTO
    mode: ModePolicy = ModePolicy.AUTO
    evidence: tuple[str, ...] = field(default_factory=tuple)
    uncertain: bool = False


_CLAUSE_SPLIT_RE = re.compile(r"[。！？!?；;，,\n]+")
_NEGATION = r"(?:不要|不使用|不用|无需|禁止|别用|避免|don't|do not|without|no need to)"
_TEAM = r"(?:agent\s*team|\bteam\b|多智能体|多个\s*agent)"
_SUBAGENT = r"(?:sub[ -]?agent|子(?:agent|智能体))"
_PLANNING = r"(?:任务拆分|拆分任务|任务计划|任务规划|执行计划|task[ -]?planning)"
_MUTATION = r"(?:修改|改动|编辑|写入|新建|创建|新增|删除|移除|替换|实现|开发|修复|重构|迁移|安装|提交|推送|运行|执行|测试|验证|modify|edit|write|create|delete|implement|fix|refactor|run|test)"
_INSPECT = r"(?:查看|检查|分析|审查|审阅|诊断|排查|总结|评估|读取|搜索|read|inspect|review|analy[sz]e|diagnose)"


def extract_intent_constraints(text: str) -> IntentConstraints:
    """Extract explicit constraints with denial winning over positive intent."""
    clauses = [item.strip() for item in _CLAUSE_SPLIT_RE.split(text) if item.strip()]
    evidence: list[str] = []

    team_deny = any(_denies(clause, _TEAM) for clause in clauses)
    team_force = any(
        _forces(clause, _TEAM) and not _denies(clause, _TEAM)
        for clause in clauses
    )
    team = TeamPolicy.DENY if team_deny else TeamPolicy.FORCE if team_force else TeamPolicy.AUTO
    if team_deny:
        evidence.append("明确禁止 Agent Team")
    elif team_force:
        evidence.append("明确要求 Agent Team")

    subagent_deny = any(_denies(clause, _SUBAGENT) for clause in clauses)
    subagent_read_only = bool(
        re.search(_SUBAGENT + r".{0,72}?(?:只读|read[ -]?only)", text, re.I)
        or re.search(r"(?:只读|read[ -]?only).{0,72}?" + _SUBAGENT, text, re.I)
    )
    subagent = (
        SubagentPolicy.DENY if subagent_deny
        else SubagentPolicy.READ_ONLY_ALLOWED if subagent_read_only
        else SubagentPolicy.AUTO
    )
    if subagent_deny:
        evidence.append("明确禁止 Subagent")
    elif subagent_read_only:
        evidence.append("Subagent 限制为只读")

    planning_deny = any(_denies(clause, _PLANNING) for clause in clauses)
    planning_force = any(
        _forces(clause, _PLANNING) and not _denies(clause, _PLANNING)
        for clause in clauses
    )
    planning = (
        PlanningPolicy.DENY if planning_deny
        else PlanningPolicy.FORCE if planning_force
        else PlanningPolicy.AUTO
    )
    if planning_deny:
        evidence.append("明确禁止任务计划")
    elif planning_force:
        evidence.append("明确要求任务计划")

    global_read_only = any(
        _global_read_only(clause) and not re.search(_SUBAGENT, clause, re.I)
        for clause in clauses
    )
    positive_modify = any(_has_positive_mutation(clause) for clause in clauses)
    direct_only = any(
        re.search(r"(?:直接|只)(?:回答|解释|输出)|(?:answer|respond)\s+directly", clause, re.I)
        for clause in clauses
    )
    if direct_only and not positive_modify:
        mode = ModePolicy.DIRECT
        evidence.append("明确要求直接回答")
    elif global_read_only and not positive_modify:
        mode = ModePolicy.INSPECT
        evidence.append("明确全局只读")
    elif positive_modify and not global_read_only:
        # Positive action vocabulary is intentionally evidence rather than a
        # hard override: words like "write", "重构" or "修改" may appear in a
        # question, quoted text, or an API name.  The established task-mode
        # rules / semantic router resolve this remaining semantic choice.
        mode = ModePolicy.AUTO
        evidence.append("检测到实施动词，等待任务模式路由确认")
    else:
        mode = ModePolicy.AUTO

    # Conflicts intentionally remain unresolved; only the conservative router
    # may choose the final mode after seeing the complete conversation.
    uncertain = (team_deny and team_force) or (planning_deny and planning_force) or (
        global_read_only and positive_modify
    )
    return IntentConstraints(
        team=team, subagent=subagent, planning=planning, mode=mode,
        evidence=tuple(evidence), uncertain=uncertain,
    )


def _denies(clause: str, target: str) -> bool:
    return bool(re.search(_NEGATION + r".{0,48}?" + target, clause, re.I))


def _forces(clause: str, target: str) -> bool:
    return bool(re.search(
        r"(?:请|必须|务必|使用|启用|采用|用|please|must|use).{0,24}?" + target,
        clause,
        re.I,
    ))


def _global_read_only(clause: str) -> bool:
    # "不要只分析，直接修改" negates the read-only instruction itself.
    if re.search(_NEGATION + r".{0,8}?(?:只|仅).{0,8}?" + _INSPECT, clause, re.I):
        return False
    return bool(
        _denies(clause, _MUTATION)
        or re.search(r"(?:只|仅)(?:做)?" + _INSPECT, clause, re.I)
        or re.search(r"(?:^|[，,])\s*(?:只读|read[ -]?only)(?:任务|模式)?(?:$|[，,])", clause, re.I)
    )


def _has_positive_mutation(clause: str) -> bool:
    """Find an action verb not syntactically covered by a nearby negation."""
    action = (
        r"(?:实现|新增|修复|重构|迁移|创建|写入|删除|安装|提交|推送|"
        r"运行|执行|启动|构建|modify|fix|refactor|implement|create|"
        r"delete|install|commit|push|run|execute|start|build)"
    )
    for match in re.finditer(action, clause, re.I):
        prefix = clause[max(0, match.start() - 18):match.start()]
        if re.search(_NEGATION + r".{0,12}$", prefix, re.I):
            continue
        return True
    return False

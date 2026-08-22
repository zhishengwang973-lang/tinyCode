"""Skill data models."""

from dataclasses import dataclass, field
from enum import Enum


class SkillMode(Enum):
    SHARED = "shared"        # 共享主对话上下文，结果留在历史里
    ISOLATED = "isolated"    # 独立对话，结果摘要回流


class HistoryCarry(Enum):
    FULL = "full"            # 全量摘要带回
    RECENT = "recent"        # 最近 N 条
    NONE = "none"            # 完全清空，不带历史


@dataclass
class SkillMeta:
    """Parsed from YAML frontmatter."""
    name: str
    description: str = ""
    mode: SkillMode = SkillMode.SHARED
    model: str | None = None
    tools: list[str] | None = None       # None = all tools, [] = no tools
    history_carry: HistoryCarry = HistoryCarry.FULL
    recent_count: int = 10               # for RECENT mode
    source: str = ""                     # file path (for hot-reload)


@dataclass
class SkillDefinition:
    """A fully loaded skill — meta + body instructions."""
    meta: SkillMeta
    body: str                            # Markdown SOP instructions
    directory: str = ""                  # if directory skill, the dir path
    custom_tools: list[dict] = field(default_factory=list)

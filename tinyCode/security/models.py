"""Security data models — rules, decisions, security levels."""

from dataclasses import dataclass, field
from enum import Enum


class SecurityLevel(Enum):
    """Global permission mode."""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


class RuleAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class HITLDecision(Enum):
    """User response to a human-in-the-loop prompt."""
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    ALLOW_PERMANENT = "allow_permanent"
    DENY = "deny"


class RuleScope(Enum):
    SESSION = "session"
    PROJECT = "project"
    GLOBAL = "global"


@dataclass
class SecurityRule:
    """A single security rule matching tool invocations."""
    tool: str
    action: RuleAction
    path_pattern: str | None = None       # glob-style: "*.env", "src/**"
    command_pattern: str | None = None    # glob-style or regex
    scope: RuleScope = RuleScope.PROJECT

    def matches(self, tool_name: str, path: str | None = None, command: str | None = None) -> bool:
        if not self._tool_matches(tool_name):
            return False
        if self.path_pattern is not None:
            if path is None:
                return False
            if not self._glob_match(self.path_pattern, path):
                return False
        if self.command_pattern is not None:
            if command is None:
                return False
            if not self._glob_match(self.command_pattern, command):
                return False
        return True

    def _tool_matches(self, tool_name: str) -> bool:
        # Support wildcard
        import fnmatch
        return fnmatch.fnmatch(tool_name, self.tool)

    @staticmethod
    def _glob_match(pattern: str, value: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(value, pattern)

"""Tool filter — multi-layer defense against nested sub-agent chains."""

from tinyCode.subagent.models import SubAgentRole

# Always blocked in sub-agents (prevents A→B→C chains)
GLOBAL_BLOCKED = {"sub_agent", "skill_loader", "request_user_input"}

# Fallback for direct/library callers that do not provide registry categories.
BACKGROUND_WHITELIST = {
    "read_file", "glob", "grep", "tool_result_search", "tool_result_read",
    "web_search", "web_fetch",
}


class ToolFilter:
    """Filters tool availability for sub-agents."""

    def __init__(
        self,
        role: SubAgentRole | None,
        background: bool = False,
        parent_tools: list[str] | None = None,
        read_tools: set[str] | None = None,
    ) -> None:
        self._role = role
        self._background = background
        self._parent_tools = parent_tools or []
        self._read_tools = set(read_tools or BACKGROUND_WHITELIST)

    def filter(self, tool_names: list[str]) -> list[str]:
        """Return the list of allowed tool names."""
        allowed = set(tool_names)
        if self._parent_tools:
            allowed &= set(self._parent_tools)

        # Layer 1: global blocked
        allowed -= GLOBAL_BLOCKED

        # Layer 2: role allow/deny
        if self._role:
            if self._role.tools_allow is not None:
                allowed &= set(self._role.tools_allow)
            allowed -= set(self._role.tools_deny)
            if self._role.permission == "strict":
                allowed &= self._read_tools

        # Layer 3: background workers — read-only
        if self._background:
            allowed &= self._read_tools

        return sorted(allowed)

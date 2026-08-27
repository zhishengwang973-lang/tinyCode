"""PromptBuilder — assembles system prompt from modules with cache breakpoints."""

from tinyCode.prompts.loader import PromptModule, load_modules


class PromptBuilder:
    """Assembles the system prompt from prioritized modules.

    Supports two output formats:
    - ``build()`` → plain string (OpenAI / DeepSeek / general use)
    - ``build_anthropic()`` → list of Anthropic content blocks with cache_control
    """

    def __init__(self) -> None:
        self._modules: list[PromptModule] = load_modules()
        self._extra_sections: list[str] = []

    def add_section(self, text: str) -> None:
        """Append an extra section after all modules (not cached)."""
        self._extra_sections.append(text)

    def clear_extra(self) -> None:
        self._extra_sections.clear()

    # -- output ---------------------------------------------------------------

    def build(self, *, include_tool_instructions: bool = True) -> str:
        """Plain string system prompt.

        Direct, self-contained questions have no tool schema and therefore do
        not need the long operational tool policy.  The remaining modules are
        unchanged, so behavior, safety, and answer-format instructions stay
        intact.
        """
        parts: list[str] = []
        for m in self._modules:
            if not include_tool_instructions and m.name == "tools":
                continue
            parts.append(m.content)
        parts.extend(self._extra_sections)
        return "\n\n".join(parts)

    def build_anthropic(self, *, include_tool_instructions: bool = True) -> list[dict]:
        """Anthropic content blocks with ``cache_control`` on the last block.

        All stable modules share one cache breakpoint at the end.
        Extra (dynamic) sections are appended without cache markers.
        """
        blocks: list[dict] = []
        stable_text = "\n\n".join(
            m.content
            for m in self._modules
            if include_tool_instructions or m.name != "tools"
        )

        # Wrap stable text in a cache-controlled block
        blocks.append({
            "type": "text",
            "text": stable_text,
            "cache_control": {"type": "ephemeral"},
        })

        # Dynamic extras — no cache
        for extra in self._extra_sections:
            blocks.append({"type": "text", "text": extra})

        return blocks

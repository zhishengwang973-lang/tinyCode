"""Template variable substitution — ``{{var}}`` replaced from context.

Undefined variables → empty string (never raises).
"""

import re

_VAR_RE = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")


class TemplateEngine:
    """Substitute ``{{var}}`` placeholders from a flat + nested context dict."""

    def render(self, template: str, context: dict) -> str:
        def _replace(match: re.Match) -> str:
            path = match.group(1)
            parts = path.split(".")
            current = context
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part, "")
                else:
                    return ""
            return str(current) if current is not None else ""

        return _VAR_RE.sub(_replace, template)

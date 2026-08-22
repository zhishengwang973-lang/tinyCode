"""PromptInjector — manages per-round dynamic instruction injection."""

from tinyCode.prompts.loader import load_injection

#: Round interval for repeating the full injection (slim used on off-rounds).
FULL_INJECTION_INTERVAL = 3


class PromptInjector:
    """Generates per-round injection messages for session-level state.

    Injection messages use the ``user`` role with a ``[TinyCode]`` prefix
    so the model treats them as system-level context rather than user input.
    """

    def __init__(self) -> None:
        self._plan_only = False
        self._round_counter = 0
        self._pending_injections: list[str] = []

    # -- state ----------------------------------------------------------------

    def set_plan_only(self, enabled: bool) -> None:
        self._plan_only = enabled

    def queue_injection(self, text: str) -> None:
        """Add a one-shot injection for the next round only."""
        self._pending_injections.append(f"[TinyCode] {text}")

    # -- per-round injection --------------------------------------------------

    def build_injection(self, round_number: int) -> str | None:
        """Return the injection message for the given round, or None."""
        parts: list[str] = []

        # One-shot injections
        parts.extend(self._pending_injections)
        self._pending_injections.clear()

        # Plan-only mode: full every N rounds, slim on off-rounds
        if self._plan_only:
            is_first = (round_number == 1)
            is_full_round = (round_number % FULL_INJECTION_INTERVAL == 0)
            if is_first or is_full_round:
                text = load_injection("plan-mode")
            else:
                text = load_injection("plan-mode-slim")
            if text:
                parts.append(text)

        if not parts:
            return None
        return "\n\n".join(parts)

    def preview_injection(self, round_number: int = 1) -> str | None:
        """Preview an injection without consuming one-shot entries."""
        parts = list(self._pending_injections)

        if self._plan_only:
            is_first = round_number == 1
            is_full_round = round_number % FULL_INJECTION_INTERVAL == 0
            name = "plan-mode" if is_first or is_full_round else "plan-mode-slim"
            text = load_injection(name)
            if text:
                parts.append(text)

        return "\n\n".join(parts) if parts else None

"""PromptInjector — manages task snapshots and append-only recovery context."""

from tinyCode.prompts.loader import load_injection

class PromptInjector:
    """Generates stable task instructions and queued recovery context.

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

    # -- task and recovery context -------------------------------------------

    def build_task_injection(self) -> str | None:
        """Return a task-stable instruction snapshot.

        Plan mode used to alternate between two strings every round.  That
        placed a changing message before the complete conversation and broke
        prompt-cache reuse.  Enforcement remains in the tool layer, while a
        single full instruction is stable for the active task.
        """
        if not self._plan_only:
            return None
        text = load_injection("plan-mode")
        return text or None

    def consume_pending_injections(self) -> list[str]:
        """Consume recovery prompts for append-only conversation history."""
        pending = self._pending_injections
        self._pending_injections = []
        return pending

    # -- compatibility helpers -----------------------------------------------

    def build_injection(self, round_number: int) -> str | None:
        """Return current context for legacy callers.

        AgentLoop uses :meth:`build_task_injection` and persists one-shot
        entries instead, so this method deliberately no longer varies by
        round number.
        """
        del round_number
        parts = self.consume_pending_injections()
        task_instruction = self.build_task_injection()
        if task_instruction:
            parts.append(task_instruction)
        return "\n\n".join(parts) if parts else None

    def preview_injection(self, round_number: int = 1) -> str | None:
        """Preview an injection without consuming one-shot entries."""
        parts = list(self._pending_injections)

        del round_number
        task_instruction = self.build_task_injection()
        if task_instruction:
            parts.append(task_instruction)

        return "\n\n".join(parts) if parts else None

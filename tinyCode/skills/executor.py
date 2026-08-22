"""Skill executor — shared vs isolated execution modes."""

from tinyCode.providers.base import BaseProvider
from tinyCode.skills.models import HistoryCarry, SkillDefinition, SkillMode


MAX_ISOLATED_SKILL_OUTPUT_CHARS = 500_000


class SkillExecutor:
    """Handles isolated execution of skills (shared runs inline by Agent)."""

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    async def execute_isolated(
        self,
        skill: SkillDefinition,
        task: str,
        history: list[dict],
    ) -> str:
        """Run a skill in an isolated conversation, return summary.

        Args:
            skill: The skill definition.
            task: The user's task description.
            history: Current conversation messages (for context carry).

        Returns:
            A summary string to inject back into the main conversation.
        """
        if skill.meta.history_carry == HistoryCarry.NONE:
            context_messages: list[dict] = []
        elif skill.meta.history_carry == HistoryCarry.RECENT:
            context_messages = history[-skill.meta.recent_count * 2:]
        else:
            # FULL: summarize and carry
            context_messages = history

        # Build isolated conversation
        messages: list[dict] = [
            {"role": "system", "content": f"[Skill: {skill.meta.name}]\n{skill.body}"},
        ]
        if context_messages:
            summary_parts: list[str] = []
            for m in context_messages[-20:]:
                content = m.get("content", "")
                if isinstance(content, str):
                    summary_parts.append(f"[{m.get('role', '?')}]: {content[:500]}")
            context_text = "\n".join(summary_parts)
            messages.append({
                "role": "user",
                "content": f"[对话上下文摘要]\n{context_text}\n\n---\n任务: {task}",
            })
        else:
            messages.append({"role": "user", "content": task})

        # Run in isolation
        response_parts: list[str] = []
        output_chars = 0
        async for token in self._provider.chat_stream(messages):
            if isinstance(token, str) and not token.startswith("<<"):
                response_parts.append(token)
                output_chars += len(token)
                if output_chars > MAX_ISOLATED_SKILL_OUTPUT_CHARS:
                    raise ValueError(
                        "隔离 Skill 输出超过 "
                        f"{MAX_ISOLATED_SKILL_OUTPUT_CHARS} 字符上限"
                    )

        response = "".join(response_parts)
        return f"[Skill '{skill.meta.name}' 执行结果]\n{response}"

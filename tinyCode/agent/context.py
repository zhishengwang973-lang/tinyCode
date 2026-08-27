"""Prompt-context assembly, isolated from the ReAct state machine."""

from collections.abc import Callable
from typing import Any

from tinyCode.conversation.history import ConversationHistory
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.tools.registry import ToolRegistry


class PromptContextAssembler:
    def __init__(
        self,
        *,
        protocol: str,
        prompt_builder: PromptBuilder,
        prompt_injector: PromptInjector,
        instructions_text: str = "",
        environment_text: str | Callable[[], str] = "",
        notes_text: str | Callable[[], str] = "",
        skill_registry: Any = None,
    ) -> None:
        self._protocol = protocol
        self._prompt_builder = prompt_builder
        self._prompt_injector = prompt_injector
        self._instructions_text = instructions_text
        self._environment_text = environment_text
        self._notes_text = notes_text
        self._skill_registry = skill_registry

    def environment_text(self) -> str:
        if callable(self._environment_text):
            return self._environment_text()
        return self._environment_text

    def notes_text(self) -> str:
        if callable(self._notes_text):
            return self._notes_text()
        return self._notes_text

    def assemble(
        self,
        history: ConversationHistory,
        round_number: int,
        *,
        environment_text: str | None = None,
        notes_text: str | None = None,
        skill_instructions: str | None = None,
        task_injection: str | None = None,
        direct_answer: bool = False,
        include_environment: bool = True,
    ) -> list[dict]:
        result: list[dict] = []
        is_anthropic = self._protocol == "anthropic"
        if not is_anthropic:
            system_prompt = self._prompt_builder.build(
                include_tool_instructions=not direct_answer,
            )
            if system_prompt:
                result.append({"role": "system", "content": system_prompt})

        self._append_pinned(
            result, "Instructions", self._instructions_text, is_anthropic,
        )
        if self._skill_registry:
            self._append_pinned(
                result,
                "Activated Skills",
                (
                    self._skill_registry.get_active_instructions()
                    if skill_instructions is None
                    else skill_instructions
                ),
                is_anthropic,
            )

        # Persistent project notes and workspace facts are useful for an
        # in-project task, but add irrelevant cost (and cache churn) to a
        # self-contained question such as an algorithm explanation.
        if not direct_answer:
            self._append_pinned(
                result,
                "Notes",
                self.notes_text() if notes_text is None else notes_text,
                is_anthropic,
            )

        injection = (
            self._prompt_injector.build_injection(round_number)
            if task_injection is None else task_injection
        )
        if injection:
            result.append({"role": "user", "content": injection})

        if not direct_answer or include_environment:
            self._append_pinned(
                result,
                "Environment",
                self.environment_text() if environment_text is None else environment_text,
                is_anthropic,
            )

        conversation = history.get_messages()
        if is_anthropic:
            conversation = [
                {**message, "role": "user"}
                if message.get("role") == "system" else message
                for message in conversation
            ]
        result.extend(conversation)
        return result

    def system_blocks(self, *, direct_answer: bool = False) -> list[dict] | None:
        if self._protocol != "anthropic":
            return None
        return self._prompt_builder.build_anthropic(
            include_tool_instructions=not direct_answer,
        )

    def tool_definitions(self, registry: ToolRegistry) -> list[dict]:
        tools = (
            registry.to_anthropic_format()
            if self._protocol == "anthropic"
            else registry.to_openai_format()
        )
        if not self._skill_registry:
            return tools
        whitelist = self._skill_registry.get_active_tool_whitelist()
        if whitelist is None:
            return tools
        allowed = set(whitelist) | {"skill_loader"}
        return [tool for tool in tools if self._tool_name(tool) in allowed]

    @staticmethod
    def _append_pinned(
        target: list[dict], label: str, text: str, anthropic: bool,
    ) -> None:
        if text:
            target.append({
                "role": "user" if anthropic else "system",
                "content": f"[{label}]\n{text}",
            })

    @staticmethod
    def _tool_name(tool: dict) -> str:
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name", "")
            return name if isinstance(name, str) else ""
        name = tool.get("name", "")
        return name if isinstance(name, str) else ""

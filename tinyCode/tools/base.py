"""Tool system — base classes."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCategory(Enum):
    """Classification for tool execution batching.

    READ tools can run concurrently with other READ tools.
    WRITE tools must run serially to avoid conflicts.
    """

    READ = "read"
    WRITE = "write"


@dataclass
class ToolResult:
    """Structured result from executing a tool."""

    success: bool
    content: str
    error: str = ""

    def to_message(self) -> str:
        """Render as a plain-text result for injection into conversation."""
        if self.success:
            return self.content
        return f"工具执行失败: {self.error}\n\n输出:\n{self.content}"


@dataclass
class ToolParameter:
    """Description of a single tool parameter."""

    name: str
    type: str  # JSON Schema type: "string", "integer", "boolean"
    description: str
    required: bool = True
    default: Any = None
    item_type: str | None = None


class BaseTool(ABC):
    """Abstract interface for a tool the agent can use."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier, e.g. ``"read_file"``."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the model."""
        ...

    @property
    def category(self) -> ToolCategory:
        """READ (concurrent-safe) or WRITE (serial-only).

        Default is WRITE (safe default). Override in READ tools.
        """
        return ToolCategory.WRITE

    @property
    def timeout_exempt(self) -> bool:
        """Whether ToolExecutor should wait without its ordinary deadline.

        Only foreground human interaction should normally override the shared
        tool deadline. Cancellation of the owning turn still propagates.
        """
        return False

    @property
    def available_in_inspect(self) -> bool:
        """Whether the schema may be advertised for a read-only task.

        Most tools have fixed side effects and can use their category. A
        dispatcher may override this while still deciding each concrete call
        with :meth:`may_modify`.
        """
        return self.category is ToolCategory.READ

    def may_modify(self, params: dict[str, Any]) -> bool:
        """Return whether this concrete call may mutate project state."""
        del params
        return self.category is not ToolCategory.READ

    def security_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return arguments used to scope security rules for this call."""
        return dict(params)

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Parameter schema (name, type, description, required)."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given named parameters."""
        ...

    def approval_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return the user-visible parameters for an approval prompt.

        Execution always receives the original provider arguments. Tools with
        indirect capabilities (for example a delegated worker) may override
        this hook to expose the effective permission envelope to the user.
        """
        return dict(params)

    def to_openai_schema(self) -> dict:
        """Render tool definition in OpenAI-compatible format."""
        props: dict[str, dict] = {}
        required: list[str] = []
        for p in self.parameters:
            props[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.item_type is not None:
                props[p.name]["items"] = {"type": p.item_type}
            if not p.required and p.default is not None:
                props[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        """Render tool definition in Anthropic-compatible format."""
        props: dict[str, dict] = {}
        required: list[str] = []
        for p in self.parameters:
            props[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.item_type is not None:
                props[p.name]["items"] = {"type": p.item_type}
            if not p.required and p.default is not None:
                props[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

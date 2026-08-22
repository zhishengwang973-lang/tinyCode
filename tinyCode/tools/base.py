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
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Parameter schema (name, type, description, required)."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given named parameters."""
        ...

    def to_openai_schema(self) -> dict:
        """Render tool definition in OpenAI-compatible format."""
        props: dict[str, dict] = {}
        required: list[str] = []
        for p in self.parameters:
            props[p.name] = {
                "type": p.type,
                "description": p.description,
            }
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

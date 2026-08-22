"""Tool registry — central catalog of available tools."""

import re

from tinyCode.tools.base import BaseTool


_PROVIDER_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JSON_SCHEMA_TYPES = {
    "array", "boolean", "integer", "null", "number", "object", "string",
}


class ToolRegistry:
    """Registers tools and provides lookups and API-format conversion."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry."""
        name = tool.name
        if not isinstance(name, str) or not name:
            raise ValueError("工具名必须是非空字符串")
        if not _PROVIDER_TOOL_NAME_RE.fullmatch(name):
            raise ValueError(
                f"工具名必须匹配 [A-Za-z0-9_-] 且不超过 64 字符: {name}"
            )
        description = tool.description
        if not isinstance(description, str):
            raise ValueError(f"工具描述必须是字符串: {name}")
        self._validate_parameters(tool)
        if name in self._tools:
            raise ValueError(f"工具名重复: {name}")
        self._tools[name] = tool

    def _validate_parameters(self, tool: BaseTool) -> None:
        parameters = tool.parameters
        if not isinstance(parameters, list):
            raise ValueError(f"工具参数必须是列表: {tool.name}")

        seen_names: set[str] = set()
        for parameter in parameters:
            name = getattr(parameter, "name", None)
            param_type = getattr(parameter, "type", None)
            description = getattr(parameter, "description", None)
            required = getattr(parameter, "required", None)
            item_type = getattr(parameter, "item_type", None)
            if not isinstance(name, str) or not name:
                raise ValueError(f"工具参数名必须是非空字符串: {tool.name}")
            if name in seen_names:
                raise ValueError(f"工具参数名重复: {tool.name}.{name}")
            seen_names.add(name)
            if not isinstance(param_type, str) or not param_type:
                raise ValueError(f"工具参数 type 必须是非空字符串: {tool.name}.{name}")
            if param_type not in _JSON_SCHEMA_TYPES:
                raise ValueError(
                    f"工具参数 type 不是有效 JSON Schema 类型: "
                    f"{tool.name}.{name}={param_type}"
                )
            if not isinstance(description, str):
                raise ValueError(f"工具参数 description 必须是字符串: {tool.name}.{name}")
            if not isinstance(required, bool):
                raise ValueError(f"工具参数 required 必须是布尔值: {tool.name}.{name}")
            if item_type is not None:
                if param_type != "array":
                    raise ValueError(
                        f"只有 array 参数可以声明 item_type: {tool.name}.{name}"
                    )
                if item_type not in _JSON_SCHEMA_TYPES:
                    raise ValueError(
                        f"工具参数 item_type 不是有效 JSON Schema 类型: "
                        f"{tool.name}.{name}={item_type}"
                    )

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_openai_format(self) -> list[dict]:
        """Return tool definitions in OpenAI tool-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def to_anthropic_format(self) -> list[dict]:
        """Return tool definitions in Anthropic tool-use format."""
        return [t.to_anthropic_schema() for t in self._tools.values()]

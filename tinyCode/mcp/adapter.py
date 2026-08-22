"""MCP adapters — wrap MCP tools/resources/prompts as TinyCode BaseTool."""

import hashlib
import re
from collections.abc import Awaitable, Callable
from typing import Any

from tinyCode.mcp.client import MCPClient
from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.validation import require_string


ClientResolver = Callable[[], Awaitable[MCPClient | None]]
_SUPPORTED_PARAMETER_TYPES = {
    "array", "boolean", "integer", "null", "number", "object", "string",
}


def provider_safe_tool_name(server_name: str, remote_name: str) -> str:
    """Build a stable tool name accepted by OpenAI/Anthropic style schemas."""
    raw = f"mcp_{server_name}_{remote_name}"
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if safe == raw and len(safe) <= 64:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:55]}_{digest}"[:64]


class _ClientBackedAdapter:
    def _bind_client(
        self, client: MCPClient, resolver: ClientResolver | None,
    ) -> None:
        self._client = client
        self._server_name = client.server_name
        self._client_resolver = resolver

    async def _resolve_client(self) -> MCPClient:
        if self._client_resolver is None:
            return self._client
        client = await self._client_resolver()
        if client is None:
            raise ConnectionError(f"MCP server '{self._server_name}' 当前不可用")
        self._client = client
        return client


class MCPToolAdapter(_ClientBackedAdapter, BaseTool):
    """Wraps a single MCP tool as a TinyCode BaseTool."""

    def __init__(
        self, client: MCPClient, tool_def: dict,
        client_resolver: ClientResolver | None = None,
    ) -> None:
        self._bind_client(client, client_resolver)
        self._name = tool_def["name"]
        self._public_name = provider_safe_tool_name(
            self._server_name, f"tool_{self._name}",
        )
        self._description = tool_def.get("description", "")
        input_schema = tool_def.get("inputSchema", {})
        self._input_schema = input_schema if isinstance(input_schema, dict) else {}

    @property
    def name(self) -> str:
        return self._public_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WRITE

    @property
    def parameters(self) -> list[ToolParameter]:
        params: list[ToolParameter] = []
        schema_props = self._input_schema.get("properties", {})
        if not isinstance(schema_props, dict):
            schema_props = {}
        required = self._input_schema.get("required", [])
        if not isinstance(required, list):
            required = []
        required_names = {name for name in required if isinstance(name, str)}
        for prop_name, prop_schema in schema_props.items():
            if not isinstance(prop_name, str) or not isinstance(prop_schema, dict):
                continue
            param_type = prop_schema.get("type", "string")
            if (
                not isinstance(param_type, str)
                or param_type not in _SUPPORTED_PARAMETER_TYPES
            ):
                param_type = "string"
            description = prop_schema.get("description", "")
            if not isinstance(description, str):
                description = ""
            params.append(ToolParameter(
                name=prop_name,
                type=param_type,
                description=description,
                required=prop_name in required_names,
            ))
        return params

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = await self._resolve_client()
            text = await client.call_tool(self._name, kwargs)
            return ToolResult(success=True, content=text)
        except Exception as exc:
            return ToolResult(success=False, content="", error=str(exc))


# ---------------------------------------------------------------------------
# Lazy adapters — defer list_resources / list_prompts to first execute()
# ---------------------------------------------------------------------------

class MCPResourceAdapter(_ClientBackedAdapter, BaseTool):
    """Lazy: reads a resource by URI. Discovers available URIs on first call."""

    def __init__(
        self, client: MCPClient, client_resolver: ClientResolver | None = None,
    ) -> None:
        self._bind_client(client, client_resolver)
        self._discovered: list[dict] | None = None  # None = not yet loaded

    @property
    def name(self) -> str:
        return provider_safe_tool_name(self._server_name, "resource")

    @property
    def description(self) -> str:
        if self._discovered is None:
            return f"读取 MCP server '{self._server_name}' 上的资源（首次调用时发现可用资源列表）"
        return self._build_description()

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("uri", "string", "资源 URI，如 file:///path/to/file"),
        ]

    async def execute(self, uri: str) -> ToolResult:
        try:
            uri = require_string(uri, "uri")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        # Lazy discovery on first call
        if self._discovered is None:
            try:
                client = await self._resolve_client()
                self._discovered = await client.list_resources()
            except Exception as exc:
                return ToolResult(success=False, content="", error=f"资源发现失败: {exc}")

        try:
            client = await self._resolve_client()
            data = await client.read_resource(uri)
            return ToolResult(success=True, content=data.get("text", ""))
        except Exception as exc:
            return ToolResult(success=False, content="", error=str(exc))

    def _build_description(self) -> str:
        lines = [f"读取 MCP server '{self._server_name}' 上的资源。可用资源:"]
        for r in (self._discovered or [])[:50]:
            lines.append(f"  - {r.get('uri', '?')} ({r.get('name', '?')})")
        return "\n".join(lines)


class MCPPromptAdapter(_ClientBackedAdapter, BaseTool):
    """Lazy: fetches a prompt template. Discovers available prompts on first call."""

    def __init__(
        self, client: MCPClient, client_resolver: ClientResolver | None = None,
    ) -> None:
        self._bind_client(client, client_resolver)
        self._discovered: list[dict] | None = None  # None = not yet loaded

    @property
    def name(self) -> str:
        return provider_safe_tool_name(self._server_name, "prompt")

    @property
    def description(self) -> str:
        if self._discovered is None:
            return f"获取 MCP server '{self._server_name}' 上的提示词模板（首次调用时发现可用模板列表）"
        return self._build_description()

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("name", "string", "提示词模板名称"),
            ToolParameter("arguments", "string", "模板参数，JSON 格式", required=False),
        ]

    async def execute(self, name: str, arguments: str = "{}") -> ToolResult:
        import json

        try:
            name = require_string(name, "name")
            arguments = require_string(arguments, "arguments")
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        # Lazy discovery on first call
        if self._discovered is None:
            try:
                client = await self._resolve_client()
                self._discovered = await client.list_prompts()
            except Exception as exc:
                return ToolResult(success=False, content="", error=f"提示词发现失败: {exc}")

        try:
            args_dict = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return ToolResult(success=False, content="", error=f"无效的 JSON 参数: {arguments}")
        if not isinstance(args_dict, dict):
            return ToolResult(success=False, content="", error="模板参数必须是 JSON 对象")

        try:
            client = await self._resolve_client()
            result = await client.get_prompt(name, args_dict)
            messages = result.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            parts: list[str] = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown")
                if not isinstance(role, str):
                    role = "unknown"
                content = msg.get("content", "")
                if isinstance(content, dict):
                    content = content.get("text", "")
                if not isinstance(content, str):
                    continue
                parts.append(f"[{role}]: {content}")
            return ToolResult(success=True, content="\n".join(parts))
        except Exception as exc:
            return ToolResult(success=False, content="", error=str(exc))

    def _build_description(self) -> str:
        lines = [f"获取 MCP server '{self._server_name}' 上的提示词模板。可用模板:"]
        for p in (self._discovered or [])[:50]:
            lines.append(f"  - {p.get('name', '?')}: {p.get('description', '?')}")
        return "\n".join(lines)

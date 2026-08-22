"""MCP manager — orchestrates eager connection + discovery + adapter registration."""

import asyncio

from tinyCode.mcp.adapter import MCPPromptAdapter, MCPResourceAdapter, MCPToolAdapter
from tinyCode.mcp.config import MCPServerConfig, load_mcp_config
from tinyCode.mcp.pool import MCPPool
from tinyCode.tools.registry import ToolRegistry


class MCPManager:
    """Connects to all MCP servers at startup, discovers their capabilities,
    and registers adapters into the tool registry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry
        self._pool: MCPPool | None = None
        self._server_configs: list[MCPServerConfig] = []
        self.config_errors: list[str] = []

    # -- initialization -------------------------------------------------------

    def load_config(self, *, include_project: bool = True) -> list[MCPServerConfig]:
        """Load MCP server configs from project + global files."""
        self.config_errors = []
        self._server_configs = load_mcp_config(
            include_project=include_project,
            diagnostics=self.config_errors,
        )
        self._pool = MCPPool(self._server_configs)
        return self._server_configs

    @property
    def is_configured(self) -> bool:
        return len(self._server_configs) > 0

    @property
    def connected_servers(self) -> list[str]:
        if self._pool is None:
            return []
        return self._pool.connected_server_names

    # -- discovery + registration ---------------------------------------------

    async def discover_and_register(self) -> int:
        """Connect to all servers in parallel, then discover tools eagerly.

        Tools are registered immediately so the LLM sees them.  Resource and
        prompt discovery is *deferred* — adapters are registered with lazy
        ``list_resources`` / ``list_prompts`` on first ``execute()``.

        Returns the total number of adapters registered.
        """
        if self._pool is None:
            return 0

        pool = self._pool
        clients = await pool.connect_all()

        async def _discover_one(name: str, client) -> int:
            count = 0
            resolver = lambda: pool.get_client(name)
            try:
                # Tools — eager (LLM needs visibility)
                tools = await client.list_tools()
                for tool_def in tools:
                    if not isinstance(tool_def, dict):
                        continue
                    if not isinstance(tool_def.get("name"), str) or not tool_def["name"]:
                        continue
                    try:
                        self._registry.register(
                            MCPToolAdapter(client, tool_def, resolver)
                        )
                        count += 1
                    except Exception as exc:
                        self._log_discovery_error(name, f"工具 {tool_def.get('name')}", exc)
            except Exception as exc:
                self._log_discovery_error(name, "工具列表", exc)

            # Resources/prompts are independent capabilities. A tools/list
            # failure must not make both disappear.
            for label, adapter in (
                ("资源", MCPResourceAdapter(client, resolver)),
                ("提示词", MCPPromptAdapter(client, resolver)),
            ):
                try:
                    self._registry.register(adapter)
                    count += 1
                except Exception as exc:
                    self._log_discovery_error(name, label, exc)
            return count

        tasks = [_discover_one(name, client) for name, client in clients.items()]
        results = await asyncio.gather(*tasks)
        return sum(results)

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.shutdown()

    @staticmethod
    def _log_discovery_error(name: str, capability: str, exc: Exception) -> None:
        import sys
        print(f"MCP [{name}] {capability}: 注册失败 — {exc}", file=sys.stderr)

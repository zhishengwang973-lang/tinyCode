"""MCP server configuration loading."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MCPServerConfig:
    name: str
    transport: str  # "stdio" | "http"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


def _list_of_strings(value) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return value


def _dict_of_strings(value) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        return None
    return value


def load_mcp_config(
    project_path: Path | None = None,
    global_path: Path | None = None,
    *,
    include_project: bool = True,
    diagnostics: list[str] | None = None,
) -> list[MCPServerConfig]:
    """Load MCP server configurations from project and global files.

    Project config overrides global entries with the same ``name``.
    """
    if project_path is None:
        project_path = Path.cwd() / ".tinyCode-mcp.yaml"
    if global_path is None:
        global_path = Path.home() / ".tinyCode" / "mcp.yaml"

    servers: dict[str, MCPServerConfig] = {}

    # Load global first
    for entry in _load_file(global_path, diagnostics):
        servers[entry.name] = entry

    # Project-scoped MCP configuration can launch local executables.  The
    # application entry point disables it unless the user explicitly trusts
    # the current project; direct/library callers retain the opt-in default.
    if include_project:
        for entry in _load_file(project_path, diagnostics):
            servers[entry.name] = entry

    return list(servers.values())


def _load_file(
    path: Path, diagnostics: list[str] | None = None,
) -> list[MCPServerConfig]:
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        if diagnostics is not None:
            diagnostics.append(f"{path}: 读取或解析失败 — {exc}")
        return []
    if not isinstance(raw, dict):
        if diagnostics is not None:
            diagnostics.append(f"{path}: 顶层必须是 mapping")
        return []
    if "servers" not in raw or not isinstance(raw["servers"], list):
        if diagnostics is not None:
            diagnostics.append(f"{path}: 'servers' 必须是列表")
        return []
    result: list[MCPServerConfig] = []
    for index, entry in enumerate(raw["servers"]):
        if not isinstance(entry, dict):
            if diagnostics is not None:
                diagnostics.append(f"{path}: server #{index} 必须是 mapping，已跳过")
            continue
        config = _parse_server(entry)
        if config:
            result.append(config)
        elif diagnostics is not None:
            name = entry.get("name")
            label = name if isinstance(name, str) and name else f"#{index}"
            diagnostics.append(f"{path}: server {label} 配置无效，已跳过")
    return result


def _parse_server(entry: dict) -> MCPServerConfig | None:
    name = entry.get("name", "")
    transport = entry.get("transport", "")
    if not isinstance(name, str) or not name.strip():
        return None
    if transport not in ("stdio", "http"):
        return None

    command = entry.get("command", "")
    url = entry.get("url", "")
    if transport == "stdio" and (not isinstance(command, str) or not command.strip()):
        return None
    if transport == "http" and (not isinstance(url, str) or not url.strip()):
        return None

    args = entry.get("args", [])
    env = entry.get("env", {})
    headers = entry.get("headers", {})
    timeout = entry.get("timeout", 30.0)
    parsed_args = _list_of_strings(args)
    parsed_env = _dict_of_strings(env)
    parsed_headers = _dict_of_strings(headers)
    if parsed_args is None or parsed_env is None or parsed_headers is None:
        return None
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or not 0 < timeout <= 3600
    ):
        return None

    return MCPServerConfig(
        name=name.strip(),
        transport=transport,
        command=command,
        args=parsed_args,
        env=parsed_env,
        url=url,
        headers=parsed_headers,
        timeout=float(timeout),
    )

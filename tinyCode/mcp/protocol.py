"""JSON-RPC 2.0 message types and helpers."""

import json
from dataclasses import dataclass, field
from typing import Any

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class JSONRPCRequest:
    method: str
    params: dict[str, Any] | None = None
    id: str | int | None = 0
    jsonrpc: str = JSONRPC_VERSION


@dataclass
class JSONRPCResponse:
    id: str | int | None
    result: Any = None
    error: dict | None = None
    jsonrpc: str = JSONRPC_VERSION


@dataclass
class JSONRPCNotification:
    method: str
    params: dict[str, Any] | None = None
    jsonrpc: str = JSONRPC_VERSION


def encode_message(msg: JSONRPCRequest | JSONRPCResponse | JSONRPCNotification) -> str:
    """Encode a JSON-RPC message to a JSON string (single line, no trailing newline)."""
    data: dict[str, Any] = {"jsonrpc": msg.jsonrpc}
    if isinstance(msg, JSONRPCRequest):
        data["id"] = msg.id
        data["method"] = msg.method
        if msg.params is not None:
            data["params"] = msg.params
    elif isinstance(msg, JSONRPCResponse):
        data["id"] = msg.id
        if msg.error is not None:
            data["error"] = msg.error
        else:
            data["result"] = msg.result
    elif isinstance(msg, JSONRPCNotification):
        data["method"] = msg.method
        if msg.params is not None:
            data["params"] = msg.params
    return json.dumps(data, ensure_ascii=False)


def decode_message(line: str) -> JSONRPCRequest | JSONRPCResponse | JSONRPCNotification | None:
    """Decode a JSON-RPC message from a JSON string."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("jsonrpc") != JSONRPC_VERSION:
        return None
    params = data.get("params")
    if params is not None and not isinstance(params, dict):
        return None
    msg_id = data.get("id")
    if "id" in data and not isinstance(msg_id, (str, int)) and msg_id is not None:
        return None
    method = data.get("method")
    if "method" in data and "id" in data:
        if not isinstance(method, str) or not method:
            return None
        return JSONRPCRequest(
            method=method,
            params=params,
            id=msg_id,
        )
    if "method" in data and "id" not in data:
        if not isinstance(method, str) or not method:
            return None
        return JSONRPCNotification(
            method=method,
            params=params,
        )
    if "id" in data and ("result" in data or "error" in data):
        error = data.get("error")
        if error is not None and not isinstance(error, dict):
            return None
        return JSONRPCResponse(
            id=msg_id,
            result=data.get("result"),
            error=error,
        )
    return None

"""Abstract transport for MCP communication."""

from abc import ABC, abstractmethod

from tinyCode.mcp.protocol import JSONRPCRequest, JSONRPCResponse


class BaseTransport(ABC):
    """Abstract transport — stdio, HTTP, or future transports implement this."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the connection."""
        ...

    @abstractmethod
    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Send a request and wait for the matching response."""
        ...

    @abstractmethod
    async def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a one-way notification (no response expected)."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

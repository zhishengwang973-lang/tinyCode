"""Name registry — resolves member names to instance IDs for message routing."""

import asyncio


class NameRegistry:
    """Maps member names → instance IDs. Thread/async-safe."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}  # name → instance_id
        self._lock = asyncio.Lock()

    async def register(self, name: str, instance_id: str) -> None:
        async with self._lock:
            self._entries[name] = instance_id

    async def unregister(self, name: str) -> None:
        async with self._lock:
            self._entries.pop(name, None)

    async def resolve(self, name: str) -> str | None:
        async with self._lock:
            return self._entries.get(name)

    async def list_all(self) -> dict[str, str]:
        async with self._lock:
            return dict(self._entries)

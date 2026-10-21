import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.context import use_workspace
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.workspace_lease import (
    WorkspaceLeaseError,
    WorkspaceWriteLease,
    _path_lock_name,
)


class _PathWriteTool(BaseTool):
    def __init__(self, path: str, entered: asyncio.Event, release: asyncio.Event) -> None:
        self.path = path
        self.entered = entered
        self.release = release

    @property
    def name(self) -> str:
        return "path_write"

    @property
    def description(self) -> str:
        return "test workspace write lease"

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def workspace_lock_paths(self, params: dict) -> tuple[str, ...] | None:
        del params
        return (self.path,)

    async def execute(self) -> ToolResult:
        self.entered.set()
        await self.release.wait()
        return ToolResult(True, self.path)


class WorkspaceLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_path_waits_until_first_writer_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_entered, second_entered = asyncio.Event(), asyncio.Event()
            first_release, second_release = asyncio.Event(), asyncio.Event()
            executor = ToolExecutor()
            with use_workspace(root):
                first = asyncio.create_task(executor.execute(
                    _PathWriteTool("src/app.py", first_entered, first_release), {},
                ))
                await first_entered.wait()
                second = asyncio.create_task(executor.execute(
                    _PathWriteTool("src/app.py", second_entered, second_release), {},
                ))
                await asyncio.sleep(0.12)
                self.assertFalse(second_entered.is_set())

                first_release.set()
                self.assertTrue((await first).success)
                await second_entered.wait()
                second_release.set()
                self.assertTrue((await second).success)

    async def test_different_paths_can_proceed_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_entered, second_entered = asyncio.Event(), asyncio.Event()
            release = asyncio.Event()
            executor = ToolExecutor()
            with use_workspace(root):
                first = asyncio.create_task(executor.execute(
                    _PathWriteTool("src/one.py", first_entered, release), {},
                ))
                second = asyncio.create_task(executor.execute(
                    _PathWriteTool("src/two.py", second_entered, release), {},
                ))
                await asyncio.wait_for(
                    asyncio.gather(first_entered.wait(), second_entered.wait()), timeout=1,
                )
                release.set()
                self.assertTrue((await first).success)
                self.assertTrue((await second).success)

    async def test_whole_workspace_lease_conflicts_with_file_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            async with WorkspaceWriteLease(root, ("src/app.py",)):
                with self.assertRaisesRegex(WorkspaceLeaseError, "工作区写入冲突"):
                    async with WorkspaceWriteLease(root, None, timeout_seconds=0.1):
                        self.fail("global lease must not be acquired")

    async def test_dead_process_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / ".tinyCode" / "locks" / "workspace-write"
            lock_dir.mkdir(parents=True)
            stale = lock_dir / _path_lock_name("src/app.py")
            stale.write_text(json.dumps({"pid": -1, "token": "dead"}), encoding="utf-8")

            async with WorkspaceWriteLease(root, ("src/app.py",), timeout_seconds=0.2):
                owner = json.loads(stale.read_text(encoding="utf-8"))
                self.assertNotEqual("dead", owner["token"])


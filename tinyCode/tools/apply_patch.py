"""Multi-file contextual patch tool with workspace and concurrency guards."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from tinyCode.security.sensitive_paths import is_sensitive_path
from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.validation import require_string


MAX_PATCH_CHARS = 1_000_000
MAX_PATCH_FILES = 50
MAX_FILE_BYTES = 5_000_000
MAX_TOTAL_FILE_BYTES = 20_000_000
_ACTION_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")


@dataclass(frozen=True)
class _Hunk:
    old: str
    new: str


@dataclass(frozen=True)
class _Operation:
    action: str
    path: str
    content: str = ""
    hunks: tuple[_Hunk, ...] = ()


@dataclass(frozen=True)
class _Snapshot:
    exists: bool
    content: bytes
    mode: int
    fingerprint: tuple[int, int, int] | None


def extract_patch_paths(patch: object) -> list[str]:
    """Return declared paths without touching the filesystem."""
    if not isinstance(patch, str):
        return []
    paths: list[str] = []
    for line in patch.splitlines():
        match = _ACTION_RE.fullmatch(line)
        if match:
            paths.append(match.group(2).strip())
    return paths


class ApplyPatchTool(BaseTool):
    """Apply one validated contextual patch across multiple workspace files."""

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "使用一个上下文补丁新增、修改或删除多个工作区文件。"
            "格式必须以 '*** Begin Patch' 开始、'*** End Patch' 结束；"
            "每个操作使用 '*** Add File: path'、'*** Update File: path' 或 "
            "'*** Delete File: path'。Update 中用 @@ 分隔多个 hunk，"
            "上下文行以空格开头，删除行以 - 开头，新增行以 + 开头。"
            "所有内容会先校验，任何一处上下文不匹配都会拒绝整个补丁。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter("patch", "string", "完整的 apply_patch 文本。")]

    async def execute(self, patch: str) -> ToolResult:
        try:
            patch = require_string(patch, "patch")
        except ValueError as exc:
            return ToolResult(False, "", str(exc))
        # Keep bounded writes in the owning task. A cancelled ``to_thread``
        # call cannot stop its worker and could otherwise mutate files after
        # ToolExecutor has already reported a timeout.
        return self._apply_sync(patch)

    def _apply_sync(self, patch: str) -> ToolResult:
        try:
            operations = self._parse(patch)
            prepared = self._prepare(operations)
            self._commit(prepared)
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            return ToolResult(False, "", str(exc))

        counts = {
            action: sum(operation.action == action for operation, *_ in prepared)
            for action in ("Add", "Update", "Delete")
        }
        details = ", ".join(
            f"{label} {counts[action]}"
            for action, label in (("Add", "新增"), ("Update", "修改"), ("Delete", "删除"))
            if counts[action]
        )
        paths = "\n".join(f"- {operation.path}" for operation, *_ in prepared)
        return ToolResult(True, f"补丁已应用（{details}）:\n{paths}")

    def _parse(self, patch: str) -> list[_Operation]:
        if not patch.strip():
            raise ValueError("patch 不能为空")
        if len(patch) > MAX_PATCH_CHARS:
            raise ValueError(f"patch 不能超过 {MAX_PATCH_CHARS} 字符")
        lines = patch.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != "*** Begin Patch":
            raise ValueError("patch 必须以 '*** Begin Patch' 开始")

        operations: list[_Operation] = []
        index = 1
        found_end = False
        while index < len(lines):
            line = lines[index].rstrip("\r\n")
            if line == "*** End Patch":
                if index != len(lines) - 1:
                    trailing = "".join(lines[index + 1:]).strip()
                    if trailing:
                        raise ValueError("'*** End Patch' 之后不能有其他内容")
                found_end = True
                break
            match = _ACTION_RE.fullmatch(line)
            if not match:
                raise ValueError(f"无效的补丁操作行: {line[:200]}")
            action, path = match.group(1), match.group(2).strip()
            index += 1
            body: list[str] = []
            while index < len(lines):
                marker = lines[index].rstrip("\r\n")
                if marker == "*** End Patch" or _ACTION_RE.fullmatch(marker):
                    break
                body.append(lines[index])
                index += 1
            operations.append(self._parse_operation(action, path, body))
            if len(operations) > MAX_PATCH_FILES:
                raise ValueError(f"单个补丁最多操作 {MAX_PATCH_FILES} 个文件")

        if not found_end:
            raise ValueError("patch 必须以 '*** End Patch' 结束")
        if not operations:
            raise ValueError("patch 中没有文件操作")
        paths = [operation.path for operation in operations]
        if len(set(paths)) != len(paths):
            raise ValueError("同一补丁中不能重复操作同一路径")
        return operations

    def _parse_operation(
        self, action: str, path: str, body: list[str],
    ) -> _Operation:
        self._validate_relative_path(path)
        if action == "Add":
            if any(not line.startswith("+") for line in body):
                raise ValueError(f"新增文件 {path} 的每一行都必须以 '+' 开头")
            return _Operation(action, path, content="".join(line[1:] for line in body))
        if action == "Delete":
            if body and any(not line.startswith(("-", " ")) for line in body):
                raise ValueError(f"删除文件 {path} 的校验内容只能使用 '-' 或空格行")
            expected = "".join(line[1:] for line in body)
            return _Operation(action, path, content=expected)

        chunks: list[list[str]] = []
        current: list[str] = []
        for line in body:
            if line.startswith("@@"):
                if current:
                    chunks.append(current)
                    current = []
                continue
            current.append(line)
        if current:
            chunks.append(current)
        if not chunks:
            raise ValueError(f"修改文件 {path} 至少需要一个 hunk")

        hunks: list[_Hunk] = []
        for chunk in chunks:
            if any(not line.startswith((" ", "+", "-")) for line in chunk):
                raise ValueError(f"修改文件 {path} 的 hunk 行必须以空格、'+' 或 '-' 开头")
            old = "".join(line[1:] for line in chunk if not line.startswith("+"))
            new = "".join(line[1:] for line in chunk if not line.startswith("-"))
            if not old:
                raise ValueError(f"修改文件 {path} 的 hunk 必须包含可唯一定位的原文")
            if old == new:
                raise ValueError(f"修改文件 {path} 的 hunk 没有产生变化")
            hunks.append(_Hunk(old, new))
        return _Operation(action, path, hunks=tuple(hunks))

    def _prepare(
        self, operations: list[_Operation],
    ) -> list[tuple[_Operation, Path, _Snapshot, bytes | None]]:
        root = get_workspace_root()
        prepared: list[tuple[_Operation, Path, _Snapshot, bytes | None]] = []
        total_bytes = 0
        for operation in operations:
            resolved = self._resolve(root, operation.path)
            snapshot = self._snapshot(resolved)
            original_size = (
                snapshot.fingerprint[1]
                if snapshot.fingerprint is not None
                else 0
            )
            total_bytes += original_size
            if total_bytes > MAX_TOTAL_FILE_BYTES:
                raise ValueError(
                    f"补丁涉及的原文件总量不能超过 {MAX_TOTAL_FILE_BYTES} 字节"
                )

            if operation.action == "Add":
                if snapshot.exists:
                    raise ValueError(f"新增失败，文件已存在: {operation.path}")
                new_content = operation.content.encode("utf-8")
            else:
                if not snapshot.exists:
                    raise ValueError(f"文件不存在: {operation.path}")
                if original_size > MAX_FILE_BYTES:
                    raise ValueError(
                        f"文件过大，apply_patch 最大支持 {MAX_FILE_BYTES} 字节: "
                        f"{operation.path}"
                    )
                try:
                    current = snapshot.content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"文件不是 UTF-8 文本: {operation.path}") from exc
                if operation.action == "Delete":
                    if operation.content and operation.content != current:
                        raise ValueError(f"删除失败，文件内容与补丁不一致: {operation.path}")
                    new_content = None
                else:
                    for hunk_index, hunk in enumerate(operation.hunks, start=1):
                        count = current.count(hunk.old)
                        if count == 0:
                            raise ValueError(
                                f"补丁上下文未匹配: {operation.path} hunk #{hunk_index}"
                            )
                        if count > 1:
                            raise ValueError(
                                f"补丁上下文不唯一: {operation.path} hunk #{hunk_index}"
                            )
                        current = current.replace(hunk.old, hunk.new, 1)
                    new_content = current.encode("utf-8")
            if new_content is not None and len(new_content) > MAX_FILE_BYTES:
                raise ValueError(
                    f"补丁后文件超过 {MAX_FILE_BYTES} 字节: {operation.path}"
                )
            prepared.append((operation, resolved, snapshot, new_content))
        return prepared

    def _commit(
        self,
        prepared: list[tuple[_Operation, Path, _Snapshot, bytes | None]],
    ) -> None:
        staged: dict[Path, Path] = {}
        touched: list[tuple[Path, _Snapshot]] = []
        try:
            for operation, path, snapshot, new_content in prepared:
                self._assert_unchanged(path, snapshot)
                if new_content is None:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    "wb", dir=path.parent, prefix=f".{path.name}.",
                    suffix=".tmp", delete=False,
                ) as handle:
                    handle.write(new_content)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temp_path = Path(handle.name)
                os.chmod(temp_path, snapshot.mode if snapshot.exists else 0o644)
                staged[path] = temp_path

            # Recheck every target after staging, immediately before mutation.
            for _, path, snapshot, _ in prepared:
                self._assert_unchanged(path, snapshot)

            for _, path, snapshot, new_content in prepared:
                touched.append((path, snapshot))
                if new_content is None:
                    path.unlink()
                else:
                    os.replace(staged.pop(path), path)
                self._fsync_parent(path.parent)
        except BaseException as exc:
            rollback_errors: list[str] = []
            for path, snapshot in reversed(touched):
                try:
                    if snapshot.exists:
                        self._atomic_write(path, snapshot.content, snapshot.mode)
                    else:
                        path.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{path}: {rollback_exc}")
            detail = f"补丁写入失败: {type(exc).__name__}: {exc}"
            if rollback_errors:
                detail += "; 回滚部分失败: " + "; ".join(rollback_errors)
            raise RuntimeError(detail) from exc
        finally:
            for temp_path in staged.values():
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _validate_relative_path(path: str) -> None:
        if not path or "\x00" in path:
            raise ValueError("补丁文件路径不能为空")
        candidate = Path(path)
        if candidate.is_absolute():
            raise ValueError(f"不允许绝对路径: {path}")
        if ".." in candidate.parts:
            raise ValueError(f"路径遍历不被允许: {path}")
        if is_sensitive_path(path):
            raise ValueError(f"拒绝修改包含模型凭据的本地配置文件: {path}")

    def _resolve(self, root: Path, path: str) -> Path:
        self._validate_relative_path(path)
        resolved = (root / path).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"路径遍历不被允许: {path}") from exc
        if resolved.exists() and not resolved.is_file():
            raise ValueError(f"路径不是普通文件: {path}")
        return resolved

    @staticmethod
    def _snapshot(path: Path) -> _Snapshot:
        if not path.exists():
            return _Snapshot(False, b"", 0o644, None)
        stat = path.stat()
        if stat.st_size > MAX_FILE_BYTES:
            # Keep the read bounded; _prepare emits the user-facing error.
            return _Snapshot(
                True, b"", stat.st_mode & 0o7777,
                (stat.st_ino, stat.st_size, stat.st_mtime_ns),
            )
        content = path.read_bytes()
        return _Snapshot(
            True, content, stat.st_mode & 0o7777,
            (stat.st_ino, stat.st_size, stat.st_mtime_ns),
        )

    @staticmethod
    def _assert_unchanged(path: Path, snapshot: _Snapshot) -> None:
        if not snapshot.exists:
            if path.exists():
                raise RuntimeError(f"文件在补丁期间被创建，已拒绝覆盖: {path}")
            return
        try:
            stat = path.stat()
        except OSError as exc:
            raise RuntimeError(f"文件在补丁期间消失: {path}") from exc
        current = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current != snapshot.fingerprint:
            raise RuntimeError(f"文件在补丁期间被其他进程修改，已拒绝覆盖: {path}")

    @classmethod
    def _atomic_write(cls, path: Path, content: bytes, mode: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.chmod(temp_path, mode)
            os.replace(temp_path, path)
            cls._fsync_parent(path.parent)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_parent(parent: Path) -> None:
        try:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

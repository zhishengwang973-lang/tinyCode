"""Lightweight per-turn workspace change detection."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


# These trees are either version-control metadata, dependency installations,
# or generated caches. Walking them can turn a small task into hundreds of
# thousands of stat calls, while their contents are not useful source changes.
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "venv",
}
_INTERNAL_RESULT_DIR = (".tinyCode", "tool_results")


@dataclass(frozen=True)
class _FileFingerprint:
    kind: int
    size: int
    modified_ns: int
    changed_ns: int
    mode: int
    inode: int
    symlink_target: str = ""


@dataclass(frozen=True)
class WorkspaceChanges:
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.added or self.modified or self.deleted)


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """A point-in-time map of project-local files.

    Fingerprints use filesystem metadata instead of reading every file. TinyCode's
    own file tools replace files atomically, and ordinary command-line writers
    update mtime/ctime, so changes are detected without making large repositories
    noticeably slower.
    """

    root: Path
    files: dict[str, _FileFingerprint]

    @classmethod
    def capture(cls, root: Path) -> "WorkspaceSnapshot":
        resolved = root.resolve()
        return cls(root=resolved, files=_scan_files(resolved))

    def compare(self) -> WorkspaceChanges:
        current = _scan_files(self.root)
        before_paths = set(self.files)
        current_paths = set(current)
        return WorkspaceChanges(
            added=tuple(sorted(current_paths - before_paths)),
            modified=tuple(sorted(
                path
                for path in before_paths & current_paths
                if self.files[path] != current[path]
            )),
            deleted=tuple(sorted(before_paths - current_paths)),
        )


def _scan_files(root: Path) -> dict[str, _FileFingerprint]:
    files: dict[str, _FileFingerprint] = {}
    pending: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    while pending:
        directory, relative_parts = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            # A concurrent tool may remove or temporarily lock an entry. The
            # next snapshot will still report all paths that remain observable.
            continue
        for entry in entries:
            parts = relative_parts + (entry.name,)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_directory:
                if _should_skip_directory(parts):
                    continue
                pending.append((Path(entry.path), parts))
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
                target = os.readlink(entry.path) if stat.S_ISLNK(metadata.st_mode) else ""
            except OSError:
                continue
            files[Path(*parts).as_posix()] = _FileFingerprint(
                kind=stat.S_IFMT(metadata.st_mode),
                size=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
                changed_ns=metadata.st_ctime_ns,
                mode=stat.S_IMODE(metadata.st_mode),
                inode=metadata.st_ino,
                symlink_target=target,
            )
    return files


def _should_skip_directory(parts: tuple[str, ...]) -> bool:
    if parts[-1] in _EXCLUDED_DIR_NAMES:
        return True
    return parts[:len(_INTERNAL_RESULT_DIR)] == _INTERNAL_RESULT_DIR

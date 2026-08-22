"""Durable JSONL journal primitives used by session persistence."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        os.replace(tmp_path, path)
        # fsync(file) makes the bytes durable; fsync(parent) makes the rename
        # durable across a sudden power loss on filesystems that support it.
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some platforms do not allow opening/fsyncing directories.
            pass
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                # Cleanup failure must not hide the original write failure.
                pass


class JSONLJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in rows
        ) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def replace(self, rows: list[dict[str, Any]]) -> None:
        payload = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in rows
        )
        atomic_write_text(self.path, payload + ("\n" if rows else ""))

    def read_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

"""Tool result truncator — layer 1 of token management.

Truncates oversized individual tool results and caps total size across
all tool results in a single conversation round.
"""

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tinyCode.providers.base import Message

TOOL_RESULT_STORAGE_SUBDIR = Path(".tinyCode") / "tool_results"
# Keep ordinary tool observations compact enough to be repeated across a few
# ReAct turns.  The complete result is persisted locally and can be searched
# or read in pages, so this does not discard information.
DEFAULT_PER_RESULT_THRESHOLD = 16_000


def default_storage_dir(project_root: Path | None = None) -> Path:
    """Return the project-local cache directory for oversized tool results."""
    root = project_root or Path.cwd()
    return (root / TOOL_RESULT_STORAGE_SUBDIR).resolve()


# Backward-compatible import for callers that treated this module constant as
# the startup project's default. Internal callers use ``default_storage_dir``
# so the path is resolved when their instance is created.
DEFAULT_STORAGE_DIR = default_storage_dir()


@dataclass
class TruncateConfig:
    per_result_threshold: int = DEFAULT_PER_RESULT_THRESHOLD
    total_round_threshold: int = 64_000      # chars — total tool-result context budget
    preview_length: int = 2_000              # chars of preview kept in-conversation
    storage_dir: Path = field(default_factory=default_storage_dir)


class ToolResultTruncator:
    """Scans conversation messages for oversized tool results, writes full
    content to disk, and replaces them with previews."""

    def __init__(self, config: TruncateConfig | None = None) -> None:
        self._cfg = config or TruncateConfig()
        self._sequence = 0
        self._stored_results: dict[str, Path] = {}
        self.storage_error = ""
        self.set_storage_dir(self._cfg.storage_dir)

    @property
    def storage_dir(self) -> Path:
        return self._cfg.storage_dir

    @property
    def has_available_results(self) -> bool:
        """Whether the active project's cache contains a readable result file.

        Disk is the source of truth here so persisted results remain available
        after the process or session is reopened. ``_stored_results`` is only
        an in-process index used to avoid rewriting identical content.
        """
        try:
            return any(path.is_file() for path in self._cfg.storage_dir.glob("*.txt"))
        except OSError:
            return False

    def set_storage_dir(self, storage_dir: Path) -> None:
        """Switch cache roots, for example after entering another worktree."""
        self._cfg.storage_dir = storage_dir.resolve()
        self._sequence = 0
        self._stored_results.clear()
        self.storage_error = ""
        try:
            self._cfg.storage_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.storage_error = f"{type(exc).__name__}: {exc}"

    def set_project_root(self, project_root: Path) -> None:
        self.set_storage_dir(default_storage_dir(project_root))

    # -- public API -----------------------------------------------------------

    def process_round(self, messages: list[Message]) -> tuple[list[Message], list[dict]]:
        """Like ``process`` but additionally enforces the round-level cap.

        Returns ``(truncated_messages, truncation_infos)`` where each info
        dict has keys ``tool_name``, ``original_chars``, ``file_path``.
        """
        infos: list[dict] = []

        # First pass: identify tool messages and their sizes
        tool_indices: list[tuple[int, int]] = []  # (index, char_count)
        for i, msg in enumerate(messages):
            content = self._tool_result_content(msg)
            if content is not None:
                tool_indices.append((i, len(content)))

        # Enforce the per-result limit first, then keep selecting the largest
        # remaining results until the projected model-visible total is within
        # budget.  The old calculation subtracted down to the threshold even
        # though a truncated result is actually only a preview; with several
        # medium results it could select every result and still believe it was
        # over budget.
        replacement_budget = self._cfg.preview_length + 512
        to_truncate = {
            idx for idx, chars in tool_indices
            if chars > self._cfg.per_result_threshold
        }
        projected_total = sum(
            min(chars, replacement_budget) if idx in to_truncate else chars
            for idx, chars in tool_indices
        )

        if projected_total > self._cfg.total_round_threshold:
            for idx, chars in sorted(tool_indices, key=lambda item: item[1], reverse=True):
                if idx in to_truncate or chars <= replacement_budget:
                    continue
                to_truncate.add(idx)
                projected_total -= chars - replacement_budget
                if projected_total <= self._cfg.total_round_threshold:
                    break

        if not to_truncate:
            return list(messages), infos

        result: list[Message] = []
        for i, msg in enumerate(messages):
            if i in to_truncate and self._tool_result_content(msg) is not None:
                truncated, file_path = self._truncate_tool_msg_with_path(msg)
                result.append(truncated)
                infos.append({
                    "tool_name": self._tool_name(msg),
                    "original_chars": len(self._tool_result_content(msg) or ""),
                    "file_path": file_path,
                })
            else:
                result.append(msg)
        return result, infos

    # -- internals ------------------------------------------------------------

    def _truncate_tool_msg_with_path(self, msg: Message) -> tuple[Message, str]:
        """Truncate and return (message, file_path)."""
        content = self._tool_result_content(msg) or ""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        tool_name = self._tool_name(msg)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)
        digest = hashlib.sha256(content.encode("utf-8", errors="surrogatepass")).hexdigest()
        cached = self._stored_results.get(digest)
        if not self.storage_error and cached is not None and cached.exists():
            file_path = str(cached)
        elif not self.storage_error:
            self._sequence += 1
            stored = self._cfg.storage_dir / f"{ts}_{safe_name}_{self._sequence}_{digest[:12]}.txt"
            try:
                stored.write_text(content, encoding="utf-8")
                self._stored_results[digest] = stored
                file_path = str(stored)
            except OSError as exc:
                self.storage_error = f"{type(exc).__name__}: {exc}"
                file_path = ""
        else:
            file_path = ""
        preview = content[:self._cfg.preview_length]
        storage_note = (
            f"完整内容已保存到磁盘\n文件: {file_path}\n"
            "可使用 tool_result_search 搜索，或用 tool_result_read 分段读取"
            if file_path
            else f"缓存目录不可写，完整内容未保存（{self.storage_error}）"
        )
        truncated_text = (
            f"[工具结果过大，{storage_note}]\n"
            f"预览（前 {self._cfg.preview_length} 字符）:\n{preview}\n"
            f"...（省略 {len(content) - self._cfg.preview_length} 字符）"
        )
        truncated_msg = self._replace_tool_result_content(msg, truncated_text)
        return truncated_msg, file_path

    def _tool_result_content(self, msg: Message) -> str | None:
        content = msg.get("content", "")
        if msg.get("role") == "tool" and isinstance(content, str):
            return content
        if msg.get("role") == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block_content = block.get("content", "")
                    if isinstance(block_content, str):
                        return block_content
        return None

    def _replace_tool_result_content(self, msg: Message, content: str) -> Message:
        if msg.get("role") == "tool":
            return {**msg, "content": content}

        new_content = []
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                new_content.append({**block, "content": content})
            else:
                new_content.append(block)
        return {**msg, "content": new_content}

    def _tool_name(self, msg: Message) -> str:
        if msg.get("name"):
            return str(msg["name"])
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return str(block.get("tool_use_id", "unknown"))
        return "unknown"

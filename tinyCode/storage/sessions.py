"""Session persistence — JSONL append + meta file + recovery.

Each session is stored as:
  - ``{sessions_dir}/{id}.jsonl`` — append-only message log
  - ``{sessions_dir}/{id}.meta.json`` — summary for listing

On load: corrupt lines are skipped, unpaired tool_use triggers truncation,
token overflow triggers compression, and time gaps inject reminders.
"""

import json
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tinyCode.conversation.history import ConversationHistory
from tinyCode.storage.journal import JSONLJournal, atomic_write_text as _atomic_write_text

SESSIONS_DIR = Path.home() / ".tinyCode" / "sessions"
TIME_GAP_MINUTES = 30  # inject reminder after this inactivity
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class SessionStore:
    """JSONL-backed session persistence."""

    def __init__(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._current_id: str | None = None
        self._persisted_messages: list[dict[str, Any]] = []
        self._persisted_timestamps: list[str] = []
        self.last_migration_error = ""

    # -- create / switch ------------------------------------------------------

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._current_id = sid
        self._persisted_messages = []
        self._persisted_timestamps = []
        _atomic_write_text(self._path_for(sid), "")
        self._write_meta(sid, created=True)
        return sid

    @property
    def current_id(self) -> str | None:
        return self._current_id

    # -- append ---------------------------------------------------------------

    def append_message(self, message: dict[str, Any]) -> None:
        """Append a single message to the JSONL (O(1) write)."""
        if not self._current_id:
            self.new_session()
        sid = self._current_id
        assert sid is not None
        timestamp = datetime.now(timezone.utc).isoformat()
        JSONLJournal(self._path_for(sid)).append([{**message, "timestamp": timestamp}])
        self._persisted_messages.append(self._without_timestamp(message))
        self._persisted_timestamps.append(timestamp)
        self._update_meta(sid, message_count_delta=1)

    def append_messages(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple messages in one write."""
        if not messages:
            return
        if not self._current_id:
            self.new_session()
        sid = self._current_id
        assert sid is not None
        ts = datetime.now(timezone.utc).isoformat()
        JSONLJournal(self._path_for(sid)).append([
            {**message, "timestamp": ts} for message in messages
        ])
        self._persisted_messages.extend(self._without_timestamp(m) for m in messages)
        self._persisted_timestamps.extend([ts] * len(messages))
        self._update_meta(sid, message_count_delta=len(messages))

    # -- save full history (compat) -------------------------------------------

    def save(
        self,
        history: ConversationHistory,
        provider_name: str,
        model: str,
        session_name: str = "default",
    ) -> None:
        """Full save — overwrites JSONL with current history.

        Called at the end of each exchange as a safety net.
        """
        if not self._current_id:
            sid = session_name if session_name != "default" else uuid.uuid4().hex[:12]
            self._current_id = sid
        sid = self._current_id

        msgs = [
            self._without_timestamp(message)
            for message in history.get_messages()
            if not self._is_transient_time_gap(message)
        ]
        prefix_matches = (
            len(msgs) >= len(self._persisted_messages)
            and msgs[:len(self._persisted_messages)] == self._persisted_messages
        )
        if prefix_matches:
            tail = msgs[len(self._persisted_messages):]
            if tail:
                self.append_messages(tail)
        else:
            timestamps = self._timestamps_for_rewrite(msgs)
            JSONLJournal(self._path_for(sid)).replace([
                {**message, "timestamp": timestamp}
                for message, timestamp in zip(msgs, timestamps)
            ])
            self._persisted_messages = list(msgs)
            self._persisted_timestamps = timestamps

        self._write_meta(sid, provider=provider_name, model=model,
                         message_count=len(msgs))

    # -- load / recover -------------------------------------------------------

    def load(
        self, session_name: str = "default",
    ) -> tuple[ConversationHistory, str, str] | None:
        """Load a session with recovery.

        Returns ``(history, provider_name, model)`` or None.
        """
        sid = self._resolve_id(session_name)
        if sid is None:
            return None

        file_path = self._path_for(sid)
        if not file_path.exists():
            return None

        history = ConversationHistory()
        messages: list[dict] = []

        messages = self._sanitize_loaded_messages(JSONLJournal(file_path).read_rows())

        # Truncate at unpaired tool_use
        messages = self._truncate_unpaired(messages)

        persisted_messages = [self._without_timestamp(message) for message in messages]
        persisted_timestamps = [
            str(message.get("timestamp", "")) for message in messages
        ]
        messages = self._insert_time_gaps(messages)

        # Reconstruct history
        for msg in messages:
            clean_msg = self._without_timestamp(msg)
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                if "[Prior conversation summary]" in str(content):
                    history.add_context_message(content)
                elif "[结构化摘要]" in str(content):
                    history.add_context_message(content)
                elif "[时间跨度提醒]" in str(content):
                    history.add_context_message(content)
                # Skip old system prompts
            elif role == "user":
                if isinstance(content, str):
                    history.add_user_message(content)
                else:
                    history.add_raw_message(clean_msg)
            elif role == "assistant":
                if msg.get("tool_calls") or isinstance(content, list):
                    history.add_raw_message(clean_msg)
                elif isinstance(content, str) and content:
                    history.add_assistant_message(content)
            elif role == "tool":
                history.add_raw_message(clean_msg)

        self._current_id = sid
        self._persisted_messages = persisted_messages
        self._persisted_timestamps = persisted_timestamps

        # Read meta for provider/model
        meta = self._read_meta(sid)
        provider = meta.get("provider", "")
        model = meta.get("model", "")

        return history, provider, model

    # -- list sessions --------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """Return summary of all sessions from meta files."""
        if not SESSIONS_DIR.exists():
            return []
        entries: list[tuple[float, dict]] = []
        seen: set[str] = set()
        try:
            meta_files = sorted(SESSIONS_DIR.glob("*.meta.json"), reverse=True)
            journal_files = sorted(SESSIONS_DIR.glob("*.jsonl"), reverse=True)
        except OSError:
            return []
        for f in meta_files:
            sid = f.stem.replace(".meta", "")
            try:
                if not self._path_for(sid).exists():
                    continue
                modified = f.stat().st_mtime
            except OSError:
                continue
            meta = self._read_meta(sid)
            if meta:
                entries.append((modified, meta))
                seen.add(sid)
        for f in journal_files:
            if f.stem in seen:
                continue
            try:
                modified = f.stat().st_mtime
            except OSError:
                continue
            try:
                derived = self._derive_meta_from_jsonl(f.stem)
            except (OSError, UnicodeError):
                continue
            entries.append((modified, derived))
        return [meta for _, meta in sorted(entries, key=lambda item: item[0], reverse=True)]

    def get_session_title(self, sid: str) -> str | None:
        meta = self._read_meta(sid)
        return meta.get("title") if meta else None

    # -- migration ------------------------------------------------------------

    def migrate_old_format(self) -> bool:
        """Migrate ``default.json`` to JSONL format. Returns True if migrated."""
        self.last_migration_error = ""
        old_path = SESSIONS_DIR / "default.json"
        if not old_path.exists():
            return False
        try:
            data = json.loads(old_path.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            provider = data.get("provider", "")
            model = data.get("model", "")

            sid = uuid.uuid4().hex[:12]
            ts = datetime.now(timezone.utc).isoformat()
            new_path = self._path_for(sid)
            JSONLJournal(new_path).replace([
                {**message, "timestamp": ts} for message in messages
            ])

            title = self._guess_title(messages)
            self._write_meta(sid, provider=provider, model=model,
                             message_count=len(messages), title=title)
            self._current_id = sid
            old_path.rename(old_path.with_suffix(".json.bak"))
            return True
        except Exception as exc:
            self.last_migration_error = f"{type(exc).__name__}: {exc}"
            return False

    # -- internals ------------------------------------------------------------

    def _path_for(self, sid: str) -> Path:
        safe = sid.replace("\\", "_").replace("/", "_")
        return SESSIONS_DIR / f"{safe}.jsonl"

    def _meta_path_for(self, sid: str) -> Path:
        safe = sid.replace("\\", "_").replace("/", "_")
        return SESSIONS_DIR / f"{safe}.meta.json"

    def delete(self, session_id: str) -> bool:
        """Delete a session and its meta file from disk."""
        sid = self._resolve_id(session_id)
        if sid is None:
            return False
        jsonl = self._path_for(sid)
        meta = self._meta_path_for(sid)
        deleted = False
        if jsonl.exists():
            jsonl.unlink()
            deleted = True
        if meta.exists():
            meta.unlink()
            deleted = True
        if deleted and self._current_id == sid:
            self._current_id = None
            self._persisted_messages = []
            self._persisted_timestamps = []
        return deleted

    def _resolve_id(self, session_name: str) -> str | None:
        if session_name != "default":
            if not isinstance(session_name, str) or not _SESSION_ID_RE.fullmatch(
                session_name
            ):
                return None
            # Support partial ID matching (prefix)
            jsonl = SESSIONS_DIR / f"{session_name}.jsonl"
            if jsonl.exists():
                return session_name
            # Try prefix match
            matches = [
                path for path in SESSIONS_DIR.glob("*.jsonl")
                if path.stem.startswith(session_name)
            ]
            if len(matches) == 1:
                return matches[0].stem
            elif len(matches) > 1:
                return None  # ambiguous
            return None
        # Find most recent session
        try:
            metas = sorted(
                SESSIONS_DIR.glob("*.meta.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            metas = []
        for meta in metas:
            sid = meta.stem.replace(".meta", "")
            if self._path_for(sid).exists():
                return sid
        try:
            jsonls = sorted(
                SESSIONS_DIR.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            jsonls = []
        if jsonls:
            return jsonls[0].stem
        return None

    def _write_meta(self, sid: str, **kwargs) -> None:
        meta_path = self._meta_path_for(sid)
        existing = self._read_meta(sid)
        now = datetime.now(timezone.utc).isoformat()
        if kwargs.pop("created", False):
            existing["id"] = sid
            existing["created_at"] = now
        existing["last_active_at"] = now
        for k, v in kwargs.items():
            existing[k] = v
        _atomic_write_text(
            meta_path,
            json.dumps(existing, ensure_ascii=False, indent=2),
        )

    def _update_meta(self, sid: str, message_count_delta: int = 0) -> None:
        meta = self._read_meta(sid)
        meta["last_active_at"] = datetime.now(timezone.utc).isoformat()
        if message_count_delta:
            meta["message_count"] = self._message_count(meta) + message_count_delta
        _atomic_write_text(
            self._meta_path_for(sid),
            json.dumps(meta, ensure_ascii=False, indent=2),
        )

    def _read_meta(self, sid: str) -> dict:
        meta_path = self._meta_path_for(sid)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    return meta
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        return {"id": sid, "message_count": 0}

    def _derive_meta_from_jsonl(self, sid: str) -> dict:
        messages: list[dict] = []
        jsonl_path = self._path_for(sid)
        if not jsonl_path.exists():
            return {"id": sid, "message_count": 0}
        for message in JSONLJournal(jsonl_path).read_rows():
            if self._is_message_row(message):
                messages.append(message)

        meta = {"id": sid, "message_count": len(messages)}
        title = self._guess_title(messages)
        if title:
            meta["title"] = title
        if messages:
            last_active = messages[-1].get("timestamp", "")
            if isinstance(last_active, str):
                meta["last_active_at"] = last_active
        return meta

    @staticmethod
    def _message_count(meta: dict) -> int:
        count = meta.get("message_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return 0
        return count

    @staticmethod
    def _is_message_row(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return value.get("role") in {"system", "user", "assistant", "tool"}

    @staticmethod
    def _sanitize_loaded_messages(messages: list[dict]) -> list[dict]:
        """Keep only provider-safe rows from a potentially corrupted journal."""
        sanitized: list[dict] = []
        for raw in messages:
            role = raw.get("role")
            content = raw.get("content")
            if role == "system":
                if not isinstance(content, str):
                    continue
                clean = {"role": role, "content": content}
            elif role == "user":
                if not isinstance(content, (str, list)):
                    continue
                if isinstance(content, list) and not all(
                    isinstance(block, dict) for block in content
                ):
                    continue
                clean = {"role": role, "content": content}
            elif role == "assistant":
                if content is not None and not isinstance(content, (str, list)):
                    continue
                if isinstance(content, list) and not all(
                    isinstance(block, dict) for block in content
                ):
                    continue
                tool_calls = raw.get("tool_calls")
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        continue
                    valid_calls: list[dict] = []
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        call_id = call.get("id")
                        function = call.get("function")
                        if (
                            not isinstance(call_id, str) or not call_id
                            or not isinstance(function, dict)
                            or not isinstance(function.get("name"), str)
                            or not function["name"]
                            or not isinstance(function.get("arguments"), str)
                        ):
                            continue
                        valid_calls.append(call)
                    tool_calls = valid_calls
                if not content and not tool_calls:
                    continue
                clean = {"role": role, "content": content}
                if tool_calls:
                    clean["tool_calls"] = tool_calls
            elif role == "tool":
                call_id = raw.get("tool_call_id")
                if (
                    not isinstance(call_id, str) or not call_id
                    or not isinstance(content, str)
                ):
                    continue
                clean = {
                    "role": role,
                    "tool_call_id": call_id,
                    "content": content,
                }
                name = raw.get("name")
                if isinstance(name, str) and name:
                    clean["name"] = name
            else:
                continue

            timestamp = raw.get("timestamp")
            if isinstance(timestamp, str):
                clean["timestamp"] = timestamp
            sanitized.append(clean)
        return sanitized

    @staticmethod
    def _truncate_unpaired(messages: list[dict]) -> list[dict]:
        """Remove orphan results and recover a partial final tool batch."""
        # id -> (assistant index, protocol style, tool name)
        open_tool_calls: dict[str, tuple[int, str, str]] = {}
        cleaned: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                content = msg.get("content", "")
                message_index = len(cleaned)
                # Anthropic style: content is list with tool_use blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            call_id = block.get("id", "")
                            if call_id:
                                open_tool_calls[call_id] = (
                                    message_index,
                                    "anthropic",
                                    str(block.get("name", "")),
                                )
                # OpenAI style: tool_calls field
                if not isinstance(tool_calls, list):
                    tool_calls = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    call_id = tc.get("id", "")
                    if call_id:
                        function = tc.get("function", {})
                        name = function.get("name", "") if isinstance(function, dict) else ""
                        open_tool_calls[call_id] = (message_index, "openai", str(name))
                cleaned.append(msg)

            elif role == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id not in open_tool_calls:
                    continue
                del open_tool_calls[tc_id]
                cleaned.append(msg)

            elif role == "user" and isinstance(msg.get("content"), list):
                result_ids = [
                    block.get("tool_use_id", "")
                    for block in msg["content"]
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                ]
                if result_ids:
                    if not all(call_id in open_tool_calls for call_id in result_ids):
                        continue
                    for call_id in result_ids:
                        del open_tool_calls[call_id]
                cleaned.append(msg)

            else:
                cleaned.append(msg)

        if open_tool_calls:
            first_unpaired_idx = min(value[0] for value in open_tool_calls.values())
            tail = cleaned[first_unpaired_idx + 1:]
            tail_is_only_results = all(
                msg.get("role") == "tool"
                or (
                    msg.get("role") == "user"
                    and isinstance(msg.get("content"), list)
                    and all(
                        isinstance(block, dict) and block.get("type") == "tool_result"
                        for block in msg["content"]
                    )
                )
                for msg in tail
            )
            if not tail_is_only_results:
                return cleaned[:first_unpaired_idx]

            # A crash may happen after some tools in a batch completed.  Keep
            # those durable results and synthesize failures for calls that had
            # not completed, producing a valid provider message sequence.
            anthropic_blocks: list[dict] = []
            for call_id, (_, style, name) in open_tool_calls.items():
                recovery_text = "[会话恢复] 上次运行在该工具返回结果前中断；该操作状态未知，请先检查再决定是否重试。"
                if style == "openai":
                    cleaned.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": recovery_text,
                    })
                else:
                    anthropic_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": recovery_text,
                        "is_error": True,
                    })
            if anthropic_blocks:
                cleaned.append({"role": "user", "content": anthropic_blocks})
        return cleaned

    @staticmethod
    def _insert_time_gaps(messages: list[dict]) -> list[dict]:
        """Insert gap reminders at their chronological position."""
        result: list[dict] = []
        prev_ts: datetime | None = None
        for msg in messages:
            ts_str = msg.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                result.append(msg)
                continue
            if ts.tzinfo is None:
                # Legacy sessions may contain naive ISO timestamps. Treat
                # those as UTC so mixing old/new rows cannot raise while
                # calculating a gap.
                ts = ts.replace(tzinfo=timezone.utc)
            if prev_ts and (ts - prev_ts) > timedelta(minutes=TIME_GAP_MINUTES):
                delta = ts - prev_ts
                hours = delta.total_seconds() / 3600
                text = f"[时间跨度提醒] 距上次活跃约 {hours:.1f} 小时，以下是新消息。"
                result.append({
                    "role": "system",
                    "content": text,
                })
            result.append(msg)
            prev_ts = ts
        return result

    @staticmethod
    def _without_timestamp(message: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in message.items() if key != "timestamp"}

    @staticmethod
    def _is_transient_time_gap(message: dict[str, Any]) -> bool:
        return (
            message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("[时间跨度提醒]")
        )

    def _timestamps_for_rewrite(self, messages: list[dict[str, Any]]) -> list[str]:
        existing: dict[str, deque[str]] = defaultdict(deque)
        for message, timestamp in zip(
            self._persisted_messages, self._persisted_timestamps,
        ):
            key = json.dumps(message, ensure_ascii=False, sort_keys=True)
            existing[key].append(timestamp)
        now = datetime.now(timezone.utc).isoformat()
        timestamps: list[str] = []
        for message in messages:
            key = json.dumps(message, ensure_ascii=False, sort_keys=True)
            timestamps.append(existing[key].popleft() if existing[key] else now)
        return timestamps

    @staticmethod
    def _guess_title(messages: list[dict]) -> str:
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content[:60]
        return "未命名会话"

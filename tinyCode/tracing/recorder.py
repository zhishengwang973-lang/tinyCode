"""Crash-tolerant, project-local JSONL execution tracing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import threading
import uuid
import webbrowser
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic_ns, time_ns
from typing import Any

from tinyCode.config.models import TracingConfig
from tinyCode.time_utils import BEIJING_TIMEZONE, beijing_now, beijing_now_iso


TRACE_SCHEMA_VERSION = 1
MAX_TRACE_BYTES = 10_000_000
MAX_ATTRIBUTE_DEPTH = 5
MAX_COLLECTION_ITEMS = 100
MAX_STRING_CHARS = 2_000
_SENSITIVE_KEY_RE = re.compile(
    r"api.?key|token|secret|password|credential|authorization|cookie",
    re.IGNORECASE,
)
_PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$")


@dataclass
class TraceHandle:
    trace_id: str
    path: Path
    started_ns: int
    started_monotonic_ns: int
    sequence: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    truncated: bool = False
    detached: bool = False


_ACTIVE_TRACE: ContextVar[TraceHandle | None] = ContextVar(
    "tinycode_active_trace", default=None,
)
_ACTIVE_SPAN: ContextVar[str | None] = ContextVar(
    "tinycode_active_trace_span", default=None,
)


class TraceSpan:
    """A synchronous scope around an async operation."""

    def __init__(
        self,
        recorder: "TraceRecorder",
        name: str,
        kind: str,
        attributes: dict[str, Any] | None,
        *,
        activate: bool = True,
    ) -> None:
        self._recorder = recorder
        self._name = name
        self._kind = kind
        self._attributes = attributes or {}
        self._activate = activate
        self._span_id = uuid.uuid4().hex[:16]
        self._parent_span_id = _ACTIVE_SPAN.get()
        self._started_monotonic_ns = 0
        self._token: Token[str | None] | None = None
        self._finished = False

    @property
    def span_id(self) -> str:
        return self._span_id

    def __enter__(self) -> "TraceSpan":
        self._started_monotonic_ns = monotonic_ns()
        self._recorder.record(
            "span_start",
            name=self._name,
            kind=self._kind,
            span_id=self._span_id,
            parent_span_id=self._parent_span_id,
            attributes=self._attributes,
        )
        if self._activate:
            self._token = _ACTIVE_SPAN.set(self._span_id)
        return self

    def event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._recorder.record(
            name,
            span_id=self._span_id,
            parent_span_id=self._parent_span_id,
            attributes=attributes,
        )

    def finish(
        self,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        duration_ms = max(0.0, (monotonic_ns() - self._started_monotonic_ns) / 1_000_000)
        self._recorder.record(
            "span_end",
            name=self._name,
            kind=self._kind,
            span_id=self._span_id,
            parent_span_id=self._parent_span_id,
            status=status,
            duration_ms=duration_ms,
            attributes=attributes,
        )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._finished:
            if exc_type is None:
                status = "ok"
            elif exc_type.__name__ == "CancelledError":
                status = "cancelled"
            elif exc_type.__name__ in {"GeneratorExit", "StopAsyncIteration"}:
                status = "interrupted"
            else:
                status = "error"
            attributes = None
            if exc is not None:
                attributes = {
                    "error_type": exc_type.__name__,
                    "error": str(exc),
                }
            self.finish(status, attributes)
        if self._token is not None:
            try:
                _ACTIVE_SPAN.reset(self._token)
            except ValueError as reset_error:
                # Tracing must never crash task cancellation. Long-lived
                # async-generator spans should use ``activate=False``; this
                # guard also protects unexpected third-party context switches.
                self._recorder.last_error = (
                    f"Trace span context reset failed: {reset_error}"
                )
            finally:
                self._token = None
        return False


class TraceRecorder:
    """Record one JSON object per line without affecting agent execution."""

    def __init__(
        self,
        config: TracingConfig,
        project_root: Path | None = None,
    ) -> None:
        self._config = config
        self._project_root = (project_root or Path.cwd()).resolve()
        self._runtime_enabled = config.enabled
        self._last_path: Path | None = None
        self._current_handle: TraceHandle | None = None
        self._active_handles: dict[str, TraceHandle] = {}
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return self._runtime_enabled

    @property
    def storage_dir(self) -> Path:
        return self._project_root / ".tinyCode" / "traces"

    @property
    def capture_payloads(self) -> bool:
        return self._config.capture_payloads

    def set_enabled(self, enabled: bool) -> bool:
        self._runtime_enabled = bool(enabled)
        return self._runtime_enabled

    def set_project_root(self, root: Path) -> None:
        self._project_root = root.resolve()
        self._last_path = None

    def begin_task(
        self,
        task_text: str,
        *,
        session_id: str = "",
        model: str = "",
        context_window: int = 0,
        detached: bool = False,
        parent_task_id: str = "",
        role: str = "",
    ) -> TraceHandle | None:
        if not self.enabled:
            return None
        parent_handle = _ACTIVE_TRACE.get() or self._current_handle
        parent_span = _ACTIVE_SPAN.get()
        handle: TraceHandle | None = None
        try:
            storage_dir = self._validated_storage_dir()
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_dir = self._validated_storage_dir()
            self._prune()
            now = beijing_now()
            trace_id = uuid.uuid4().hex[:12]
            path = storage_dir / f"{now:%Y%m%d_%H%M%S}_{trace_id}.jsonl"
            handle = TraceHandle(
                trace_id=trace_id,
                path=path,
                started_ns=time_ns(),
                started_monotonic_ns=monotonic_ns(),
                detached=detached,
            )
            self._active_handles[handle.trace_id] = handle
            _ACTIVE_TRACE.set(handle)
            _ACTIVE_SPAN.set(None)
            if not detached:
                self._current_handle = handle
                self._last_path = path
            attributes: dict[str, Any] = {
                "session_id": session_id,
                "model": model,
                "context_window": max(0, context_window),
                "task_chars": len(task_text),
                "task_sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest()[:16],
                "workspace": str(self._project_root),
            }
            if detached and parent_handle is not None:
                attributes["parent_trace_id"] = parent_handle.trace_id
            if parent_task_id:
                attributes["subagent_task_id"] = parent_task_id
            if role:
                attributes["subagent_role"] = role
            if self.capture_payloads:
                attributes["task"] = task_text
            self.record("task_start", status="running", attributes=attributes)
            if detached and parent_handle is not None:
                self.record_for_handle(
                    parent_handle,
                    "subagent_trace_started",
                    parent_span_id=parent_span,
                    attributes={
                        "task_id": parent_task_id,
                        "role": role or "fork",
                        "child_trace_id": handle.trace_id,
                        "trace_path": str(path),
                    },
                )
            return handle
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if handle is not None:
                self._active_handles.pop(handle.trace_id, None)
            _ACTIVE_TRACE.set(parent_handle if detached else None)
            _ACTIVE_SPAN.set(parent_span if detached else None)
            if not detached:
                self._current_handle = None
            return None

    def finish_task(
        self,
        handle: TraceHandle | None,
        *,
        status: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if handle is None:
            return
        try:
            duration_ms = max(
                0.0,
                (monotonic_ns() - handle.started_monotonic_ns) / 1_000_000,
            )
            self.record_for_handle(
                handle,
                "task_end",
                status=status,
                duration_ms=duration_ms,
                attributes=attributes,
            )
        finally:
            if _ACTIVE_TRACE.get() is handle:
                _ACTIVE_TRACE.set(None)
                _ACTIVE_SPAN.set(None)
            if self._current_handle is handle:
                self._current_handle = None
            self._active_handles.pop(handle.trace_id, None)

    def record_for_handle(
        self,
        handle: TraceHandle,
        event: str,
        **kwargs: Any,
    ) -> None:
        """Record to an explicit trace without disturbing the caller context."""
        token = _ACTIVE_TRACE.set(handle)
        try:
            self.record(event, **kwargs)
        finally:
            _ACTIVE_TRACE.reset(token)

    def span(
        self,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None = None,
        *,
        activate: bool = True,
    ) -> TraceSpan:
        return TraceSpan(
            self, name, kind, attributes, activate=activate,
        )

    def record(
        self,
        event: str,
        *,
        name: str = "",
        kind: str = "event",
        span_id: str | None = None,
        parent_span_id: str | None = None,
        status: str = "",
        duration_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        # Input steering and cancellation are handled by the TUI input task,
        # which is a sibling of the foreground agent task and therefore does
        # not inherit its ContextVar.  Fall back to the one active foreground
        # handle.  A background child that inherited an older handle still
        # wins via ContextVar, so its events cannot leak into a newer task.
        handle = _ACTIVE_TRACE.get() or self._current_handle
        if not self.enabled or handle is None or handle.truncated:
            return
        try:
            with handle.lock:
                if handle.path.exists() and handle.path.stat().st_size >= MAX_TRACE_BYTES:
                    handle.truncated = True
                    return
                handle.sequence += 1
                row: dict[str, Any] = {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "trace_id": handle.trace_id,
                    "sequence": handle.sequence,
                    "timestamp": beijing_now_iso(),
                    "elapsed_ms": max(
                        0.0,
                        (monotonic_ns() - handle.started_monotonic_ns) / 1_000_000,
                    ),
                    "event": event,
                    "kind": kind,
                }
                if name:
                    row["name"] = name
                if span_id:
                    row["span_id"] = span_id
                effective_parent = parent_span_id
                if effective_parent is None and event != "span_start":
                    effective_parent = _ACTIVE_SPAN.get()
                if effective_parent:
                    row["parent_span_id"] = effective_parent
                if status:
                    row["status"] = status
                if duration_ms is not None:
                    row["duration_ms"] = round(max(0.0, duration_ms), 3)
                if attributes:
                    row["attributes"] = self._sanitize(attributes)
                encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(handle.path, flags, 0o600)
                with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                    stream.write(encoded + "\n")
                    stream.flush()
                    if event in {"task_start", "task_end"}:
                        os.fsync(stream.fileno())
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    def tool_attributes(self, name: str, parameters: object) -> dict[str, Any]:
        if not isinstance(parameters, dict):
            return {"tool": name, "parameter_type": type(parameters).__name__}
        attributes: dict[str, Any] = {
            "tool": name,
            "parameter_keys": sorted(str(key) for key in parameters)[:MAX_COLLECTION_ITEMS],
        }
        path = parameters.get("path") or parameters.get("file")
        if isinstance(path, str):
            attributes["path"] = path[:MAX_STRING_CHARS]
        if name == "apply_patch" and isinstance(parameters.get("patch"), str):
            paths = []
            for line in parameters["patch"].splitlines():
                match = _PATCH_PATH_RE.fullmatch(line)
                if match:
                    paths.append(match.group(1).strip())
            attributes["paths"] = paths[:MAX_COLLECTION_ITEMS]
        if name == "run_command" and isinstance(parameters.get("command"), str):
            command = parameters["command"]
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = []
            attributes.update({
                "command": Path(parts[0]).name if parts else "(无法解析)",
                "command_chars": len(command),
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest()[:16],
            })
        if self.capture_payloads:
            attributes["parameters"] = parameters
        return attributes

    def latest_path(self) -> Path | None:
        if (
            self._last_path is not None
            and not self._last_path.is_symlink()
            and self._last_path.is_file()
        ):
            return self._last_path
        try:
            paths = [
                path for path in self._validated_storage_dir().glob("*.jsonl")
                if (
                    not path.is_symlink()
                    and path.is_file()
                    and not self._is_detached_trace(path)
                )
            ]
            self._last_path = max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        return self._last_path

    def render_last_text(self) -> str:
        from tinyCode.tracing.render import render_text

        path = self.latest_path()
        return render_text(path) if path else "暂无执行 Trace"

    def render_last_html(self) -> Path | None:
        from tinyCode.tracing.render import linked_trace_paths, render_html

        path = self.latest_path()
        if path is None:
            return None
        html_path = path.with_suffix(".html")
        try:
            storage_dir = self._validated_storage_dir()
            for child_path in linked_trace_paths(path):
                resolved_child = child_path.resolve()
                if (
                    resolved_child.parent != storage_dir
                    or resolved_child.suffix != ".jsonl"
                    or resolved_child.is_symlink()
                    or not resolved_child.is_file()
                ):
                    continue
                child_html = resolved_child.with_suffix(".html")
                child_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    child_flags |= os.O_NOFOLLOW
                child_descriptor = os.open(child_html, child_flags, 0o600)
                with os.fdopen(
                    child_descriptor, "w", encoding="utf-8",
                ) as child_stream:
                    child_stream.write(render_html(resolved_child))
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(html_path, flags, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(render_html(path))
            return html_path
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def open_last(self) -> Path | None:
        self.last_error = ""
        path = self.render_last_html()
        if path is not None:
            try:
                if not webbrowser.open(path.resolve().as_uri()):
                    self.last_error = "系统未能打开浏览器"
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        return path

    def clear(self) -> int:
        removed = 0
        try:
            storage_dir = self._validated_storage_dir()
            for suffix in ("*.jsonl", "*.html"):
                for path in storage_dir.glob(suffix):
                    if path.is_file():
                        path.unlink()
                        removed += 1
            self._last_path = None
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return removed

    def status_text(self) -> str:
        latest = self.latest_path()
        return (
            f"Trace: {'ON' if self.enabled else 'OFF'}\n"
            f"  目录: {self.storage_dir}\n"
            f"  Payload: {'完整参数（敏感字段脱敏）' if self.capture_payloads else '仅元数据/摘要'}\n"
            f"  保留: {self._config.retention_days} 天 / 最多 {self._config.max_files} 个\n"
            f"  最近: {latest.name if latest else '无'}"
            + (f"\n  最近错误: {self.last_error}" if self.last_error else "")
        )

    def _prune(self) -> None:
        try:
            paths = sorted(
                (
                    path for path in self._validated_storage_dir().glob("*.jsonl")
                    if not path.is_symlink() and path.is_file()
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            cutoff = beijing_now() - timedelta(days=self._config.retention_days)
            keep: list[Path] = []
            for path in paths:
                modified = datetime.fromtimestamp(path.stat().st_mtime, BEIJING_TIMEZONE)
                if modified < cutoff:
                    self._remove_trace_pair(path)
                else:
                    keep.append(path)
            active_paths = {
                handle.path.resolve()
                for handle in self._active_handles.values()
            }
            for path in keep[max(0, self._config.max_files - 1):]:
                if path.resolve() not in active_paths:
                    self._remove_trace_pair(path)
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _remove_trace_pair(path: Path) -> None:
        path.unlink(missing_ok=True)
        path.with_suffix(".html").unlink(missing_ok=True)

    @staticmethod
    def _is_detached_trace(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if row.get("event") != "task_start":
                        continue
                    attrs = row.get("attributes")
                    return (
                        isinstance(attrs, dict)
                        and isinstance(attrs.get("parent_trace_id"), str)
                        and bool(attrs["parent_trace_id"])
                    )
        except OSError:
            return False
        return False

    def _sanitize(self, value: Any, depth: int = 0, key: str = "") -> Any:
        normalized_key = key.lower()
        is_token_metric = (
            normalized_key.endswith("_tokens")
            or normalized_key.startswith("estimated_tokens_")
            or normalized_key in {"first_token_ms", "token_count"}
        )
        if not is_token_metric and _SENSITIVE_KEY_RE.search(key):
            return "[REDACTED]"
        if depth >= MAX_ATTRIBUTE_DEPTH:
            return "[DEPTH_LIMIT]"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:MAX_STRING_CHARS] + ("…" if len(value) > MAX_STRING_CHARS else "")
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for index, (item_key, item_value) in enumerate(value.items()):
                if index >= MAX_COLLECTION_ITEMS:
                    result["__truncated__"] = True
                    break
                normalized_key = str(item_key)[:200]
                result[normalized_key] = self._sanitize(
                    item_value, depth + 1, normalized_key,
                )
            return result
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            result = [self._sanitize(item, depth + 1, key) for item in items[:MAX_COLLECTION_ITEMS]]
            if len(items) > MAX_COLLECTION_ITEMS:
                result.append("[ITEM_LIMIT]")
            return result
        return str(value)[:MAX_STRING_CHARS]

    def _validated_storage_dir(self) -> Path:
        candidate = self.storage_dir
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._project_root)
        except ValueError as exc:
            raise ValueError("Trace 存储目录不能指向项目外部") from exc
        return candidate

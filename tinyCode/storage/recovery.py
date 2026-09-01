"""Durable foreground-task recovery state and tool write-ahead journal.

The session journal preserves provider-visible messages. This module preserves
the runtime facts that are unsafe to infer after a power loss: which task was
active, queued steering, and tools that may have produced side effects without
returning a result.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from time import monotonic
from typing import Any

from tinyCode.storage.journal import JSONLJournal, atomic_write_text
from tinyCode.time_utils import beijing_now_iso, timestamp_sort_key


RECOVERY_DIR = Path.home() / ".tinyCode" / "recovery"
_ACTIVE_STATES = {
    "preparing", "running", "waiting_approval", "waiting_round_limit",
    "waiting_progress", "waiting_user_input", "recovery_pending",
}
_RESUMABLE_STATES = _ACTIVE_STATES | {"interrupted", "paused"}
_PATH_KEYS = {
    "path", "file", "file_path", "directory", "cwd", "workdir",
    "destination", "target", "output_path",
}


def _json_hash(value: object) -> str:
    try:
        raw = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raw = repr(value)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _target_hints(arguments: object) -> list[str]:
    """Keep reconciliation hints without persisting complete tool inputs."""
    found: list[str] = []

    def visit(value: object, key: str = "") -> None:
        if len(found) >= 24:
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key).lower())
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key in _PATH_KEYS and isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text and text not in found:
                found.append(text[:1000])

    visit(arguments)
    return found


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class TaskRecoveryStore:
    """Atomic task manifests plus an fsynced per-task tool WAL."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = (storage_dir or RECOVERY_DIR).expanduser().resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._draft_state: dict[str, tuple[int, float]] = {}

    def start_task(
        self,
        *,
        session_id: str,
        user_task: str,
        workspace: Path,
        turn_id: int,
        max_rounds: int,
        hard_max_rounds: int,
    ) -> dict[str, Any]:
        self._supersede_resumable(session_id, workspace)
        task_id = uuid.uuid4().hex[:16]
        now = beijing_now_iso()
        task = {
            "version": 1, "task_id": task_id, "session_id": session_id,
            "state": "preparing", "user_task": user_task,
            "workspace": str(workspace.resolve()), "process_id": os.getpid(),
            "turn_id": max(0, int(turn_id)), "current_round": 0,
            "current_limit": max(1, int(max_rounds)),
            "hard_limit": max(1, int(hard_max_rounds)), "waiting": {},
            "pending_steering": [], "inflight_tools": {},
            "last_safe_checkpoint": "task_created", "checkpoint_seq": 1,
            "started_at": now, "updated_at": now, "error": "",
        }
        self._write(task)
        return task

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self._read(self._manifest_path(task_id))

    def update(self, task_id: str, **changes: Any) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.update(changes)
        task["updated_at"] = beijing_now_iso()
        task["process_id"] = os.getpid()
        task["checkpoint_seq"] = int(task.get("checkpoint_seq", 0)) + 1
        self._write(task)
        return task

    def checkpoint(self, task_id: str, label: str, **changes: Any) -> None:
        changes["last_safe_checkpoint"] = label
        self.update(task_id, **changes)

    def finish(self, task_id: str, state: str, *, error: str = "") -> None:
        if state in _ACTIVE_STATES:
            raise ValueError(f"终态不能是 {state}")
        unresolved = self.unresolved_tools(task_id)
        # A graceful Python exception does not prove that an already-started
        # external operation was rolled back. Keep such tasks recoverable.
        if unresolved and state != "abandoned":
            state = "interrupted"
        self.update(
            task_id, state=state, waiting={},
            inflight_tools=unresolved, error=str(error)[:4000],
            finished_at=beijing_now_iso(), last_safe_checkpoint=f"task_{state}",
        )
        if state == "completed":
            self.checkpoint_draft(task_id, "", force=True)

    def _supersede_resumable(self, session_id: str, workspace: Path) -> None:
        """A later user turn in the same session intentionally moves forward."""
        target = str(workspace.resolve())
        for path in self.storage_dir.glob("*.task.json"):
            task = self._read(path)
            if not task:
                continue
            if task.get("session_id") != session_id or task.get("workspace") != target:
                continue
            if task.get("state") not in {"interrupted", "paused"}:
                continue
            task["state"] = "superseded"
            task["finished_at"] = beijing_now_iso()
            task["updated_at"] = beijing_now_iso()
            task["last_safe_checkpoint"] = "superseded_by_later_turn"
            self._write(task)

    def record_tool_intent(
        self,
        task_id: str,
        *,
        call_id: str,
        tool_name: str,
        arguments: object,
        round_number: int,
        may_modify: bool,
    ) -> dict[str, Any]:
        row = {
            "event": "intent", "timestamp": beijing_now_iso(), "task_id": task_id,
            "call_id": call_id, "tool": tool_name,
            "round": max(0, int(round_number)), "may_modify": bool(may_modify),
            "arguments_sha256": _json_hash(arguments),
            "targets": _target_hints(arguments),
        }
        # WAL first: after this fsync startup can discover the operation even
        # when the following manifest update never completes.
        JSONLJournal(self._wal_path(task_id)).append([row])
        self.update(
            task_id, state="running", inflight_tools=self.unresolved_tools(task_id),
            last_safe_checkpoint="before_tool_execution",
        )
        return row

    def record_tool_result(
        self,
        task_id: str,
        *,
        call_id: str,
        tool_name: str,
        success: bool,
        error: str = "",
        content: str = "",
    ) -> None:
        JSONLJournal(self._wal_path(task_id)).append([{
            "event": "result", "timestamp": beijing_now_iso(), "task_id": task_id,
            "call_id": call_id, "tool": tool_name, "success": bool(success),
            "error": str(error)[:4000], "content_sha256": _json_hash(content),
            "content_chars": len(content),
        }])
        self.update(
            task_id, inflight_tools=self.unresolved_tools(task_id),
            last_safe_checkpoint="after_tool_result",
        )

    def unresolved_tools(self, task_id: str) -> dict[str, dict[str, Any]]:
        pending: dict[str, dict[str, Any]] = {}
        for row in JSONLJournal(self._wal_path(task_id)).read_rows():
            call_id = str(row.get("call_id", ""))
            if not call_id:
                continue
            if row.get("event") == "intent":
                pending[call_id] = row
            elif row.get("event") == "result":
                pending.pop(call_id, None)
        return pending

    def checkpoint_draft(self, task_id: str, text: str, *, force: bool = False) -> None:
        """Persist a UI-only draft at a bounded rate, never as protocol input."""
        previous_length, previous_time = self._draft_state.get(task_id, (0, 0.0))
        now = monotonic()
        if not force and len(text) - previous_length < 2048 and now - previous_time < 1.0:
            return
        path = self._draft_path(task_id)
        if text:
            atomic_write_text(path, text)
        elif path.exists():
            path.unlink(missing_ok=True)
        self._draft_state[task_id] = (len(text), now)

    def find_latest_interrupted(self, workspace: Path) -> dict[str, Any] | None:
        target = str(workspace.resolve())
        candidates: list[dict[str, Any]] = []
        for path in self.storage_dir.glob("*.task.json"):
            task = self._read(path)
            if not task or task.get("workspace") != target:
                continue
            if task.get("state") not in _RESUMABLE_STATES:
                continue
            if _pid_is_alive(task.get("process_id")):
                continue
            task["inflight_tools"] = self.unresolved_tools(str(task["task_id"]))
            task["has_draft"] = self._draft_path(str(task["task_id"])).exists()
            candidates.append(task)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: timestamp_sort_key(item.get("updated_at", "")),
        )

    def prepare_recovery(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        interrupted_state = str(task.get("state", ""))
        interrupted_waiting = task.get("waiting", {})
        interrupted_checkpoint = str(task.get("last_safe_checkpoint", ""))
        prepared = self.update(
            task_id, state="recovery_pending",
            inflight_tools=self.unresolved_tools(task_id), waiting={},
            interrupted_state=interrupted_state,
            interrupted_waiting=interrupted_waiting,
            interrupted_checkpoint=interrupted_checkpoint,
            recovered_at=beijing_now_iso(), last_safe_checkpoint="recovery_selected",
        )
        if prepared is not None:
            prepared["has_draft"] = self._draft_path(task_id).exists()
        return prepared

    def describe(self, task: dict[str, Any]) -> str:
        pending = task.get("inflight_tools") or self.unresolved_tools(
            str(task.get("task_id", ""))
        )
        lines = [
            f"任务: {task.get('user_task', '')}",
            f"会话: {task.get('session_id', '')}",
            f"工作区: {task.get('workspace', '')}",
            f"状态: {task.get('state', '')}",
            f"最后安全点: {task.get('last_safe_checkpoint', '')}",
            f"轮次: {task.get('current_round', 0)} / {task.get('current_limit', 0)}",
        ]
        if pending:
            lines.append("状态未知的工具调用:")
            for call_id, row in pending.items():
                targets = ", ".join(row.get("targets", [])) or "未记录目标"
                lines.append(f"- {row.get('tool', 'unknown')} ({call_id}) · {targets}")
        else:
            lines.append("状态未知的工具调用: 无")
        steering = task.get("pending_steering", [])
        if steering:
            lines.append(f"待注入追加指令: {len(steering)} 条")
        waiting = task.get("waiting") or task.get("interrupted_waiting")
        if waiting:
            lines.append(f"中断时等待状态: {waiting.get('kind', 'unknown')}")
        if task.get("has_draft"):
            lines.append("未完成模型草稿: 已保留（仅供审计，不注入模型上下文）")
        return "\n".join(lines)

    def build_recovery_prompt(self, task: dict[str, Any]) -> str:
        pending = task.get("inflight_tools") or {}
        unknown = "\n".join(
            f"- call_id={call_id} tool={row.get('tool', 'unknown')} "
            f"targets={row.get('targets', [])}"
            for call_id, row in pending.items()
        ) or "- 无"
        steering_count = len(task.get("pending_steering", []))
        waiting = task.get("interrupted_waiting") or {}
        waiting_kind = str(waiting.get("kind", "无"))
        return (
            "[中断任务恢复]\n"
            f"原始任务：{task.get('user_task', '')}\n"
            f"工作区：{task.get('workspace', '')}\n"
            f"上次轮次：{task.get('current_round', 0)}；"
            "最后安全检查点："
            f"{task.get('interrupted_checkpoint') or task.get('last_safe_checkpoint', '')}\n"
            f"中断时等待状态：{waiting_kind}。任何旧审批都已失效，"
            "需要时必须重新发起确认。\n"
            "以下工具调用只有执行意图、没有可信的完成记录，最终状态未知：\n"
            f"{unknown}\n"
            f"运行时还原了 {steering_count} 条尚未注入的用户追加指令，"
            "它们会在本次请求前按原顺序进入上下文。\n\n"
            "请从协议安全边界恢复任务。第一步必须检查当前工作区、相关文件和"
            "外部副作用，判断上次操作实际完成到哪里；对状态未知的写操作不得"
            "盲目重试。确认现状后继续完成原始任务，并在最终回答中说明恢复检查结果。"
        )

    def _write(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("task_id", ""))
        if not task_id or not task_id.replace("-", "").isalnum():
            raise ValueError("无效的恢复任务 ID")
        atomic_write_text(
            self._manifest_path(task_id),
            json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _manifest_path(self, task_id: str) -> Path:
        return self.storage_dir / f"{task_id}.task.json"

    def _wal_path(self, task_id: str) -> Path:
        return self.storage_dir / f"{task_id}.tools.jsonl"

    def _draft_path(self, task_id: str) -> Path:
        return self.storage_dir / f"{task_id}.draft.txt"

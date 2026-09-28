"""Project-local persistence for session-scoped Goals."""

from __future__ import annotations

import json
from pathlib import Path

from tinyCode.goals.models import Goal
from tinyCode.storage.journal import atomic_write_text


class GoalStore:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._dir = self._root / ".tinyCode" / "goals"
        self.last_error = ""

    def save(self, goal: Goal) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self._path(goal.session_id),
            json.dumps(goal.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    def load(self, session_id: str) -> Goal | None:
        self.last_error = ""
        try:
            raw = json.loads(self._path(session_id).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Goal 顶层必须是对象")
            goal = Goal.from_dict(raw)
            if goal.session_id != session_id:
                raise ValueError("Goal 会话标识不匹配")
            return goal
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def clear(self, session_id: str) -> None:
        try:
            self._path(session_id).unlink(missing_ok=True)
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _path(self, session_id: str) -> Path:
        safe = "".join(char for char in session_id if char.isalnum() or char in "-_")
        if not safe or safe != session_id:
            raise ValueError("会话 ID 无效")
        return self._dir / f"{safe}.json"

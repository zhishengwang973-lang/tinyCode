"""Team persistence — save/load team state from disk."""

import json
import re
from pathlib import Path

from tinyCode.teams.models import MemberDef, TeamDef

PROJECT_TEAMS_DIR = Path.cwd() / ".tinyCode" / "teams"
USER_TEAMS_DIR = Path.home() / ".tinyCode" / "teams"
_TEAM_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_safe_name(name: str) -> bool:
    return isinstance(name, str) and bool(_TEAM_NAME_RE.fullmatch(name))


def load_team_def(name: str) -> TeamDef | None:
    """Load a team definition from user-level directory."""
    if not _is_safe_name(name):
        return None
    path = USER_TEAMS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None

    if not isinstance(data, dict):
        return None
    raw_members = data.get("members", [])
    if not isinstance(raw_members, list):
        return None

    members = []
    seen_names: set[str] = set()
    for m in raw_members:
        member_name = m.get("name", "") if isinstance(m, dict) else ""
        if not _is_safe_name(member_name) or member_name in seen_names:
            continue
        seen_names.add(member_name)
        role = m.get("role", "")
        worktree = m.get("worktree", "")
        backend = m.get("backend", "coro")
        model = m.get("model", "")
        if not all(isinstance(value, str) for value in (role, worktree, backend, model)):
            continue
        needs_approval = m.get("needs_approval", False)
        if not isinstance(needs_approval, bool):
            continue
        members.append(MemberDef(
            name=member_name, role=role, worktree=worktree, backend=backend,
            needs_approval=needs_approval, model=model,
        ))

    def _text(key: str, default: str = "") -> str:
        value = data.get(key, default)
        return value if isinstance(value, str) else default

    dispatch_mode = data.get("dispatch_mode", False)
    if not isinstance(dispatch_mode, bool):
        return None
    max_rounds = data.get("max_rounds_per_member", 10)
    if (
        not isinstance(max_rounds, int) or isinstance(max_rounds, bool)
        or not 1 <= max_rounds <= 100
    ):
        return None
    return TeamDef(
        name=_text("name", name), description=_text("description"),
        lead_role=_text("lead_role"), members=members,
        dispatch_mode=dispatch_mode, max_rounds_per_member=max_rounds,
    )


def list_team_defs() -> list[str]:
    """List available team definition names."""
    if not USER_TEAMS_DIR.exists():
        return []
    try:
        return [p.stem for p in USER_TEAMS_DIR.glob("*.json") if _is_safe_name(p.stem)]
    except OSError:
        return []


def get_team_dir(name: str) -> Path:
    """Get the project-level working directory for a team."""
    if not _is_safe_name(name):
        raise ValueError(f"Invalid team name: {name}")
    d = PROJECT_TEAMS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d

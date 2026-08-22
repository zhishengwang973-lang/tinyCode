"""Role loader — scan three tiers for YAML+Markdown role definitions."""

import re
import sys
from pathlib import Path

import yaml

from tinyCode.subagent.models import SubAgentRole

PROJECT_DIR = Path.cwd() / ".tinyCode" / "roles"
USER_DIR = Path.home() / ".tinyCode" / "roles"
BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RoleLoader:
    """Loads sub-agent role definitions from three-tier directories."""

    def load_all(self) -> dict[str, SubAgentRole]:
        index: dict[str, SubAgentRole] = {}
        self._scan_dir(BUILTIN_DIR, index)
        self._scan_dir(USER_DIR, index)
        self._scan_dir(PROJECT_DIR, index)
        return index

    def _scan_dir(self, directory: Path, index: dict[str, SubAgentRole]) -> None:
        if not directory.exists():
            return
        for md_file in sorted(directory.glob("*.md")):
            role = self._parse(md_file)
            if role:
                index[role.name] = role

    def _parse(self, path: Path) -> SubAgentRole | None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"Role [{path}]: 读取失败 — {exc}", file=sys.stderr)
            return None

        match = _FRONTMATTER_RE.match(text)
        if not match:
            print(f"Role [{path}]: 缺少 YAML frontmatter", file=sys.stderr)
            return None

        try:
            fm = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            print(f"Role [{path}]: YAML 解析失败 — {exc}", file=sys.stderr)
            return None

        if not isinstance(fm, dict):
            print(f"Role [{path}]: frontmatter 不是字典", file=sys.stderr)
            return None

        name = fm.get("name", path.stem)
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            print(f"Role [{path}]: name 必须只包含字母、数字、_、-", file=sys.stderr)
            return None
        tools_allow = fm.get("tools_allow")
        if tools_allow is not None:
            if not isinstance(tools_allow, list) or not all(isinstance(t, str) for t in tools_allow):
                print(f"Role [{name}]: tools_allow 必须是字符串列表", file=sys.stderr)
                return None

        tools_deny = fm.get("tools_deny", [])
        if not isinstance(tools_deny, list) or not all(isinstance(t, str) for t in tools_deny):
            print(f"Role [{name}]: tools_deny 必须是字符串列表", file=sys.stderr)
            return None

        body = text[match.end():].strip()

        description = fm.get("description", "")
        if not isinstance(description, str):
            description = ""
        model = fm.get("model")
        if model is not None and not isinstance(model, str):
            model = None
        max_rounds = fm.get("max_rounds", 5)
        if (
            isinstance(max_rounds, bool)
            or not isinstance(max_rounds, int)
            or not 1 <= max_rounds <= 100
        ):
            max_rounds = 5
        permission = fm.get("permission", "normal")
        if permission not in {"strict", "normal", "permissive"}:
            permission = "normal"

        return SubAgentRole(
            name=name,
            description=description,
            tools_allow=tools_allow,
            tools_deny=tools_deny,
            model=model,
            max_rounds=max_rounds,
            permission=permission,
            system_prompt=body,
            source=str(path),
        )

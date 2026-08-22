"""Skill loader — scan three tiers, parse YAML frontmatter + Markdown body."""

import re
import sys
from pathlib import Path

import yaml

from tinyCode.skills.models import HistoryCarry, SkillDefinition, SkillMeta, SkillMode

#: Project-level skills directory
PROJECT_DIR = Path.cwd() / ".tinyCode" / "skills"
#: User-level skills directory
USER_DIR = Path.home() / ".tinyCode" / "skills"
#: Built-in skills directory (relative to this package)
BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SkillLoader:
    """Scans three-tier skill directories, resolves overrides by name,
    and parses each valid skill file."""

    def load_all(self) -> list[SkillDefinition]:
        """Phase 1: load all skills (names + descriptions only, bodies deferred)."""
        index: dict[str, SkillDefinition] = {}

        # Built-in first (lowest priority)
        self._scan_dir(BUILTIN_DIR, index)
        # User next
        self._scan_dir(USER_DIR, index)
        # Project last (highest priority — overrides same name)
        self._scan_dir(PROJECT_DIR, index)

        return list(index.values())

    def load_one(self, source_path: str) -> SkillDefinition | None:
        """Hot-reload a single skill file. Returns None on parse failure."""
        return self._parse_file(Path(source_path))

    # -- internals -----------------------------------------------------------

    def _scan_dir(self, directory: Path, index: dict[str, SkillDefinition]) -> None:
        if not directory.exists():
            return

        # Single-file skills: *.md
        for md_file in sorted(directory.glob("*.md")):
            skill = self._parse_file(md_file)
            if skill:
                index[skill.meta.name] = skill  # override by name

        # Directory skills: subdirectories with skill.md
        for subdir in sorted(directory.iterdir()):
            if not subdir.is_dir():
                continue
            skill_md = subdir / "skill.md"
            if not skill_md.exists():
                continue
            skill = self._parse_file(skill_md)
            if skill:
                skill.directory = str(subdir)
                index[skill.meta.name] = skill

    def _parse_file(self, path: Path) -> SkillDefinition | None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"Skill [{path}]: 读取失败 — {exc}", file=sys.stderr)
            return None

        match = _FRONTMATTER_RE.match(text)
        if not match:
            print(f"Skill [{path}]: 缺少 YAML frontmatter (--- ... ---)", file=sys.stderr)
            return None

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            print(f"Skill [{path}]: YAML 解析失败 — {exc}", file=sys.stderr)
            return None

        if not isinstance(frontmatter, dict):
            print(f"Skill [{path}]: frontmatter 不是字典", file=sys.stderr)
            return None

        name = frontmatter.get("name", path.stem)
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            print(f"Skill [{path}]: name 必须只包含字母、数字、_、-", file=sys.stderr)
            return None

        # Parse mode
        mode_value = frontmatter.get("mode", "shared")
        mode_str = mode_value.lower() if isinstance(mode_value, str) else "shared"
        try:
            mode = SkillMode(mode_str)
        except ValueError:
            print(f"Skill [{name}]: 无效 mode '{mode_str}'，使用 shared", file=sys.stderr)
            mode = SkillMode.SHARED

        # Parse history_carry
        hc_value = frontmatter.get("history_carry", "full")
        hc_str = hc_value.lower() if isinstance(hc_value, str) else "full"
        try:
            history_carry = HistoryCarry(hc_str)
        except ValueError:
            history_carry = HistoryCarry.FULL

        tools = frontmatter.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                print(f"Skill [{name}]: tools 必须是字符串列表", file=sys.stderr)
                return None

        description = frontmatter.get("description", "")
        if not isinstance(description, str):
            description = ""
        model = frontmatter.get("model")
        if model is not None and not isinstance(model, str):
            model = None
        recent_count = frontmatter.get("recent_count", 10)
        if (
            isinstance(recent_count, bool)
            or not isinstance(recent_count, int)
            or not 1 <= recent_count <= 100
        ):
            recent_count = 10

        meta = SkillMeta(
            name=name,
            description=description,
            mode=mode,
            model=model,
            tools=tools,
            history_carry=history_carry,
            recent_count=recent_count,
            source=str(path),
        )

        body = text[match.end():].strip()
        return SkillDefinition(meta=meta, body=body)

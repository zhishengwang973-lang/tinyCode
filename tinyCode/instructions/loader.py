"""Instructions loader — reads TINYCODE.md files with @include support."""

import re
from pathlib import Path

MAX_INCLUDE_DEPTH = 3
MAX_INSTRUCTION_CHARS = 1_000_000
_INCLUDE_RE = re.compile(r'^@include\((.+)\)$', re.MULTILINE)


class InstructionsLoader:
    """Loads project + user instruction files, resolving @include directives.

    Priority: project-level (``TINYCODE.md`` in cwd) first, then user-level
    (``~/.tinyCode/instructions.md``) — higher priority = first in output.
    """

    def load(self, cwd: Path | None = None) -> str:
        cwd = (cwd or Path.cwd()).resolve()
        parts: list[str] = []

        project_file = cwd / "TINYCODE.md"
        if project_file.exists():
            parts.append(self._load_entry(project_file, cwd))

        user_file = Path.home() / ".tinyCode" / "instructions.md"
        if user_file.exists():
            parts.append(
                self._load_entry(user_file, Path.home().resolve())
            )

        return "\n\n".join(parts)

    def load_project(self, cwd: Path | None = None) -> str:
        cwd = (cwd or Path.cwd()).resolve()
        project_file = cwd / "TINYCODE.md"
        if project_file.exists():
            return self._load_entry(project_file, cwd)
        return ""

    def load_user(self) -> str:
        user_file = Path.home() / ".tinyCode" / "instructions.md"
        if user_file.exists():
            return self._load_entry(user_file, Path.home().resolve())
        return ""

    # -- internals -----------------------------------------------------------

    def _load_entry(self, file_path: Path, allowed_root: Path) -> str:
        return self._load_with_includes(
            file_path,
            depth=0,
            allowed_root=allowed_root,
            remaining=[MAX_INSTRUCTION_CHARS],
            active=set(),
        )

    def _load_with_includes(
        self,
        file_path: Path,
        depth: int,
        allowed_root: Path,
        remaining: list[int] | None = None,
        active: set[Path] | None = None,
    ) -> str:
        remaining = remaining if remaining is not None else [MAX_INSTRUCTION_CHARS]
        active = active if active is not None else set()
        if depth > MAX_INCLUDE_DEPTH:
            raise ValueError(
                f"@include 嵌套深度超过 {MAX_INCLUDE_DEPTH} 层: {file_path}"
            )
        resolved_file = file_path.resolve()
        if resolved_file in active:
            raise ValueError(f"@include 检测到循环引用: {file_path}")
        active.add(resolved_file)
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                content = handle.read(remaining[0] + 1)
            if len(content) > remaining[0]:
                raise ValueError(
                    f"指令内容总量超过 {MAX_INSTRUCTION_CHARS} 字符限制"
                )
            remaining[0] -= len(content)
            base_dir = file_path.parent.resolve()

            def _resolve_include(match: re.Match) -> str:
                include_path = match.group(1).strip()
                full_path = (base_dir / include_path).resolve()
                # Block escaping the root selected by the entry-point loader.
                try:
                    full_path.relative_to(allowed_root)
                except ValueError:
                    raise ValueError(f"@include 路径越界: {include_path}")
                if not full_path.exists():
                    raise ValueError(f"@include 文件不存在: {include_path}")
                return self._load_with_includes(
                    full_path,
                    depth + 1,
                    allowed_root,
                    remaining,
                    active,
                )

            return _INCLUDE_RE.sub(_resolve_include, content)
        finally:
            active.discard(resolved_file)

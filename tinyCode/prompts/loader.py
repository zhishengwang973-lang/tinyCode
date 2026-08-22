"""Prompt module loader — reads .txt files from the modules directory."""

import re
from dataclasses import dataclass
from pathlib import Path


_INJECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class PromptModule:
    priority: int       # lower = higher priority (1 comes first)
    name: str           # derived from filename, e.g. "01-identity" → "identity"
    content: str        # raw text content


def load_modules(modules_dir: Path | None = None) -> list[PromptModule]:
    """Load all .txt prompt modules from the given directory, sorted by priority.

    Files must follow naming convention ``NN-name.txt`` where NN is a
    two-digit integer priority number.
    """
    if modules_dir is None:
        modules_dir = Path(__file__).resolve().parent / "modules"

    modules: list[PromptModule] = []

    for file_path in sorted(modules_dir.glob("*.txt")):
        match = re.match(r"^(\d+)-(.+)\.txt$", file_path.name)
        if not match:
            continue
        priority = int(match.group(1))
        name = match.group(2)
        content = file_path.read_text(encoding="utf-8").strip()
        modules.append(PromptModule(priority=priority, name=name, content=content))

    modules.sort(key=lambda m: m.priority)
    return modules


def load_injection(name: str, injections_dir: Path | None = None) -> str:
    """Load an injection template by name (without .txt extension)."""
    if not _INJECTION_NAME_RE.fullmatch(name):
        return ""
    if injections_dir is None:
        injections_dir = Path(__file__).resolve().parent / "injections"
    file_path = injections_dir / f"{name}.txt"
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8").strip()

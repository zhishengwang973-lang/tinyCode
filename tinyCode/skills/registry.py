"""Skill registry — two-phase loading, activation tracking."""

from tinyCode.skills.loader import SkillLoader
from tinyCode.skills.models import SkillDefinition, SkillMeta


class SkillRegistry:
    """Manages skill lifecycle: load (phase 1), activate (phase 2), deactivate."""

    def __init__(self, loader: SkillLoader) -> None:
        self._loader = loader
        self._all: dict[str, SkillDefinition] = {}       # phase 1
        self._activated: dict[str, SkillDefinition] = {}  # phase 2
        self._loaded = False

    # -- phase 1: load names + descriptions ----------------------------------

    def load_all(self) -> list[SkillMeta]:
        """Scan all tiers and return metadata summaries (names + descriptions).

        This is called at startup — only metadata is injected into the
        conversation so the Agent knows what skills exist.
        """
        self._all.clear()
        for skill in self._loader.load_all():
            self._all[skill.meta.name] = skill
        self._loaded = True
        return [s.meta for s in self._all.values()]

    def list_available(self) -> list[SkillMeta]:
        return [s.meta for s in self._all.values()]

    # -- phase 2: activate (full load) ---------------------------------------

    def activate(self, name: str) -> SkillDefinition | None:
        """Load full instructions for a skill and pin it."""
        skill = self._all.get(name)
        if skill is None:
            return None
        # Hot-reload from source if file-based
        if skill.meta.source:
            reloaded = self._loader.load_one(skill.meta.source)
            if reloaded:
                skill = reloaded
        self._activated[name] = skill
        return skill

    def deactivate(self, name: str) -> bool:
        if name in self._activated:
            del self._activated[name]
            return True
        return False

    def clear_activated(self) -> None:
        """Clear all activated skills (e.g., on /clear)."""
        self._activated.clear()

    # -- access ---------------------------------------------------------------

    @property
    def activated(self) -> list[SkillDefinition]:
        return list(self._activated.values())

    def get_active_instructions(self) -> str:
        """Return combined instructions from all activated skills, pinned."""
        if not self._activated:
            return ""
        parts: list[str] = []
        for skill in self._activated.values():
            parts.append(f"## Skill: {skill.meta.name}\n{skill.body}")
        return "\n\n".join(parts)

    def get_active_tool_whitelist(self) -> list[str] | None:
        """Return the intersection of all activated skills' tool whitelists.

        A skill with no whitelist represents the universal set, so it does not
        weaken a finite whitelist supplied by another active skill. Returns
        None only when no active skill imposes a tool restriction.
        """
        whitelists: list[set[str]] = []
        for skill in self._activated.values():
            if skill.meta.tools is None:
                continue
            whitelists.append(set(skill.meta.tools))
        if not whitelists:
            return None
        return list(set.intersection(*whitelists))

    def get_skill(self, name: str) -> SkillDefinition | None:
        return self._all.get(name)

    def get_meta(self, name: str) -> SkillMeta | None:
        skill = self._all.get(name)
        return skill.meta if skill else None

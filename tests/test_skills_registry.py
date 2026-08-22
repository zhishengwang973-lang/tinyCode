import unittest

from tinyCode.skills.models import SkillDefinition, SkillMeta
from tinyCode.skills.registry import SkillRegistry


class FakeSkillLoader:
    def __init__(self, skills):
        self.skills = skills

    def load_all(self):
        return list(self.skills)

    def load_one(self, source_path):
        return None


class SkillRegistryTests(unittest.TestCase):
    def test_unrestricted_skill_does_not_weaken_another_skill_whitelist(self):
        loader = FakeSkillLoader([
            SkillDefinition(SkillMeta(name="restricted", tools=["read_file", "grep"]), ""),
            SkillDefinition(SkillMeta(name="unrestricted", tools=None), ""),
        ])
        registry = SkillRegistry(loader)
        registry.load_all()
        registry.activate("restricted")
        registry.activate("unrestricted")

        self.assertEqual(
            {"read_file", "grep"},
            set(registry.get_active_tool_whitelist() or []),
        )

    def test_multiple_finite_whitelists_are_intersected(self):
        loader = FakeSkillLoader([
            SkillDefinition(SkillMeta(name="one", tools=["read_file", "grep"]), ""),
            SkillDefinition(SkillMeta(name="two", tools=["grep", "glob"]), ""),
        ])
        registry = SkillRegistry(loader)
        registry.load_all()
        registry.activate("one")
        registry.activate("two")

        self.assertEqual(["grep"], registry.get_active_tool_whitelist())


if __name__ == "__main__":
    unittest.main()

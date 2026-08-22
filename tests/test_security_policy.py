import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.security.models import RuleAction, SecurityLevel, SecurityRule
from tinyCode.security.policy import SecurityPolicy


class SecurityPolicyTests(unittest.TestCase):
    def test_normal_mode_allows_read_only_shell_commands(self):
        policy = self.make_policy(SecurityLevel.NORMAL)

        self.assertEqual(
            RuleAction.ASK,
            policy.evaluate("run_command", command="ls -la"),
        )
        self.assertEqual(
            RuleAction.ASK,
            policy.evaluate(
                "run_command",
                command='ls -la && find . -name "*.java" | head -20',
            ),
        )
        self.assertEqual(
            RuleAction.ASK,
            policy.evaluate("run_command", command="git status --short"),
        )
        self.assertEqual(
            RuleAction.ASK,
            policy.evaluate(
                "run_command",
                command="ls src 2>/dev/null; find tests -type f 2>/dev/null",
            ),
        )

    def test_normal_mode_still_asks_for_mutating_or_ambiguous_commands(self):
        policy = self.make_policy(SecurityLevel.NORMAL)
        commands = [
            "touch result.txt",
            "ls -la > files.txt",
            "ls -la 2> errors.txt",
            "echo $(touch result.txt)",
            "find . -delete",
            "find . -exec rm {} ;",
            "sed -i s/old/new/ app.py",
            "git checkout main",
            "cat file | tee copy.txt",
        ]

        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(
                    RuleAction.ASK,
                    policy.evaluate("run_command", command=command),
                )

    def make_policy(self, level=SecurityLevel.NORMAL):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        project_root = Path(tmp.name) / "project"
        project_root.mkdir()

        with patch.object(SecurityPolicy, "_load_global_rules", lambda self: None):
            return SecurityPolicy(level=level, project_root=project_root)

    def test_normal_mode_allows_read_tools_and_asks_for_write_tools(self):
        policy = self.make_policy(SecurityLevel.NORMAL)

        self.assertEqual(RuleAction.ALLOW, policy.evaluate("read_file", path="app.py"))
        self.assertEqual(RuleAction.ASK, policy.evaluate("edit_file", path="app.py"))
        self.assertEqual(RuleAction.ASK, policy.evaluate("apply_patch", path="app.py"))
        self.assertEqual(RuleAction.ALLOW, policy.evaluate("web_search"))
        self.assertEqual(RuleAction.ALLOW, policy.evaluate("web_fetch"))
        self.assertEqual(RuleAction.ALLOW, policy.evaluate("request_user_input"))

    def test_strict_mode_asks_before_transmitting_web_requests(self):
        policy = self.make_policy(SecurityLevel.STRICT)

        self.assertEqual(RuleAction.ASK, policy.evaluate("web_search"))
        self.assertEqual(RuleAction.ASK, policy.evaluate("web_fetch"))
        self.assertEqual(RuleAction.ALLOW, policy.evaluate("request_user_input"))

    def test_strict_mode_allows_default_whitelisted_paths_only(self):
        policy = self.make_policy(SecurityLevel.STRICT)

        self.assertEqual(RuleAction.ALLOW, policy.evaluate("read_file", path="src/app.py"))
        self.assertEqual(RuleAction.ASK, policy.evaluate("read_file", path="secrets.env"))

    def test_strict_mode_never_uses_read_path_allowlist_for_writes(self):
        policy = self.make_policy(SecurityLevel.STRICT)

        self.assertEqual(RuleAction.ASK, policy.evaluate("edit_file", path="src/app.py"))
        self.assertEqual(RuleAction.ASK, policy.evaluate("delete_file", path="app.py"))

    def test_session_rule_overrides_mode_default(self):
        policy = self.make_policy(SecurityLevel.NORMAL)
        policy.add_session_rule(
            SecurityRule(tool="edit_file", action=RuleAction.DENY, path_pattern="src/**")
        )

        self.assertEqual(RuleAction.DENY, policy.evaluate("edit_file", path="src/app.py"))
        self.assertEqual(RuleAction.ASK, policy.evaluate("edit_file", path="docs/readme.md"))

    def test_malformed_persisted_rules_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            (project_root / ".tinyCode-security.yaml").write_text(
                "\n".join([
                    "rules:",
                    "  - deny-everything",
                    "  - tool: edit_file",
                    "    action: deny",
                    "    path_pattern: src/**",
                    "",
                ]),
                encoding="utf-8",
            )

            with patch.object(SecurityPolicy, "_load_global_rules", lambda self: None):
                policy = SecurityPolicy(level=SecurityLevel.NORMAL, project_root=project_root)

            self.assertEqual(RuleAction.DENY, policy.evaluate("edit_file", path="src/app.py"))
            self.assertEqual(RuleAction.ASK, policy.evaluate("edit_file", path="docs/readme.md"))
            self.assertTrue(any("规则 #0" in error for error in policy.load_errors))

    def test_non_string_persisted_rule_fields_never_crash_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            (project_root / ".tinyCode-security.yaml").write_text(
                "rules:\n"
                "  - tool: 123\n"
                "    action: deny\n"
                "  - tool: edit_file\n"
                "    action: deny\n"
                "    path_pattern: 456\n",
                encoding="utf-8",
            )

            with patch.object(SecurityPolicy, "_load_global_rules", lambda self: None):
                policy = SecurityPolicy(
                    level=SecurityLevel.NORMAL,
                    project_root=project_root,
                )

            self.assertEqual(RuleAction.ASK, policy.evaluate("edit_file", path="app.py"))
            self.assertEqual(2, len(policy.load_errors))

    def test_failed_permanent_rule_write_rolls_back_in_memory_rule(self):
        policy = self.make_policy(SecurityLevel.NORMAL)
        rule = SecurityRule(
            tool="edit_file",
            action=RuleAction.ALLOW,
            path_pattern="src/**",
        )

        with patch(
            "tinyCode.security.policy.atomic_write_text",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                policy.add_permanent_rule(rule)

        self.assertEqual(
            RuleAction.ASK,
            policy.evaluate("edit_file", path="src/app.py"),
        )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from tinyCode.security.sandbox import PathSandbox
from tinyCode.security.guard import SecurityGuard
from tinyCode.security.models import HITLDecision, RuleAction, SecurityLevel
from tinyCode.security.policy import SecurityPolicy


class PathSandboxTests(unittest.TestCase):
    def test_rejects_absolute_path_even_inside_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            inside_file = project_root / "app.py"

            safe, message = PathSandbox(project_root).validate(str(inside_file))

            self.assertFalse(safe)
            self.assertIn("绝对路径", message)

    def test_security_guard_checks_glob_pattern_for_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(level=SecurityLevel.NORMAL, project_root=project_root),
                sandbox=PathSandbox(project_root),
            )

            allowed, reason = guard.check("glob", {"pattern": "../*.py"})

            self.assertFalse(allowed)
            self.assertIn("路径", reason)

    def test_security_guard_checks_delete_file_path_for_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(level=SecurityLevel.NORMAL, project_root=project_root),
                sandbox=PathSandbox(project_root),
            )

            allowed, reason = guard.check("delete_file", {"path": "../secret.txt"})

            self.assertFalse(allowed)
            self.assertIn("路径", reason)

    def test_security_guard_uses_glob_pattern_for_hitl_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(level=SecurityLevel.STRICT, project_root=project_root),
                sandbox=PathSandbox(project_root),
            )

            self.assertFalse(guard.needs_hitl("glob", {"pattern": "src/*.py"}))

    def test_security_guard_persists_glob_hitl_rule_for_pattern_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            policy = SecurityPolicy(level=SecurityLevel.STRICT, project_root=project_root)
            guard = SecurityGuard(policy=policy, sandbox=PathSandbox(project_root))

            guard.apply_hitl(
                HITLDecision.ALLOW_SESSION,
                "glob",
                {"pattern": "safe/*.log"},
            )

            self.assertEqual(RuleAction.ALLOW, policy.evaluate("glob", path="safe/app.log"))
            self.assertEqual(RuleAction.ASK, policy.evaluate("glob", path="secrets.env"))

    def test_security_guard_rejects_non_string_model_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(level=SecurityLevel.NORMAL, project_root=project_root),
                sandbox=PathSandbox(project_root),
            )

            path_allowed, path_reason = guard.check("read_file", {"path": {"bad": True}})
            cmd_allowed, cmd_reason = guard.check("run_command", {"command": ["ls"]})

            self.assertFalse(path_allowed)
            self.assertIn("字符串", path_reason)
            self.assertFalse(cmd_allowed)
            self.assertIn("字符串", cmd_reason)

    def test_apply_patch_validates_every_declared_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(
                    level=SecurityLevel.PERMISSIVE,
                    project_root=project_root,
                ),
                sandbox=PathSandbox(project_root),
            )
            patch = (
                "*** Begin Patch\n"
                "*** Add File: safe.txt\n"
                "+ok\n"
                "*** Add File: ../outside.txt\n"
                "+bad\n"
                "*** End Patch"
            )

            allowed, reason = guard.check("apply_patch", {"patch": patch})

            self.assertFalse(allowed)
            self.assertIn("路径", reason)

    def test_apply_patch_session_approval_is_scoped_to_each_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            policy = SecurityPolicy(level=SecurityLevel.NORMAL, project_root=project_root)
            guard = SecurityGuard(policy=policy, sandbox=PathSandbox(project_root))
            params = {
                "patch": (
                    "*** Begin Patch\n"
                    "*** Update File: a.py\n"
                    "*** Update File: b.py\n"
                    "*** End Patch"
                )
            }

            guard.apply_hitl(HITLDecision.ALLOW_SESSION, "apply_patch", params)

            self.assertEqual(RuleAction.ALLOW, policy.evaluate("apply_patch", path="a.py"))
            self.assertEqual(RuleAction.ALLOW, policy.evaluate("apply_patch", path="b.py"))
            self.assertEqual(RuleAction.ASK, policy.evaluate("apply_patch", path="c.py"))

    def test_dynamic_tool_approval_is_scoped_to_capability_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            policy = SecurityPolicy(level=SecurityLevel.STRICT, project_root=project_root)
            guard = SecurityGuard(policy=policy, sandbox=PathSandbox(project_root))

            guard.apply_hitl(
                HITLDecision.ALLOW_SESSION,
                "sub_agent",
                {"command": "subagent-capabilities:reader"},
            )

            self.assertEqual(
                RuleAction.ALLOW,
                policy.evaluate(
                    "sub_agent", command="subagent-capabilities:reader",
                ),
            )
            self.assertEqual(
                RuleAction.ASK,
                policy.evaluate(
                    "sub_agent", command="subagent-capabilities:writer",
                ),
            )

    def test_security_guard_blocks_provider_config_from_model_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(level=SecurityLevel.PERMISSIVE, project_root=project_root),
                sandbox=PathSandbox(project_root),
            )

            read_allowed, read_reason = guard.check(
                "read_file", {"path": ".tinyCode.yaml"},
            )
            command_allowed, command_reason = guard.check(
                "run_command", {"command": "cat .tinyCode.yaml"},
            )

            self.assertFalse(read_allowed)
            self.assertIn("凭据", read_reason)
            self.assertFalse(command_allowed)
            self.assertIn("凭据", command_reason)

    def test_security_guard_blocks_global_config_and_environment_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            guard = SecurityGuard(
                policy=SecurityPolicy(level=SecurityLevel.PERMISSIVE, project_root=project_root),
                sandbox=PathSandbox(project_root),
            )

            for command in ("cat ~/.tinyCode/config.yaml", "env", "printenv API_KEY"):
                with self.subTest(command=command):
                    allowed, reason = guard.check("run_command", {"command": command})
                    self.assertFalse(allowed)
                    self.assertIn("凭据", reason)


if __name__ == "__main__":
    unittest.main()

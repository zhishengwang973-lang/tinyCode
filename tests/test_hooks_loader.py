import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tinyCode.hooks.loader import load_hooks
from tinyCode.hooks.models import HookEvent


class HookLoaderTests(unittest.TestCase):
    def test_malformed_control_is_skipped_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: tool_pre_exec",
                    "    control: bad",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: stop",
                    "  - event: session_start",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual(1, len(rules))
            self.assertEqual(HookEvent.SESSION_START, rules[0].event)
            self.assertIn("control", stderr.getvalue())

    def test_async_intercept_rule_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: tool_pre_exec",
                    "    control:",
                    "      async: true",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: stop",
                    "",
                ]),
                encoding="utf-8",
            )

            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual([], rules)
            self.assertIn("不允许 async=true", stderr.getvalue())

    def test_hooks_must_be_a_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  event: session_start",
                    "  actions:",
                    "    - type: prompt_inject",
                    "      text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual([], rules)
            self.assertIn("必须是列表", stderr.getvalue())

    def test_non_string_prompt_inject_text_is_skipped_without_losing_valid_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: tool_pre_exec",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text:",
                    "          - stop",
                    "  - event: session_start",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual(1, len(rules))
            self.assertEqual(HookEvent.SESSION_START, rules[0].event)
            self.assertIn("text", stderr.getvalue())

    def test_non_boolean_async_control_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: session_start",
                    "    control:",
                    "      async: \"false\"",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual([], rules)
            self.assertIn("async", stderr.getvalue())

    def test_non_numeric_timeout_control_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: session_start",
                    "    control:",
                    "      timeout: fast",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual([], rules)
            self.assertIn("timeout", stderr.getvalue())

    def test_non_finite_timeout_control_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "hooks:\n"
                "  - event: session_start\n"
                "    control:\n"
                "      timeout: .nan\n"
                "    actions:\n"
                "      - type: prompt_inject\n"
                "        text: hello\n",
                encoding="utf-8",
            )

            rules = load_hooks(
                project_path=hook_path,
                global_path=Path(tmp) / "missing.yaml",
            )

            self.assertEqual([], rules)

    def test_non_string_condition_match_is_skipped_without_losing_valid_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: session_start",
                    "    condition:",
                    "      match: false",
                    "      rules:",
                    "        - field: tool_name",
                    "          operator: exact",
                    "          value: read_file",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: bad",
                    "  - event: session_start",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual(1, len(rules))
            self.assertEqual(HookEvent.SESSION_START, rules[0].event)
            self.assertIn("condition.match", stderr.getvalue())

    def test_non_string_condition_field_is_skipped_without_losing_valid_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook_path = Path(tmp) / ".tinyCode-hooks.yaml"
            hook_path.write_text(
                "\n".join([
                    "hooks:",
                    "  - event: session_start",
                    "    condition:",
                    "      rules:",
                    "        - field:",
                    "            - tool_name",
                    "          operator: exact",
                    "          value: read_file",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: bad",
                    "  - event: session_start",
                    "    actions:",
                    "      - type: prompt_inject",
                    "        text: hello",
                    "",
                ]),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                rules = load_hooks(project_path=hook_path, global_path=Path(tmp) / "missing.yaml")

            self.assertEqual(1, len(rules))
            self.assertEqual(HookEvent.SESSION_START, rules[0].event)
            self.assertIn("condition.field", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

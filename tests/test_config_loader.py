import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.config.constants import DEFAULT_NOTES_ENABLED
from tinyCode.config.loader import ConfigError, load_config, load_provider_config


class ConfigLoaderTests(unittest.TestCase):
    def test_load_named_provider_resolves_only_the_requested_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: executor\n"
                "    protocol: openai\n"
                "    model: executor-model\n"
                "    api_key: executor-key\n"
                "  - name: judge\n"
                "    protocol: openai\n"
                "    model: judge-model\n"
                "    api_key_env: MISSING_JUDGE_KEY\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                executor = load_provider_config("executor")
                with self.assertRaisesRegex(ConfigError, "MISSING_JUDGE_KEY"):
                    load_provider_config("judge")

        self.assertEqual("executor-key", executor.api_key)
        self.assertEqual("executor-model", executor.model)

    def test_empty_providers_exits_with_readable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text("providers: []\n", encoding="utf-8")
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "providers"):
                    load_config()

    def test_top_level_config_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text("providers\n", encoding="utf-8")
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "配置文件顶层必须是对象"):
                    load_config()

    def test_provider_gets_default_base_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text(
                "\n".join([
                    "providers:",
                    "  - name: openai",
                    "    protocol: openai",
                    "    model: gpt-test",
                    "    api_key: test-key",
                    "",
                ]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                config = load_config()

            self.assertEqual("openai", config.active_provider)
            self.assertEqual("https://api.openai.com", config.providers[0].base_url)
            self.assertEqual(30, config.max_rounds)
            self.assertEqual(10, config.round_extension)
            self.assertEqual(100, config.hard_max_rounds)
            self.assertEqual("ask", config.round_limit_action)
            self.assertEqual("normal", config.security_level)
            self.assertEqual("stream", config.ui_mode)
            self.assertEqual(DEFAULT_NOTES_ENABLED, config.notes_enabled)
            self.assertTrue(config.tracing.enabled)
            self.assertFalse(config.tracing.capture_payloads)
            self.assertFalse(config.task_mode_routing.enabled)

    def test_tracing_is_configurable_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "tracing:\n"
                "  enabled: false\n"
                "  capture_payloads: true\n"
                "  retention_days: 30\n"
                "  max_files: 250\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                config = load_config()

        self.assertFalse(config.tracing.enabled)
        self.assertTrue(config.tracing.capture_payloads)
        self.assertEqual(30, config.tracing.retention_days)
        self.assertEqual(250, config.tracing.max_files)

    def test_tracing_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "tracing:\n"
                "  max_files: 0\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "tracing.max_files"):
                    load_config()

    def test_task_mode_routing_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "task_mode_routing:\n"
                "  enabled: true\n"
                "  api_key_env: TEST_TYPESAFE_KEY\n"
                "  model: jev-1.13.0\n"
                "  confidence_threshold: 0.85\n"
                "  timeout_seconds: 3\n"
                "  llm_timeout_seconds: 15\n"
                "  llm_fallback: true\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "TINYCODE_CONFIG": str(config_path),
                "TEST_TYPESAFE_KEY": "jev-secret",
            }, clear=False):
                config = load_config()

        routing = config.task_mode_routing
        self.assertTrue(routing.enabled)
        self.assertEqual("jev-secret", routing.api_key)
        self.assertEqual("jev-1.13.0", routing.model)
        self.assertEqual(0.85, routing.confidence_threshold)
        self.assertEqual(3.0, routing.timeout_seconds)
        self.assertEqual(15.0, routing.llm_timeout_seconds)
        self.assertTrue(routing.llm_fallback)

    def test_enabled_task_mode_routing_requires_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "task_mode_routing:\n"
                "  enabled: true\n"
                "  api_key_env: MISSING_TYPESAFE_KEY\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "TINYCODE_CONFIG": str(config_path),
            }, clear=True):
                with self.assertRaisesRegex(ConfigError, "MISSING_TYPESAFE_KEY"):
                    load_config()

    def test_notes_enabled_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "notes_enabled: false\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                config = load_config()

        self.assertFalse(config.notes_enabled)

    def test_notes_enabled_rejects_non_boolean_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "notes_enabled: disabled\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(
                    ConfigError, "notes_enabled 必须是 true 或 false",
                ):
                    load_config()

    def test_security_level_is_configurable_and_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text(
                "\n".join([
                    "providers:",
                    "  - name: openai",
                    "    protocol: openai",
                    "    model: gpt-test",
                    "    api_key: test-key",
                    "security_level: PERMISSIVE",
                    "",
                ]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                config = load_config()

            self.assertEqual("permissive", config.security_level)

    def test_invalid_security_level_exits_with_readable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text(
                "\n".join([
                    "providers:",
                    "  - name: openai",
                    "    protocol: openai",
                    "    model: gpt-test",
                    "    api_key: test-key",
                    "security_level: unrestricted",
                    "",
                ]),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(
                    ConfigError, "security_level 必须是 strict、normal 或 permissive",
                ):
                    load_config()

    def test_ui_mode_is_configurable_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "ui_mode: FULLSCREEN\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                self.assertEqual("fullscreen", load_config().ui_mode)

            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "ui_mode: graphical\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "ui_mode 必须是 stream 或 fullscreen"):
                    load_config()

    def test_max_rounds_is_configurable_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text(
                "\n".join([
                    "providers:",
                    "  - name: openai",
                    "    protocol: openai",
                    "    model: gpt-test",
                    "    api_key: test-key",
                    "max_rounds: 101",
                    "",
                ]),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "max_rounds"):
                    load_config()

    def test_round_budget_policy_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "max_rounds: 20\n"
                "round_extension: 5\n"
                "hard_max_rounds: 60\n"
                "round_limit_action: AUTO\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                config = load_config()

        self.assertEqual(20, config.max_rounds)
        self.assertEqual(5, config.round_extension)
        self.assertEqual(60, config.hard_max_rounds)
        self.assertEqual("auto", config.round_limit_action)

    def test_soft_round_budget_cannot_exceed_hard_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "max_rounds: 30\n"
                "hard_max_rounds: 20\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "hard_max_rounds"):
                    load_config()

    def test_round_limit_action_rejects_unknown_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "round_limit_action: forever\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "ask、auto 或 stop"):
                    load_config()

    def test_provider_entry_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text(
                "\n".join([
                    "providers:",
                    "  - openai",
                    "",
                ]),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "Provider #0.*对象"):
                    load_config()

    def test_provider_names_must_be_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode.yaml"
            config_path.write_text(
                "\n".join([
                    "providers:",
                    "  - name: main",
                    "    protocol: openai",
                    "    model: gpt-test",
                    "    api_key: test-key",
                    "  - name: main",
                    "    protocol: deepseek",
                    "    model: deepseek-chat",
                    "    api_key: test-key",
                    "active_provider: main",
                    "",
                ]),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=False):
                with self.assertRaisesRegex(ConfigError, "Provider 名称重复: main"):
                    load_config()

    def test_project_inherits_global_security_without_merging_provider_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: global\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: global-secret\n"
                "security_level: strict\n"
                "max_rounds: 40\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "max_rounds: 12\n", encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": ""}, clear=False), \
                 patch("tinyCode.config.loader.Path.home", return_value=home), \
                 patch("tinyCode.config.loader.Path.cwd", return_value=project):
                config = load_config()

            self.assertEqual("strict", config.security_level)
            self.assertEqual(12, config.max_rounds)
            self.assertEqual("global-secret", config.providers[0].api_key)

    def test_project_cannot_enable_global_trace_payload_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: global\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n"
                "tracing:\n"
                "  enabled: true\n"
                "  capture_payloads: false\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "tracing:\n"
                "  capture_payloads: true\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": ""}, clear=False), \
                 patch("tinyCode.config.loader.Path.home", return_value=home), \
                 patch("tinyCode.config.loader.Path.cwd", return_value=project):
                config = load_config()

            self.assertTrue(config.tracing.enabled)
            self.assertFalse(config.tracing.capture_payloads)

    def test_project_cannot_enable_task_mode_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: global\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key: test-key\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "task_mode_routing:\n"
                "  enabled: true\n"
                "  api_key_env: STOLEN_KEY\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {
                "TINYCODE_CONFIG": "", "STOLEN_KEY": "secret",
            }, clear=False), patch(
                "tinyCode.config.loader.Path.home", return_value=home,
            ), patch(
                "tinyCode.config.loader.Path.cwd", return_value=project,
            ):
                config = load_config()

        self.assertFalse(config.task_mode_routing.enabled)
        self.assertIsNone(config.task_mode_routing.api_key)

    def test_project_provider_never_inherits_global_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: main\n"
                "    protocol: openai\n"
                "    model: global-model\n"
                "    api_key: global-secret\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "providers:\n"
                "  - name: local\n"
                "    protocol: openai\n"
                "    model: local-model\n"
                "    base_url: https://project.invalid\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": ""}, clear=True), \
                 patch("tinyCode.config.loader.Path.home", return_value=home), \
                 patch("tinyCode.config.loader.Path.cwd", return_value=project):
                with self.assertRaisesRegex(ConfigError, "缺少密钥"):
                    load_config()

    def test_api_key_can_come_from_named_environment_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: openai\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key_env: TINYCODE_TEST_API_KEY\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "TINYCODE_CONFIG": str(config_path),
                "TINYCODE_TEST_API_KEY": "from-env",
            }, clear=True):
                config = load_config()

            self.assertEqual("from-env", config.providers[0].api_key)

    def test_only_active_provider_requires_its_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: active\n"
                "    protocol: anthropic\n"
                "    model: claude-test\n"
                "    api_key_env: ACTIVE_TEST_KEY\n"
                "  - name: inactive\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key_env: INACTIVE_TEST_KEY\n"
                "active_provider: active\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "TINYCODE_CONFIG": str(config_path),
                "ACTIVE_TEST_KEY": "active-secret",
            }, clear=True):
                config = load_config()

            self.assertEqual("active-secret", config.providers[0].api_key)
            self.assertIsNone(config.providers[1].api_key)

    def test_inactive_provider_secret_declaration_is_still_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: active\n"
                "    protocol: anthropic\n"
                "    model: claude-test\n"
                "    api_key: active-secret\n"
                "  - name: inactive\n"
                "    protocol: openai\n"
                "    model: gpt-test\n"
                "    api_key_env: not-valid-name\n"
                "active_provider: active\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=True):
                with self.assertRaisesRegex(ConfigError, "api_key_env"):
                    load_config()

    def test_context_window_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  - name: deepseek\n"
                "    protocol: deepseek\n"
                "    model: deepseek-chat\n"
                "    api_key: test-key\n"
                "    context_window: 131072\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYCODE_CONFIG": str(config_path)}, clear=True):
                config = load_config()

            self.assertEqual(131072, config.providers[0].context_window)

    def test_project_config_cannot_weaken_explicit_global_security(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: global\n"
                "    protocol: openai\n"
                "    model: test\n"
                "    api_key: key\n"
                "security_level: strict\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "security_level: permissive\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": ""}, clear=False), \
                 patch("tinyCode.config.loader.Path.home", return_value=home), \
                 patch("tinyCode.config.loader.Path.cwd", return_value=project):
                config = load_config()

            self.assertEqual("strict", config.security_level)

    def test_project_config_can_tighten_explicit_global_security(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: global\n"
                "    protocol: openai\n"
                "    model: test\n"
                "    api_key: key\n"
                "security_level: permissive\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "security_level: strict\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": ""}, clear=False), \
                 patch("tinyCode.config.loader.Path.home", return_value=home), \
                 patch("tinyCode.config.loader.Path.cwd", return_value=project):
                config = load_config()

            self.assertEqual("strict", config.security_level)

    def test_project_config_cannot_override_global_notes_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            project = root / "project"
            (home / ".tinyCode").mkdir(parents=True)
            project.mkdir()
            (home / ".tinyCode" / "config.yaml").write_text(
                "providers:\n"
                "  - name: global\n"
                "    protocol: openai\n"
                "    model: test\n"
                "    api_key: key\n"
                "notes_enabled: false\n",
                encoding="utf-8",
            )
            (project / ".tinyCode.yaml").write_text(
                "notes_enabled: true\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TINYCODE_CONFIG": ""}, clear=False), \
                 patch("tinyCode.config.loader.Path.home", return_value=home), \
                 patch("tinyCode.config.loader.Path.cwd", return_value=project):
                config = load_config()

            self.assertFalse(config.notes_enabled)


if __name__ == "__main__":
    unittest.main()

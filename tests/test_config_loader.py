import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.config.loader import ConfigError, load_config


class ConfigLoaderTests(unittest.TestCase):
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
            self.assertEqual("normal", config.security_level)

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


if __name__ == "__main__":
    unittest.main()

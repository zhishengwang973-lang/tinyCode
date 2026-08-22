import tempfile
import unittest
from pathlib import Path

from tinyCode.mcp.config import load_mcp_config


class MCPConfigTests(unittest.TestCase):
    def test_project_config_can_be_disabled_for_untrusted_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.yaml"
            project_path.write_text(
                "servers:\n  - name: local\n    transport: stdio\n    command: python3\n",
                encoding="utf-8",
            )

            servers = load_mcp_config(
                project_path=project_path,
                global_path=Path(tmp) / "missing.yaml",
                include_project=False,
            )

            self.assertEqual([], servers)
    def test_invalid_servers_are_skipped_while_valid_servers_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode-mcp.yaml"
            config_path.write_text(
                "\n".join([
                    "servers:",
                    "  - name: missing-command",
                    "    transport: stdio",
                    "  - name: missing-url",
                    "    transport: http",
                    "  - name: unsupported",
                    "    transport: websocket",
                    "  - name: valid-stdio",
                    "    transport: stdio",
                    "    command: python3",
                    "    args:",
                    "      - -m",
                    "      - server",
                    "  - name: valid-http",
                    "    transport: http",
                    "    url: https://example.com",
                    "",
                ]),
                encoding="utf-8",
            )

            diagnostics = []
            servers = load_mcp_config(
                project_path=config_path,
                global_path=Path(tmp) / "missing.yaml",
                diagnostics=diagnostics,
            )

            self.assertEqual(["valid-stdio", "valid-http"], [server.name for server in servers])
            self.assertEqual(3, len(diagnostics))

    def test_malformed_yaml_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "broken.yaml"
            config_path.write_text("servers: [", encoding="utf-8")
            diagnostics = []

            servers = load_mcp_config(
                project_path=config_path,
                global_path=Path(tmp) / "missing.yaml",
                diagnostics=diagnostics,
            )

            self.assertEqual([], servers)
            self.assertEqual(1, len(diagnostics))
            self.assertIn("读取或解析失败", diagnostics[0])

    def test_project_config_overrides_global_server_with_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_path = Path(tmp) / "global.yaml"
            project_path = Path(tmp) / "project.yaml"
            global_path.write_text(
                "\n".join([
                    "servers:",
                    "  - name: shared",
                    "    transport: http",
                    "    url: https://global.example.com",
                    "",
                ]),
                encoding="utf-8",
            )
            project_path.write_text(
                "\n".join([
                    "servers:",
                    "  - name: shared",
                    "    transport: http",
                    "    url: https://project.example.com",
                    "",
                ]),
                encoding="utf-8",
            )

            servers = load_mcp_config(project_path=project_path, global_path=global_path)

            self.assertEqual(1, len(servers))
            self.assertEqual("https://project.example.com", servers[0].url)

    def test_stdio_server_with_non_string_arg_is_skipped_while_valid_servers_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode-mcp.yaml"
            config_path.write_text(
                "\n".join([
                    "servers:",
                    "  - name: bad-args",
                    "    transport: stdio",
                    "    command: python3",
                    "    args:",
                    "      - -m",
                    "      - 7",
                    "  - name: valid-stdio",
                    "    transport: stdio",
                    "    command: python3",
                    "    args:",
                    "      - -m",
                    "      - server",
                    "",
                ]),
                encoding="utf-8",
            )

            servers = load_mcp_config(
                project_path=config_path,
                global_path=Path(tmp) / "missing.yaml",
            )

            self.assertEqual(["valid-stdio"], [server.name for server in servers])

    def test_stdio_server_with_non_string_env_value_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode-mcp.yaml"
            config_path.write_text(
                "\n".join([
                    "servers:",
                    "  - name: bad-env",
                    "    transport: stdio",
                    "    command: python3",
                    "    env:",
                    "      TOKEN: 123",
                    "  - name: valid-stdio",
                    "    transport: stdio",
                    "    command: python3",
                    "    env:",
                    "      TOKEN: ok",
                    "",
                ]),
                encoding="utf-8",
            )

            servers = load_mcp_config(
                project_path=config_path,
                global_path=Path(tmp) / "missing.yaml",
            )

            self.assertEqual(["valid-stdio"], [server.name for server in servers])

    def test_non_finite_timeout_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".tinyCode-mcp.yaml"
            config_path.write_text(
                "servers:\n"
                "  - name: bad-timeout\n"
                "    transport: http\n"
                "    url: https://example.test\n"
                "    timeout: .inf\n",
                encoding="utf-8",
            )

            servers = load_mcp_config(
                project_path=config_path,
                global_path=Path(tmp) / "missing.yaml",
            )

            self.assertEqual([], servers)


if __name__ == "__main__":
    unittest.main()

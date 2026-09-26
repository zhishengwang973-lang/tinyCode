import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import tinyCode.commands.builtin as builtin_commands
from tinyCode.commands import register_builtins
from tinyCode.commands.dispatcher import CommandDispatcher
from tinyCode.commands.builtin import (
    config_cmd,
    cancel_cmd,
    exit_cmd,
    image_cmd,
    prompt_cmd,
    memory_cmd,
    session_cmd,
    skill_cmd,
    tasks_cmd,
    team_cmd,
    trace_cmd,
    worktree_cmd,
    mode_cmd,
)
from tinyCode.commands.parser import is_command, parse
from tinyCode.commands.registry import CommandRegistry
from tinyCode.commands.types import CommandMeta, CommandType, UIControl


class CommandParserTests(unittest.TestCase):
    def test_parse_lowercases_command_name_and_preserves_quoted_args(self):
        parsed = parse('/Review --path "src/main app.py"')

        self.assertEqual("review", parsed.command_name)
        self.assertEqual(["--path", "src/main app.py"], parsed.args)

    def test_non_slash_text_is_not_a_command_and_cannot_be_parsed(self):
        self.assertFalse(is_command("please review this"))

        with self.assertRaises(ValueError):
            parse("please review this")

    def test_parse_rejects_empty_command_name(self):
        with self.assertRaises(ValueError):
            parse("/")

    def test_parse_rejects_unclosed_quoted_argument(self):
        with self.assertRaises(ValueError):
            parse('/review "src/main.py')


class CommandRegistryTests(unittest.TestCase):
    def test_lookup_finds_command_by_name_or_alias_case_insensitively(self):
        registry = CommandRegistry()
        command = CommandMeta(
            name="review",
            description="Run review",
            usage="/review",
            cmd_type=CommandType.LOCAL,
            aliases=["rv"],
        )

        registry.register(command)

        self.assertIs(command, registry.lookup("REVIEW"))
        self.assertIs(command, registry.lookup("RV"))

    def test_register_rejects_alias_that_conflicts_with_existing_command(self):
        registry = CommandRegistry()
        registry.register(CommandMeta(
            name="review",
            description="Run review",
            usage="/review",
            cmd_type=CommandType.LOCAL,
        ))

        with self.assertRaises(ValueError):
            registry.register(CommandMeta(
                name="test",
                description="Run tests",
                usage="/test",
                cmd_type=CommandType.LOCAL,
                aliases=["review"],
            ))

    def test_register_rejects_command_name_that_conflicts_with_existing_alias(self):
        registry = CommandRegistry()
        registry.register(CommandMeta(
            name="review",
            description="Run review",
            usage="/review",
            cmd_type=CommandType.LOCAL,
            aliases=["rv"],
        ))

        with self.assertRaises(ValueError):
            registry.register(CommandMeta(
                name="rv",
                description="Another command",
                usage="/rv",
                cmd_type=CommandType.LOCAL,
            ))

    def test_register_rejects_empty_command_name(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令名必须是非空字符串"):
            registry.register(CommandMeta(
                name="",
                description="Empty command",
                usage="/",
                cmd_type=CommandType.LOCAL,
            ))

    def test_register_rejects_non_string_command_name(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令名必须是非空字符串"):
            registry.register(CommandMeta(
                name=123,
                description="Bad command",
                usage="/bad",
                cmd_type=CommandType.LOCAL,
            ))

    def test_register_rejects_aliases_that_are_not_a_string_list(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令别名必须是字符串列表"):
            registry.register(CommandMeta(
                name="review",
                description="Run review",
                usage="/review",
                cmd_type=CommandType.LOCAL,
                aliases="rv",
            ))

    def test_register_rejects_empty_or_non_string_alias(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令别名必须是非空字符串"):
            registry.register(CommandMeta(
                name="review",
                description="Run review",
                usage="/review",
                cmd_type=CommandType.LOCAL,
                aliases=[""],
            ))

        with self.assertRaisesRegex(ValueError, "命令别名必须是非空字符串"):
            registry.register(CommandMeta(
                name="test",
                description="Run tests",
                usage="/test",
                cmd_type=CommandType.LOCAL,
                aliases=[123],
            ))

    def test_register_rejects_non_string_description_or_usage(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令描述必须是字符串"):
            registry.register(CommandMeta(
                name="review",
                description=123,
                usage="/review",
                cmd_type=CommandType.LOCAL,
            ))

        with self.assertRaisesRegex(ValueError, "命令用法必须是字符串"):
            registry.register(CommandMeta(
                name="test",
                description="Run tests",
                usage=123,
                cmd_type=CommandType.LOCAL,
            ))

    def test_register_rejects_invalid_command_type(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令类型无效"):
            registry.register(CommandMeta(
                name="review",
                description="Run review",
                usage="/review",
                cmd_type="local",
            ))

    def test_register_rejects_non_callable_handler(self):
        registry = CommandRegistry()

        with self.assertRaisesRegex(ValueError, "命令 handler 必须可调用"):
            registry.register(CommandMeta(
                name="review",
                description="Run review",
                usage="/review",
                cmd_type=CommandType.LOCAL,
                handler="not-callable",
            ))


class BuiltinCommandPackageTests(unittest.TestCase):
    def test_public_exports_include_every_builtin_command_module(self):
        expected = {
            "clear_cmd",
            "compress_cmd",
            "config_cmd",
            "cancel_cmd",
            "exit_cmd",
            "prompt_cmd",
            "help_cmd",
            "image_cmd",
            "memory_cmd",
            "mode_cmd",
            "permission_cmd",
            "review_cmd",
            "session_cmd",
            "skill_cmd",
            "status_cmd",
            "tasks_cmd",
            "team_cmd",
            "trace_cmd",
            "worktree_cmd",
        }

        self.assertEqual(expected, set(builtin_commands.__all__))


class FakeUI(UIControl):
    def __init__(self):
        self.injected: list[str] = []
        self.max_rounds = 30
        self.round_extension = 10
        self.hard_max_rounds = 100
        self.round_limit_action = "ask"
        self.exit_requested = False
        self.cancelled = False
        self.image_supported = False
        self.images: list[tuple[list[dict], str]] = []
        self.image_attachment_root = Path.cwd()

    def show_system_message(self, text: str) -> None:
        pass

    def send_to_conversation(self, text: str) -> None:
        self.injected.append(text)

    def supports_image_input(self) -> bool:
        return self.image_supported

    def send_image_to_conversation(
        self, content: list[dict], display_text: str,
    ) -> bool:
        self.images.append((content, display_text))
        return True

    def get_image_attachment_root(self) -> Path:
        return self.image_attachment_root

    def toggle_plan_mode(self) -> bool:
        return False

    def set_security_level(self, level_name: str) -> str:
        return level_name

    def get_token_count(self) -> int:
        return 0

    def clear_conversation(self) -> None:
        pass

    async def trigger_compress(self) -> str:
        return ""

    def get_session_list(self) -> list[dict]:
        return []

    def load_session(self, session_id: str) -> str:
        return session_id

    def get_plan_only(self) -> bool:
        return False

    def get_security_level(self) -> str:
        return "normal"

    def get_max_rounds(self) -> int:
        return self.max_rounds

    def set_max_rounds(self, value: int) -> int:
        self.max_rounds = value
        return value

    def get_round_extension(self) -> int:
        return self.round_extension

    def set_round_extension(self, value: int) -> int:
        self.round_extension = value
        return value

    def get_hard_max_rounds(self) -> int:
        return self.hard_max_rounds

    def set_hard_max_rounds(self, value: int) -> int:
        self.hard_max_rounds = value
        return value

    def get_round_limit_action(self) -> str:
        return self.round_limit_action

    def set_round_limit_action(self, value: str) -> str:
        self.round_limit_action = value
        return value

    def request_exit(self) -> None:
        self.exit_requested = True

    def cancel_active_turn(self) -> bool:
        self.cancelled = True
        return True

    def get_system_prompt(self, section: str = "all") -> str:
        return f"prompt:{section}"

    async def confirm_action(self, prompt: str) -> bool:
        return True


class FakeNoteManager:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def read_note(self, category: str) -> str:
        return ""

    def clear_note(self, category: str) -> str:
        self.cleared.append(category)
        return f"cleared:{category}"

    def get_note_path(self, category: str) -> str:
        return f"/notes/{category}.md"


class FakeTaskManager:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def get_status_summary(self) -> str:
        return "没有后台任务"

    def get(self, task_id: str):
        return None

    def cancel(self, task_id: str) -> bool:
        self.cancelled.append(task_id)
        return True


class FakeSkillRegistry:
    activated = []

    def __init__(self) -> None:
        self.reloaded = 0

    def list_available(self) -> list:
        return []

    def load_all(self) -> None:
        self.reloaded += 1

    def clear_activated(self) -> None:
        pass

    def get_skill(self, name: str):
        return None

    def get_meta(self, name: str):
        return None


class FakeWorktreeManager:
    def __init__(self) -> None:
        self._repo_root = Path.cwd()
        self.exited: list[tuple[str, bool]] = []

    @property
    def active(self) -> str:
        return ""

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def is_available(self) -> bool:
        return True

    @property
    def availability_error(self) -> str:
        return ""

    async def status(self):
        return None

    async def list_worktrees(self) -> list:
        return []

    async def create(self, name: str, branch: str = ""):
        return None, "not implemented"

    async def enter(self, name: str):
        return True, ""

    async def exit(self, name: str, force: bool = False):
        self.exited.append((name, force))
        return True, f"exited:{name}:{force}"


class CommandDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_security_rejects_invalid_level_explicitly(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(mode_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, result = await dispatcher.dispatch("/mode security unsafe")

        self.assertIn("必须是 strict、normal 或 permissive", result)

    async def test_prompt_command_reads_selected_section_without_model_injection(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(prompt_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        was_command, result = await dispatcher.dispatch("/sp instructions")

        self.assertTrue(was_command)
        self.assertEqual("prompt:instructions", result)
        self.assertEqual([], ui.injected)

    async def test_prompt_command_rejects_unknown_section(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(prompt_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, result = await dispatcher.dispatch("/prompt secrets")

        self.assertIn("未知部分: secrets", result)
        self.assertIn("用法: /prompt", result)

    async def test_exit_command_requests_graceful_shutdown(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(exit_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        was_command, result = await dispatcher.dispatch("/quit")

        self.assertTrue(was_command)
        self.assertTrue(ui.exit_requested)
        self.assertIn("安全退出", result)

    async def test_exit_command_rejects_arguments(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(exit_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, result = await dispatcher.dispatch("/exit now")

        self.assertEqual("用法: /exit", result)
        self.assertFalse(ui.exit_requested)

    async def test_cancel_command_cancels_active_turn(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(cancel_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        was_command, result = await dispatcher.dispatch("/cancel")

        self.assertTrue(was_command)
        self.assertTrue(ui.cancelled)
        self.assertIn("正在取消", result)

    async def test_cancel_command_rejects_arguments(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(cancel_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, result = await dispatcher.dispatch("/cancel now")

        self.assertEqual("用法: /cancel", result)
        self.assertFalse(ui.cancelled)

    async def test_config_max_rounds_can_be_queried_and_changed_for_session(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(config_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, initial = await dispatcher.dispatch("/config max-rounds")
        _, changed = await dispatcher.dispatch("/config max-rounds 50")
        _, current = await dispatcher.dispatch("/cfg max_rounds")

        self.assertEqual("当前会话初始轮次预算: 30", initial)
        self.assertIn("已设为 50", changed)
        self.assertEqual("当前会话初始轮次预算: 50", current)
        self.assertEqual(50, ui.max_rounds)

    async def test_config_round_budget_policy_can_be_changed_for_session(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(config_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, extension = await dispatcher.dispatch("/config round-extension 15")
        _, hard_limit = await dispatcher.dispatch("/config hard-max-rounds 80")
        _, action = await dispatcher.dispatch("/config round-limit-action auto")
        _, summary = await dispatcher.dispatch("/config")

        self.assertIn("续跑步长已设为 15", extension)
        self.assertIn("轮次硬上限已设为 80", hard_limit)
        self.assertIn("预算耗尽策略已设为 auto", action)
        self.assertIn("初始预算: 30", summary)
        self.assertIn("续跑步长: 15", summary)
        self.assertIn("硬上限: 80", summary)
        self.assertIn("达到预算时: auto", summary)

    async def test_config_max_rounds_supports_increment_syntax(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(config_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, result = await dispatcher.dispatch("/config max-rounds +10")

        self.assertIn("已设为 40", result)
        self.assertEqual(40, ui.max_rounds)

    async def test_config_max_rounds_rejects_invalid_values(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(config_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        for value in ("0", "101", "abc"):
            with self.subTest(value=value):
                _, result = await dispatcher.dispatch(
                    f"/config max-rounds {value}"
                )
                self.assertEqual("最大轮次必须是 1 到 100 之间的整数", result)
        self.assertEqual(30, ui.max_rounds)

    async def test_register_builtins_registers_runtime_config_command(self):
        registry = CommandRegistry()
        register_builtins(registry, ui=FakeUI())

        self.assertIsNotNone(registry.lookup("config"))
        self.assertIs(registry.lookup("config"), registry.lookup("cfg"))
        self.assertIsNotNone(registry.lookup("cancel"))
        self.assertIsNotNone(registry.lookup("image"))

    async def test_image_command_builds_multimodal_turn_for_vision_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "screen.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            )
            ui = FakeUI()
            ui.image_supported = True
            ui.image_attachment_root = root
            registry = CommandRegistry()
            registry.register(image_cmd.create(ui))
            dispatcher = CommandDispatcher(registry, ui=ui)

            was_command, result = await dispatcher.dispatch(
                f'/image --detail low "{image_path}" 分析这个截图'
            )

        self.assertTrue(was_command)
        self.assertIsNone(result)
        self.assertEqual(1, len(ui.images))
        content, display = ui.images[0]
        self.assertEqual("image_file", content[1]["type"])
        self.assertEqual("low", content[1]["image_file"]["detail"])
        self.assertIn("screen.png", display)
        self.assertIn("分析这个截图", display)

    async def test_image_command_explains_unsupported_model(self):
        ui = FakeUI()
        registry = CommandRegistry()
        registry.register(image_cmd.create(ui))
        dispatcher = CommandDispatcher(registry, ui=ui)

        _, result = await dispatcher.dispatch("/image screenshot.png 看一下")

        self.assertIn("deepseek-flash", result)
        self.assertEqual([], ui.images)

    async def test_trace_command_controls_and_renders_recorder(self):
        class FakeTraceRecorder:
            enabled = True
            last_error = ""
            storage_dir = Path("/tmp/traces")

            def status_text(self):
                return "Trace: ON"

            def render_last_text(self):
                return "trace tree"

            def latest_path(self):
                return Path("/tmp/traces/latest.jsonl")

            def open_last(self):
                return Path("/tmp/traces/latest.html")

            def set_enabled(self, enabled):
                self.enabled = enabled

            def clear(self):
                return 2

        recorder = FakeTraceRecorder()
        registry = CommandRegistry()
        registry.register(trace_cmd.create(recorder))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        self.assertEqual((True, "Trace: ON"), await dispatcher.dispatch("/trace"))
        self.assertEqual((True, "trace tree"), await dispatcher.dispatch("/trace last"))
        _, opened = await dispatcher.dispatch("/trace open")
        self.assertIn("latest.html", opened)
        _, disabled = await dispatcher.dispatch("/trace off")
        self.assertIn("已关闭", disabled)
        self.assertFalse(recorder.enabled)

    async def test_dispatch_returns_false_for_non_command_input(self):
        dispatcher = CommandDispatcher(CommandRegistry(), ui=FakeUI())

        was_command, result = await dispatcher.dispatch("please review this")

        self.assertFalse(was_command)
        self.assertIsNone(result)

    async def test_dispatch_reports_unknown_command(self):
        dispatcher = CommandDispatcher(CommandRegistry(), ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/missing")

        self.assertTrue(was_command)
        self.assertIn("未知命令", result)

    async def test_dispatch_reports_parse_error_for_empty_command_name(self):
        dispatcher = CommandDispatcher(CommandRegistry(), ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/")

        self.assertTrue(was_command)
        self.assertEqual("命令解析失败", result)

    async def test_dispatch_reports_parse_error_for_unclosed_quote(self):
        dispatcher = CommandDispatcher(CommandRegistry(), ui=FakeUI())

        was_command, result = await dispatcher.dispatch('/review "src/main.py')

        self.assertTrue(was_command)
        self.assertEqual("命令解析失败", result)

    async def test_dispatch_reports_handler_error_without_raising(self):
        registry = CommandRegistry()

        async def handler(args: list[str]) -> str:
            raise RuntimeError("boom")

        registry.register(CommandMeta(
            name="explode",
            description="Explode",
            usage="/explode",
            cmd_type=CommandType.LOCAL,
            handler=handler,
        ))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/explode")

        self.assertTrue(was_command)
        self.assertIn("命令执行失败", result)

    async def test_session_list_formats_malformed_metadata_without_failing(self):
        class MalformedSessionUI(FakeUI):
            def get_session_list(self) -> list[dict]:
                return [
                    {
                        "id": 12345,
                        "title": ["not", "a", "title"],
                        "message_count": "many",
                        "last_active_at": {"bad": "timestamp"},
                    }
                ]

        registry = CommandRegistry()
        registry.register(session_cmd.create(MalformedSessionUI()))
        dispatcher = CommandDispatcher(registry, ui=MalformedSessionUI())

        was_command, result = await dispatcher.dispatch("/session list")

        self.assertTrue(was_command)
        self.assertIsNotNone(result)
        self.assertIn("会话列表:", result)
        self.assertIn("12345", result)
        self.assertIn("无标题", result)
        self.assertIn("0 条消息", result)
        self.assertNotIn("命令执行失败", result)

    async def test_session_list_subcommand_is_case_insensitive(self):
        registry = CommandRegistry()
        registry.register(session_cmd.create(FakeUI()))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/session LIST")

        self.assertTrue(was_command)
        self.assertEqual("没有保存的会话", result)

    async def test_memory_subcommand_is_case_insensitive_without_mutating_category(self):
        note_manager = FakeNoteManager()
        registry = CommandRegistry()
        registry.register(memory_cmd.create(note_manager))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/memory CLEAR ProjectKnowledge")

        self.assertTrue(was_command)
        self.assertEqual("cleared:ProjectKnowledge", result)
        self.assertEqual(["ProjectKnowledge"], note_manager.cleared)

    async def test_tasks_subcommand_is_case_insensitive_without_mutating_task_id(self):
        task_manager = FakeTaskManager()
        registry = CommandRegistry()
        registry.register(tasks_cmd.create(task_manager))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/tasks KILL TaskABC")

        self.assertTrue(was_command)
        self.assertEqual("任务 TaskABC 已取消", result)
        self.assertEqual(["TaskABC"], task_manager.cancelled)

    async def test_skill_subcommand_is_case_insensitive(self):
        skill_registry = FakeSkillRegistry()
        registry = CommandRegistry()
        registry.register(skill_cmd.create(skill_registry, ui=FakeUI()))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/skill RELOAD")

        self.assertTrue(was_command)
        self.assertEqual("已重新扫描，共 0 个 Skills", result)
        self.assertEqual(1, skill_registry.reloaded)

    async def test_team_subcommand_is_case_insensitive_without_mutating_team_name(self):
        registry = CommandRegistry()
        registry.register(team_cmd.create())
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        with patch("tinyCode.commands.builtin.team_cmd.get_team_dir") as get_team_dir:
            get_team_dir.return_value = "/teams/TeamAlpha"
            was_command, result = await dispatcher.dispatch("/team DIR TeamAlpha")

        self.assertTrue(was_command)
        self.assertEqual("Team 工作目录: /teams/TeamAlpha", result)
        get_team_dir.assert_called_once_with("TeamAlpha")

    async def test_team_run_invokes_runner_with_preserved_goal_text(self):
        calls = []

        async def runner(team_name: str, goal: str) -> str:
            calls.append((team_name, goal))
            return "team result"

        registry = CommandRegistry()
        registry.register(team_cmd.create(runner=runner))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/team RUN TeamAlpha 修复 bug 并补测试")

        self.assertTrue(was_command)
        self.assertEqual("team result", result)
        self.assertEqual([("TeamAlpha", "修复 bug 并补测试")], calls)

    async def test_team_run_requires_name_and_goal(self):
        registry = CommandRegistry()
        registry.register(team_cmd.create())
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        _, missing_all = await dispatcher.dispatch("/team run")
        _, missing_goal = await dispatcher.dispatch("/team run TeamAlpha")

        self.assertEqual("用法: /team run <名称> <目标>", missing_all)
        self.assertEqual("用法: /team run <名称> <目标>", missing_goal)

    async def test_register_builtins_wires_team_runner(self):
        calls = []

        async def runner(team_name: str, goal: str) -> str:
            calls.append((team_name, goal))
            return "wired"

        registry = CommandRegistry()
        register_builtins(registry, ui=FakeUI(), team_runner=runner)
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/team run AlphaTeam ship it")

        self.assertTrue(was_command)
        self.assertEqual("wired", result)
        self.assertEqual([("AlphaTeam", "ship it")], calls)

    async def test_team_review_commands_use_durable_review_service(self):
        class ReviewService:
            def list_reviews(self):
                return [type("Record", (), {
                    "run_id": "abcdef123456",
                    "status": "ready",
                    "goal": "ship feature",
                })()]

            def show_review(self, run_id):
                return f"show:{run_id}"

            async def apply(self, run_id):
                return True, f"applied:{run_id}"

            async def discard(self, run_id):
                return True, f"discarded:{run_id}"

        registry = CommandRegistry()
        registry.register(team_cmd.create(review_service=ReviewService()))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        _, listed = await dispatcher.dispatch("/team review list")
        _, shown = await dispatcher.dispatch("/team review show abcdef123456")

        self.assertIn("abcdef123456 · ready", listed)
        self.assertEqual("show:abcdef123456", shown)

    async def test_worktree_exit_subcommand_is_case_insensitive_without_mutating_name(self):
        manager = FakeWorktreeManager()
        registry = CommandRegistry()
        registry.register(worktree_cmd.create(manager))
        dispatcher = CommandDispatcher(registry, ui=FakeUI())

        was_command, result = await dispatcher.dispatch("/worktree EXIT FeatureABC --force")

        self.assertTrue(was_command)
        self.assertEqual("exited:FeatureABC:True", result)
        self.assertEqual([("FeatureABC", True)], manager.exited)

    async def test_prompt_inject_command_sends_handler_output_to_conversation(self):
        registry = CommandRegistry()
        ui = FakeUI()

        async def handler(args: list[str]) -> str:
            return f"Injected: {' '.join(args)}"

        registry.register(CommandMeta(
            name="review",
            description="Inject review prompt",
            usage="/review",
            cmd_type=CommandType.PROMPT_INJECT,
            handler=handler,
        ))
        dispatcher = CommandDispatcher(registry, ui=ui)

        was_command, result = await dispatcher.dispatch("/review src/app.py")

        self.assertTrue(was_command)
        self.assertIsNone(result)
        self.assertEqual(["Injected: src/app.py"], ui.injected)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from io import StringIO
from pathlib import Path

from tinyCode.config.models import ProviderConfig
from tinyCode.evals.loader import EvalConfigError, load_case
from tinyCode.evals.cli import _TerminalProgress, _load_cases
from tinyCode.evals.runner import EvalRunner
from tinyCode.providers.base import BaseProvider, Message, ToolCall


class EvalProvider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    async def chat_stream(self, messages, tools=None, system_blocks=None):
        del tools, system_blocks
        self.calls += 1
        self.last_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        if self.config.name == "judge":
            yield ('{"tool_process":5,"instruction_following":15,'
                   '"code_quality":10,"rationale":"符合要求"}')
            return
        if self.config.name == "bad-judge":
            yield "这不是 JSON"
            return
        if self.calls == 1:
            yield ToolCall("call-read", "read_file", {"path": "greeting.py"})
        elif self.calls == 2:
            yield ToolCall("call-patch", "apply_patch", {
                "patch": "*** Begin Patch\n*** Update File: greeting.py\n@@\n-    return f\"Hello, {name}\"\n+    return f\"你好，{name}\"\n*** End Patch",
            })
        else:
            yield "已修复 greeting.py，测试已通过。"

    def make_tool_calls_message(self, tool_calls, text_prefix=""):
        return {
            "role": "assistant",
            "content": text_prefix,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": "{}",
                }}
                for call in tool_calls
            ],
        }

    def make_tool_result_message(self, tool_call_id, tool_name, result_text):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result_text,
        }


class EvalTests(unittest.IsolatedAsyncioTestCase):
    def _case(self, root: Path) -> Path:
        fixture = root / "fixture"
        fixture.mkdir()
        (fixture / "greeting.py").write_text(
            'def greet(name: str) -> str:\n    return f"Hello, {name}"\n',
            encoding="utf-8",
        )
        (fixture / "test_greeting.py").write_text(
            "import unittest\nfrom greeting import greet\n"
            "class T(unittest.TestCase):\n"
            "    def test_greeting(self):\n"
            "        self.assertEqual('你好，小明', greet('小明'))\n",
            encoding="utf-8",
        )
        case = root / "case.yaml"
        case.write_text(
            "name: greeting 修复\n"
            "prompt: 修复 greeting.py\n"
            "fixture: fixture\n"
            "assertions:\n"
            "  tests: [python3 -m unittest]\n"
            "  file_contains:\n"
            "    - path: greeting.py\n"
            "      text: 'return f\"你好，{name}\"'\n"
            "  final_text_contains: [测试已通过]\n"
            "  tools_used: [read_file, apply_patch]\n"
            "  tools_not_used: [delete_file]\n"
            "budgets:\n"
            "  max_rounds: 5\n"
            "  max_model_requests: 5\n"
            "  max_tokens: 1000\n"
            "  max_duration_seconds: 30\n",
            encoding="utf-8",
        )
        return case

    @staticmethod
    def _config(name: str, model: str) -> ProviderConfig:
        return ProviderConfig(name=name, protocol="openai", model=model, api_key="x")

    async def test_runs_real_agent_and_combines_deterministic_and_judge_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            case = load_case(self._case(Path(directory)))
            runner = EvalRunner(
                self._config("executor", "executor-model"),
                self._config("judge", "judge-model"),
                provider_factory=EvalProvider,
            )
            progress: list[tuple[float, str]] = []
            report = await runner.run(
                case,
                progress=lambda fraction, message: progress.append((fraction, message)),
            )

        self.assertEqual("no_tool_call", report.status)
        self.assertEqual(100.0, report.score)
        self.assertEqual(70.0, report.deterministic_score)
        self.assertEqual(30.0, report.judge_score)
        self.assertEqual(("read_file", "apply_patch"), report.tools)
        self.assertTrue(all(check.passed for check in report.checks))
        self.assertIn("greeting.py", report.workspace_changes["modified"])
        self.assertIn("--- a/greeting.py", report.workspace_diff)
        self.assertIn('+    return f"你好，{name}"', report.workspace_diff)
        self.assertFalse(report.workspace_diff_truncated)
        self.assertTrue(report.trace_path)
        self.assertEqual(1.0, progress[-1][0])
        self.assertTrue(any("第 1/5 轮" in message for _, message in progress))
        self.assertTrue(any("独立评分" in message for _, message in progress))

    async def test_judge_failure_preserves_deterministic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            case = load_case(self._case(Path(directory)))
            runner = EvalRunner(
                self._config("executor", "executor-model"),
                self._config("bad-judge", "judge-model"),
                provider_factory=EvalProvider,
            )
            report = await runner.run(case)

        self.assertEqual(70.0, report.score)
        self.assertFalse(report.judge_result.available)
        self.assertIn("JSON", report.judge_result.error)

    def test_same_execution_and_judge_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "必须不同"):
            EvalRunner(
                self._config("executor", "same-model"),
                self._config("judge", "same-model"),
            )

    def test_loader_rejects_fixture_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "name: x\nprompt: y\nfixture: ../outside\n", encoding="utf-8",
            )
            with self.assertRaises(EvalConfigError):
                load_case(path)

    def test_directory_loader_filters_cases_by_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._case(root)
            (root / "second.yaml").write_text(
                "name: 另一个用例\nprompt: 直接回答\ntags: [direct]\n"
                "assertions:\n  final_text_contains: [回答]\n",
                encoding="utf-8",
            )

            cases = _load_cases(root, {"direct"})

        self.assertEqual(["另一个用例"], [case.name for case in cases])

    def test_progress_is_readable_when_stdout_is_not_a_terminal(self):
        stream = StringIO()
        progress = _TerminalProgress(stream, total=2)
        progress.start(1, "第一项")
        progress.update(0.5, "执行模型：第 2/4 轮")

        output = stream.getvalue()
        self.assertIn("0%", output)
        self.assertIn("25%", output)
        self.assertIn("第一项", output)


if __name__ == "__main__":
    unittest.main()

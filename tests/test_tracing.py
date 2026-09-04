import json
import tempfile
import unittest
from contextvars import Context
from pathlib import Path
from unittest.mock import patch

from tinyCode.config.models import TracingConfig
from tinyCode.tracing.recorder import TraceRecorder
from tinyCode.tracing.render import render_html, render_text


class TraceRecorderTests(unittest.TestCase):
    def test_detached_span_can_finish_from_another_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            handle = recorder.begin_task("work")
            assert handle is not None
            span = recorder.span(
                "request #1", "model_request", activate=False,
            )
            span.__enter__()

            Context().run(span.__exit__, None, None, None)
            recorder.finish_task(handle, status="cancelled")

            rows = [
                json.loads(line)
                for line in handle.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("span_start", rows[1]["event"])
            self.assertEqual("span_end", rows[2]["event"])
            self.assertEqual("ok", rows[2]["status"])
            self.assertEqual("", recorder.last_error)

    def test_records_spans_redacts_secrets_and_renders_both_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = TraceRecorder(TracingConfig(), root)
            handle = recorder.begin_task(
                "修复登录问题，secret-value 不应落盘",
                session_id="session-1",
                model="test-model",
                context_window=100_000,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            recorder.record("round_start", attributes={"round": 1, "max_rounds": 30})
            with recorder.span(
                "request #1", "model_request", {"round": 1, "api_key": "secret"},
            ) as span:
                span.event("model_first_token", {"first_token_ms": 12.5})
                span.finish("ok", {
                    "round": 1,
                    "first_token_ms": 12.5,
                    "estimated_tokens_before": 20,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cache_read_tokens": 8,
                    "cache_write_tokens": 2,
                    "cache_miss_tokens": 2,
                    "cache_usage_available": True,
                })
            recorder.record("round_end", status="completed", attributes={"round": 1})
            recorder.record(
                "tool_result",
                status="error",
                attributes={"tool": "run_command", "error": "exit 7"},
            )
            recorder.finish_task(handle, status="no_tool_call", attributes={
                "turns": 1,
                "model_requests": 1,
                "tool_calls": 0,
                "total_tokens": 15,
            })

            raw = handle.path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in raw.splitlines()]

            self.assertNotIn("secret-value", raw)
            self.assertNotIn('"api_key":"secret"', raw)
            self.assertIn("[REDACTED]", raw)
            self.assertIn('"estimated_tokens_before":20', raw)
            self.assertEqual("task_start", rows[0]["event"])
            self.assertEqual("task_end", rows[-1]["event"])
            terminal = render_text(handle.path)
            page = render_html(handle.path)
            self.assertIn("Turn 1", terminal)
            self.assertIn("模型 request #1", terminal)
            self.assertIn("关键路径耗时", terminal)
            self.assertIn("执行时间线", page)
            self.assertIn("关键路径耗时", page)
            self.assertIn("模型 Token 曲线", page)
            self.assertIn("上下文窗口曲线", page)
            self.assertIn("累计 15 Token", page)
            self.assertIn("缓存 8 (80%) · 写入 2", page)
            self.assertNotIn(">None<", page)
            self.assertIn("<details class='span-row'>", page)
            self.assertIn("<summary class='span-summary'>", page)
            self.assertIn("<pre class='span-json'>", page)
            self.assertNotIn("grid-template-columns:minmax(280px,36%) 1fr 70px", page)
            self.assertIn("<tr class='event-error'>", page)
            self.assertIn("class='event-status error'>error</span>", page)

    def test_tool_summary_does_not_store_full_command_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))

            attributes = recorder.tool_attributes(
                "run_command", {"command": "python3 script.py --token super-secret"},
            )

            self.assertEqual("python3", attributes["command"])
            self.assertNotIn("super-secret", str(attributes))
            self.assertIn("command_sha256", attributes)

    def test_incomplete_final_json_line_is_ignored_and_span_is_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            handle = recorder.begin_task("work")
            assert handle is not None
            span = recorder.span("read_file", "tool", {"round": 1})
            span.__enter__()
            with handle.path.open("a", encoding="utf-8") as stream:
                stream.write('{"broken":')

            rendered = render_text(handle.path)

            self.assertIn("异常中断", rendered)

    def test_disabled_recorder_does_not_create_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = TraceRecorder(TracingConfig(enabled=False), root)

            handle = recorder.begin_task("work")

            self.assertIsNone(handle)
            self.assertFalse((root / ".tinyCode" / "traces").exists())

    def test_trace_directory_symlink_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / ".tinyCode").mkdir()
            (root / ".tinyCode" / "traces").symlink_to(
                outside, target_is_directory=True,
            )
            recorder = TraceRecorder(TracingConfig(), root)

            handle = recorder.begin_task("work")

            self.assertIsNone(handle)
            self.assertIn("项目外部", recorder.last_error)
            self.assertEqual([], list(outside.iterdir()))

    def test_retention_keeps_configured_number_of_recent_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(
                TracingConfig(max_files=2, retention_days=14), Path(tmp),
            )

            for index in range(3):
                handle = recorder.begin_task(f"task {index}")
                self.assertIsNotNone(handle)
                recorder.finish_task(handle, status="no_tool_call")

            traces = list((Path(tmp) / ".tinyCode" / "traces").glob("*.jsonl"))
            self.assertEqual(2, len(traces))

    def test_clear_removes_jsonl_and_rendered_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            handle = recorder.begin_task("work")
            recorder.finish_task(handle, status="no_tool_call")
            self.assertIsNotNone(recorder.render_last_html())

            removed = recorder.clear()

            self.assertEqual(2, removed)
            self.assertIsNone(recorder.latest_path())

    def test_open_keeps_generated_html_when_browser_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(TracingConfig(), Path(tmp))
            handle = recorder.begin_task("work")
            recorder.finish_task(handle, status="no_tool_call")

            with patch("tinyCode.tracing.recorder.webbrowser.open", return_value=False):
                path = recorder.open_last()

            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.is_file())
            self.assertIn("未能打开浏览器", recorder.last_error)

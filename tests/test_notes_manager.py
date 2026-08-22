import asyncio
import tempfile
import unittest
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from tinyCode.notes.manager import (
    AutoNoteManager,
    MAX_NOTE_CONTEXT_CHARS,
    MAX_NOTE_OUTPUT_CHARS,
)


class CapturingProvider:
    def __init__(self, chunks: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.chunks = chunks or ["## 项目知识\n- 使用 Prompt Toolkit 构建 TUI"]

    async def chat_stream(self, messages):
        self.prompts.append(messages[0]["content"])
        for chunk in self.chunks:
            yield chunk


class ConcurrentProvider:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._usage: ContextVar[dict[str, int]] = ContextVar("note_usage", default={})

    @property
    def last_usage(self) -> dict[str, int]:
        return self._usage.get()

    def begin_request(self) -> None:
        self._usage.set({})

    async def chat_stream(self, messages):
        prompt = messages[0]["content"]
        category = next(
            name
            for name in ("用户偏好", "纠正反馈", "项目知识", "参考资料")
            if f"目标分类：{name}" in prompt
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            self._usage.set({"input_tokens": 2, "output_tokens": 1})
            yield f"## {category}\n- 测试内容"
        finally:
            self.active -= 1


class FailsOneCategoryOnceProvider(ConcurrentProvider):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def chat_stream(self, messages):
        prompt = messages[0]["content"]
        if "目标分类：纠正反馈" in prompt and not self.failed:
            self.failed = True
            raise ConnectionError("temporary note failure")
        async for chunk in super().chat_stream(messages):
            yield chunk


class AutoNoteManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_text_loads_user_and_current_project_notes(self):
        provider = CapturingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_notes = root / "user-notes"
            project_notes = root / "project-notes"
            user_notes.mkdir()
            project_notes.mkdir()
            (user_notes / "user_preferences.md").write_text(
                "## 用户偏好\n- 使用中文", encoding="utf-8",
            )
            (project_notes / "project_knowledge.md").write_text(
                "## 项目知识\n- 使用 Prompt Toolkit", encoding="utf-8",
            )
            with (
                patch("tinyCode.notes.manager.get_user_notes_dir", return_value=user_notes),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=project_notes,
                ),
            ):
                manager = AutoNoteManager(provider=provider, cwd=root)
                context = manager.context_text()

        self.assertIn("[用户偏好]", context)
        self.assertIn("使用中文", context)
        self.assertIn("[项目知识]", context)
        self.assertIn("使用 Prompt Toolkit", context)
        self.assertNotIn("[纠正反馈]", context)

    async def test_context_text_caps_large_notes(self):
        provider = CapturingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_notes = root / "user-notes"
            project_notes = root / "project-notes"
            user_notes.mkdir()
            project_notes.mkdir()
            (project_notes / "project_knowledge.md").write_text(
                "x" * (MAX_NOTE_CONTEXT_CHARS + 100), encoding="utf-8",
            )
            with (
                patch("tinyCode.notes.manager.get_user_notes_dir", return_value=user_notes),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=project_notes,
                ),
            ):
                manager = AutoNoteManager(provider=provider, cwd=root)
                context = manager.context_text()

        self.assertIn("字符未注入", context)
        self.assertLess(len(context), MAX_NOTE_CONTEXT_CHARS + 200)

    async def test_update_all_runs_categories_concurrently_and_sums_usage(self):
        provider = ConcurrentProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "tinyCode.notes.manager.get_user_notes_dir",
                    return_value=root / "user-notes",
                ),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=root / "project-notes",
                ),
            ):
                manager = AutoNoteManager(provider=provider, cwd=root)
                manager.record_round("用户消息", "助手回复")
                results = await manager.update_all()

        self.assertEqual(4, provider.max_active)
        self.assertEqual(4, len(results))
        self.assertEqual(4, manager.last_update_model_requests)
        self.assertEqual(12, manager.last_update_tokens)

    async def test_partial_update_failure_keeps_recent_text_for_retry(self):
        provider = FailsOneCategoryOnceProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "tinyCode.notes.manager.get_user_notes_dir",
                    return_value=root / "user-notes",
                ),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=root / "project-notes",
                ),
            ):
                manager = AutoNoteManager(provider=provider, cwd=root)
                manager.record_round("必须保留的用户消息", "助手回复")

                first = await manager.update_all()
                self.assertEqual(3, len(first))
                self.assertTrue(manager._recent_text)

                second = await manager.update_all()

        self.assertEqual(4, len(second))
        self.assertEqual([], manager._recent_text)

    async def test_read_only_note_directory_does_not_break_startup(self):
        provider = CapturingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = root / "blocked"
            with (
                patch("tinyCode.notes.manager.get_user_notes_dir", return_value=blocked),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=blocked,
                ),
                patch("pathlib.Path.mkdir", side_effect=PermissionError("read only")),
            ):
                manager = AutoNoteManager(provider=provider, cwd=root)

        self.assertTrue(manager.last_errors)

    async def test_update_one_prompts_for_only_the_target_category(self):
        provider = CapturingProvider()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_notes = root / "user-notes"
            project_root = root / "project"
            project_notes = project_root / ".tinyCode" / "notes"

            with (
                patch("tinyCode.notes.manager.get_user_notes_dir", return_value=user_notes),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=project_notes,
                ),
            ):
                manager = AutoNoteManager(provider=provider, cwd=project_root)
                await manager._update_one(
                    project_notes / "project_knowledge.md",
                    "项目知识",
                    "[user]: 这个项目用 Prompt Toolkit",
                )

        prompt = provider.prompts[0]
        self.assertIn("目标分类：项目知识", prompt)
        self.assertIn("只输出这个分类的完整笔记内容", prompt)
        self.assertNotIn("## 用户偏好", prompt)
        self.assertNotIn("## 纠正反馈", prompt)
        self.assertNotIn("## 参考资料", prompt)

    async def test_update_one_rejects_output_without_target_heading(self):
        provider = CapturingProvider(["无需更新"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_notes = root / "user-notes"
            project_root = root / "project"
            project_notes = project_root / ".tinyCode" / "notes"
            project_notes.mkdir(parents=True)
            note_path = project_notes / "project_knowledge.md"
            note_path.write_text("## 项目知识\n- 已有事实\n", encoding="utf-8")

            with (
                patch("tinyCode.notes.manager.get_user_notes_dir", return_value=user_notes),
                patch(
                    "tinyCode.notes.manager.get_project_notes_dir",
                    return_value=project_notes,
                ),
            ):
                manager = AutoNoteManager(provider=provider, cwd=project_root)
                result = await manager._update_one(
                    note_path,
                    "项目知识",
                    "[user]: 这个项目用 Prompt Toolkit",
                )

            self.assertIsNone(result)
            self.assertEqual("## 项目知识\n- 已有事实\n", note_path.read_text(encoding="utf-8"))

    async def test_update_one_rejects_oversized_model_output(self):
        provider = CapturingProvider(["x" * (MAX_NOTE_OUTPUT_CHARS + 1)])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("tinyCode.notes.manager.get_user_notes_dir", return_value=root / "user"),
                patch("tinyCode.notes.manager.get_project_notes_dir", return_value=root / "notes"),
            ):
                manager = AutoNoteManager(provider=provider, cwd=root)
                result = await manager._update_one(
                    root / "notes" / "project_knowledge.md",
                    "项目知识",
                    "[user]: update",
                )

        self.assertIsNone(result)
        self.assertTrue(any("字符上限" in error for error in manager.last_errors))

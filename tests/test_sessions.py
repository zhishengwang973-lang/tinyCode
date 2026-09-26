import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyCode.conversation.history import ConversationHistory
from tinyCode.storage import sessions
from tinyCode.storage.sessions import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_multimodal_user_message_survives_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            content = [
                {"type": "text", "text": "分析截图"},
                {
                    "type": "image_file",
                    "image_file": {
                        "path": "/project/.tinyCode/attachments/a.png",
                        "media_type": "image/png",
                        "detail": "auto",
                        "size": 12,
                    },
                },
            ]
            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                sid = store.new_session()
                history = ConversationHistory()
                history.add_user_message(content)
                store.save(history, "deepseek", "deepseek-flash")
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            restored, provider, model = loaded
            self.assertEqual("deepseek", provider)
            self.assertEqual("deepseek-flash", model)
            self.assertEqual(content, restored.get_messages()[0]["content"])

    def test_time_gap_handles_mixed_legacy_naive_and_aware_timestamps(self):
        messages = [
            {
                "role": "user",
                "content": "before",
                "timestamp": "2026-01-01T00:00:00",
            },
            {
                "role": "assistant",
                "content": "after",
                "timestamp": "2026-01-01T01:00:00+00:00",
            },
        ]

        restored = SessionStore._insert_time_gaps(messages)

        self.assertEqual(3, len(restored))
        self.assertIn("时间跨度提醒", restored[1]["content"])

    def test_new_session_creates_empty_jsonl_and_is_listable(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                sid = store.new_session()
                listed = store.list_sessions()

            self.assertTrue((sessions_dir / f"{sid}.jsonl").exists())
            self.assertEqual("", (sessions_dir / f"{sid}.jsonl").read_text())
            self.assertEqual([sid], [item["id"] for item in listed])

    def test_save_appends_only_new_messages_and_preserves_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                sid = store.new_session()
                history = ConversationHistory()
                history.add_user_message("first")
                store.save(history, "openai", "gpt-test")
                path = sessions_dir / f"{sid}.jsonl"
                first_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

                history.add_assistant_message("second")
                store.save(history, "openai", "gpt-test")
                rows = [json.loads(line) for line in path.read_text().splitlines()]

            self.assertEqual(2, len(rows))
            self.assertEqual(first_row["timestamp"], rows[0]["timestamp"])
            self.assertEqual(["first", "second"], [row["content"] for row in rows])

    def test_load_strips_timestamps_from_provider_visible_tool_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "timestamps"
            rows = [
                {"role": "user", "content": "read", "timestamp": "2026-01-01T00:00:00+00:00"},
                {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
                    "timestamp": "2026-01-01T00:00:01+00:00",
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok", "timestamp": "2026-01-01T00:00:02+00:00"},
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
            )
            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            history, _, _ = loaded
            self.assertTrue(all(
                "timestamp" not in message for message in history.get_messages()
            ))

    def test_generated_time_gap_is_not_written_back_as_a_real_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "gap"
            rows = [
                {"role": "user", "content": "before", "timestamp": "2026-01-01T00:00:00+00:00"},
                {"role": "assistant", "content": "after", "timestamp": "2026-01-01T02:00:00+00:00"},
            ]
            path = sessions_dir / f"{sid}.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
            )
            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                history, _, _ = store.load(sid)
                self.assertEqual(3, len(history.get_messages()))
                store.save(history, "openai", "gpt-test")

            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(2, len(saved))
            self.assertFalse(any(
                str(row.get("content", "")).startswith("[时间跨度提醒]")
                for row in saved
            ))

    def test_load_keeps_paired_anthropic_tool_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "anthropic-pair"
            rows = [
                {"role": "user", "content": "read"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "body",
                    }],
                },
                {"role": "assistant", "content": "done"},
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual(
                ["user", "assistant", "user", "assistant"],
                [message["role"] for message in history.get_messages()],
            )

    def test_load_drops_orphan_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "orphan-tool"
            rows = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {
                    "role": "tool",
                    "tool_call_id": "missing-call",
                    "name": "sub_agent",
                    "content": "background result",
                },
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual(
                ["user", "assistant"],
                [message["role"] for message in history.get_messages()],
            )

    def test_load_keeps_paired_tool_call_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "abc123"
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            meta_path = sessions_dir / f"{sid}.meta.json"
            rows = [
                {"role": "user", "content": "read the file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
                {"role": "assistant", "content": "done"},
            ]
            jsonl_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            meta_path.write_text(
                json.dumps({"id": sid, "provider": "openai", "model": "gpt-test"}),
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, provider, model = loaded
            messages = history.get_messages()
            self.assertEqual(["user", "assistant", "tool", "assistant"], [m["role"] for m in messages])
            self.assertEqual("openai", provider)
            self.assertEqual("gpt-test", model)

    def test_load_skips_corrupt_jsonl_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "def456"
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            jsonl_path.write_text(
                "\n".join([
                    json.dumps({"role": "user", "content": "hello"}),
                    "{bad json",
                    json.dumps({"role": "assistant", "content": "hi"}),
                ]) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual(["hello", "hi"], [m["content"] for m in history.get_messages()])

    def test_load_skips_non_object_jsonl_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "shape123"
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            jsonl_path.write_text(
                "\n".join([
                    json.dumps({"role": "user", "content": "hello"}),
                    json.dumps(["not", "a", "message"]),
                    json.dumps("also not a message"),
                    json.dumps({"role": "assistant", "content": "hi"}),
                ]) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual(["hello", "hi"], [m["content"] for m in history.get_messages()])

    def test_load_drops_provider_unsafe_message_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "unsafe-shapes"
            rows = [
                {"role": "user", "content": "keep"},
                {"role": "assistant", "content": 42},
                {"role": "tool", "tool_call_id": ["bad"], "content": "bad"},
                {"role": "user", "content": [{"type": "text", "text": "ok"}, "bad"]},
                {"role": "assistant", "content": "done"},
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual(
                [
                    {"role": "user", "content": "keep"},
                    {"role": "assistant", "content": "done"},
                ],
                history.get_messages(),
            )

    def test_load_filters_invalid_tool_calls_without_losing_valid_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "mixed-tool-calls"
            rows = [
                {"role": "user", "content": "read"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": [], "function": {"name": "bad", "arguments": "{}"}},
                        {"id": "good", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "good", "content": "body"},
                {"role": "assistant", "content": "done"},
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            messages = history.get_messages()
            self.assertEqual("good", messages[1]["tool_calls"][0]["id"])
            self.assertEqual(["user", "assistant", "tool", "assistant"], [m["role"] for m in messages])

    def test_load_skips_malformed_tool_call_entries_during_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "toolbad"
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            rows = [
                {"role": "user", "content": "before"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        "not a tool call",
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
                {"role": "assistant", "content": "done"},
            ]
            jsonl_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual(["user", "assistant", "tool", "assistant"], [m["role"] for m in history.get_messages()])

    def test_load_ignores_non_object_meta_without_losing_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "meta123"
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            meta_path = sessions_dir / f"{sid}.meta.json"
            jsonl_path.write_text(
                json.dumps({"role": "user", "content": "hello"}) + "\n",
                encoding="utf-8",
            )
            meta_path.write_text(
                json.dumps(["not", "a", "meta", "object"]),
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, provider, model = loaded
            self.assertEqual([{"role": "user", "content": "hello"}], history.get_messages())
            self.assertEqual("", provider)
            self.assertEqual("", model)

    def test_append_message_recovers_from_malformed_meta_message_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                sid = store.new_session()
                meta_path = sessions_dir / f"{sid}.meta.json"
                meta_path.write_text(
                    json.dumps({"id": sid, "message_count": "many"}),
                    encoding="utf-8",
                )

                store.append_message({"role": "user", "content": "hello"})

            rows = (sessions_dir / f"{sid}.jsonl").read_text(encoding="utf-8").splitlines()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(rows))
            self.assertEqual(1, meta["message_count"])

    def test_load_truncates_unpaired_tool_call_and_later_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "ghi789"
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            rows = [
                {"role": "user", "content": "before"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_missing",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "assistant", "content": "should be truncated"},
            ]
            jsonl_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual([{"role": "user", "content": "before"}], history.get_messages())

    def test_load_recovers_partial_final_tool_batch_with_synthetic_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "partial-batch"
            rows = [
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load(sid)

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            messages = history.get_messages()
            self.assertEqual(["call_1", "call_2"], [m["tool_call_id"] for m in messages[2:]])
            self.assertIn("状态未知", messages[-1]["content"])

    def test_default_load_skips_orphan_meta_without_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            good_sid = "good123"
            orphan_sid = "orphan123"
            good_jsonl = sessions_dir / f"{good_sid}.jsonl"
            good_meta = sessions_dir / f"{good_sid}.meta.json"
            orphan_meta = sessions_dir / f"{orphan_sid}.meta.json"
            good_jsonl.write_text(
                json.dumps({"role": "user", "content": "hello"}) + "\n",
                encoding="utf-8",
            )
            good_meta.write_text(json.dumps({"id": good_sid}), encoding="utf-8")
            orphan_meta.write_text(json.dumps({"id": orphan_sid}), encoding="utf-8")
            os.utime(good_meta, (100, 100))
            os.utime(orphan_meta, (200, 200))

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load()

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual([{"role": "user", "content": "hello"}], history.get_messages())

    def test_default_load_falls_back_to_recent_jsonl_when_meta_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            old_sid = "old123"
            new_sid = "new123"
            old_jsonl = sessions_dir / f"{old_sid}.jsonl"
            new_jsonl = sessions_dir / f"{new_sid}.jsonl"
            old_jsonl.write_text(
                json.dumps({"role": "user", "content": "old"}) + "\n",
                encoding="utf-8",
            )
            new_jsonl.write_text(
                json.dumps({"role": "user", "content": "new"}) + "\n",
                encoding="utf-8",
            )
            os.utime(old_jsonl, (100, 100))
            os.utime(new_jsonl, (200, 200))

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load()

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual([{"role": "user", "content": "new"}], history.get_messages())

    def test_list_sessions_skips_orphan_meta_without_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            good_sid = "good123"
            orphan_sid = "orphan123"
            (sessions_dir / f"{good_sid}.jsonl").write_text(
                json.dumps({"role": "user", "content": "hello"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"{good_sid}.meta.json").write_text(
                json.dumps({"id": good_sid, "title": "Good"}),
                encoding="utf-8",
            )
            (sessions_dir / f"{orphan_sid}.meta.json").write_text(
                json.dumps({"id": orphan_sid, "title": "Orphan"}),
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                listed = SessionStore().list_sessions()

            self.assertEqual([good_sid], [item["id"] for item in listed])

    def test_list_sessions_includes_jsonl_without_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            meta_sid = "withmeta"
            jsonl_only_sid = "jsonlonly"
            (sessions_dir / f"{meta_sid}.jsonl").write_text(
                json.dumps({"role": "user", "content": "with meta"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"{meta_sid}.meta.json").write_text(
                json.dumps({"id": meta_sid, "title": "With Meta"}),
                encoding="utf-8",
            )
            (sessions_dir / f"{jsonl_only_sid}.jsonl").write_text(
                json.dumps({"role": "user", "content": "jsonl only"}) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                listed = SessionStore().list_sessions()

            self.assertCountEqual([meta_sid, jsonl_only_sid], [item["id"] for item in listed])

    def test_list_sessions_derives_missing_meta_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "jsonlonly"
            rows = [
                {
                    "role": "user",
                    "content": "please inspect the project",
                    "timestamp": "2026-07-26T01:00:00+00:00",
                },
                {
                    "role": "assistant",
                    "content": "ok",
                    "timestamp": "2026-07-26T01:01:00+00:00",
                },
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                listed = SessionStore().list_sessions()

            self.assertEqual(1, len(listed))
            self.assertEqual(sid, listed[0]["id"])
            self.assertEqual("please inspect the project", listed[0]["title"])
            self.assertEqual(2, listed[0]["message_count"])
            self.assertEqual("2026-07-26T01:01:00+00:00", listed[0]["last_active_at"])

    def test_list_sessions_derives_meta_only_from_valid_jsonl_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "jsonlonly"
            rows = [
                {
                    "role": "user",
                    "content": "real request",
                    "timestamp": "2026-07-26T01:00:00+00:00",
                },
                {
                    "not": "a message",
                    "timestamp": "2026-07-26T09:00:00+00:00",
                },
                {
                    "role": "assistant",
                    "content": "ok",
                    "timestamp": "2026-07-26T01:01:00+00:00",
                },
                {
                    "role": 123,
                    "content": "bad role",
                    "timestamp": "2026-07-26T10:00:00+00:00",
                },
                ["not", "an", "object"],
            ]
            (sessions_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n{bad json\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                listed = SessionStore().list_sessions()

            self.assertEqual(1, len(listed))
            self.assertEqual(sid, listed[0]["id"])
            self.assertEqual("real request", listed[0]["title"])
            self.assertEqual(2, listed[0]["message_count"])
            self.assertEqual("2026-07-26T01:01:00+00:00", listed[0]["last_active_at"])

    def test_list_sessions_orders_all_entries_by_recent_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            old_meta_sid = "z_oldmeta"
            new_jsonl_sid = "a_newjsonl"
            old_meta_jsonl = sessions_dir / f"{old_meta_sid}.jsonl"
            old_meta_file = sessions_dir / f"{old_meta_sid}.meta.json"
            new_jsonl_file = sessions_dir / f"{new_jsonl_sid}.jsonl"
            old_meta_jsonl.write_text(
                json.dumps({"role": "user", "content": "old"}) + "\n",
                encoding="utf-8",
            )
            old_meta_file.write_text(
                json.dumps({"id": old_meta_sid, "title": "Old"}),
                encoding="utf-8",
            )
            new_jsonl_file.write_text(
                json.dumps({"role": "user", "content": "new"}) + "\n",
                encoding="utf-8",
            )
            os.utime(old_meta_file, (100, 100))
            os.utime(new_jsonl_file, (200, 200))

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                listed = SessionStore().list_sessions()

            self.assertEqual([new_jsonl_sid, old_meta_sid], [item["id"] for item in listed])

    def test_resolve_id_treats_glob_characters_as_literal_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "abc123"
            (sessions_dir / f"{sid}.jsonl").write_text(
                json.dumps({"role": "user", "content": "hello"}) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                self.assertIsNone(store.load("?"))
                self.assertFalse(store.delete("?"))
                self.assertTrue((sessions_dir / f"{sid}.jsonl").exists())

    def test_session_identifier_cannot_escape_session_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_dir = root / "sessions"
            sessions_dir.mkdir()
            outside = root / "outside.jsonl"
            outside.write_text(
                json.dumps({"role": "user", "content": "private"}) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                store = SessionStore()
                self.assertIsNone(store.load("../outside"))
                self.assertFalse(store.delete("../outside"))

            self.assertTrue(outside.exists())

    def test_resolve_id_still_supports_literal_prefix_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            sid = "abc123"
            (sessions_dir / f"{sid}.jsonl").write_text(
                json.dumps({"role": "user", "content": "hello"}) + "\n",
                encoding="utf-8",
            )

            with patch.object(sessions, "SESSIONS_DIR", sessions_dir):
                loaded = SessionStore().load("abc")

            self.assertIsNotNone(loaded)
            history, _, _ = loaded
            self.assertEqual([{"role": "user", "content": "hello"}], history.get_messages())


if __name__ == "__main__":
    unittest.main()

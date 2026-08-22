import tempfile
import json
import unittest
from pathlib import Path

from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.models import MessageType, TeamMessage


class MailboxTests(unittest.TestCase):
    def test_read_new_skips_invalid_ids_and_normalizes_corrupt_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            mailbox = Mailbox(team_dir, "alice")
            mailbox._file.write_text(
                "\n".join([
                    json.dumps({"id": [], "content": "skip"}),
                    json.dumps({
                        "id": "ok", "from": 42, "to": "alice",
                        "type": [], "content": ["bad"],
                    }),
                ]) + "\n",
                encoding="utf-8",
            )

            messages = mailbox.read_new()

            self.assertEqual(["ok"], [message.id for message in messages])
            self.assertEqual("", messages[0].from_member)
            self.assertEqual("", messages[0].content)
            self.assertEqual(MessageType.TEXT, messages[0].msg_type)

    def test_read_new_skips_invalid_utf8_row_and_keeps_later_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            mailbox = Mailbox(Path(tmp), "alice")
            mailbox._file.write_bytes(
                b'\xff\xfe not-json\n'
                + json.dumps({"id": "ok", "content": "survived"}).encode("utf-8")
                + b"\n"
            )

            messages = mailbox.read_new()

            self.assertEqual(["survived"], [message.content for message in messages])

    def test_send_routes_message_to_recipient_mailbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            lead_mailbox = Mailbox(team_dir, "lead")

            lead_mailbox.send(
                TeamMessage(
                    from_member="lead",
                    to_member="alice",
                    msg_type=MessageType.TEXT,
                    content="please inspect auth",
                )
            )

            self.assertEqual(
                ["please inspect auth"],
                [msg.content for msg in Mailbox(team_dir, "alice").read_new()],
            )
            self.assertEqual([], Mailbox(team_dir, "lead").read_new())

    def test_read_new_restores_message_type_enum(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            lead_mailbox = Mailbox(team_dir, "lead")
            lead_mailbox.send(
                TeamMessage(
                    from_member="lead",
                    to_member="alice",
                    msg_type=MessageType.TEXT,
                    content="please inspect auth",
                )
            )

            messages = Mailbox(team_dir, "alice").read_new()

            self.assertEqual(MessageType.TEXT, messages[0].msg_type)

    def test_broadcast_routes_to_all_other_members_without_nested_mailboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            lead_mailbox = Mailbox(team_dir, "lead")

            lead_mailbox.broadcast(
                TeamMessage(
                    from_member="lead",
                    msg_type=MessageType.BROADCAST,
                    content="sync now",
                ),
                ["lead", "alice", "bob"],
            )

            self.assertEqual(["sync now"], [m.content for m in Mailbox(team_dir, "alice").read_new()])
            self.assertEqual(["sync now"], [m.content for m in Mailbox(team_dir, "bob").read_new()])
            self.assertFalse((team_dir / "mailboxes" / "mailboxes").exists())

    def test_constructor_rejects_member_name_that_escapes_mailbox_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                Mailbox(Path(tmp), "../lead")

    def test_send_rejects_recipient_name_that_escapes_mailbox_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            team_dir = Path(tmp)
            lead_mailbox = Mailbox(team_dir, "lead")

            with self.assertRaises(ValueError):
                lead_mailbox.send(
                    TeamMessage(
                        from_member="lead",
                        to_member="../outside",
                        msg_type=MessageType.TEXT,
                        content="do not escape",
                    )
                )

            self.assertFalse((team_dir / "outside.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

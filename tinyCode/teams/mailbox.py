"""Mailbox — per-member message files for point-to-point communication."""

import json
import re
from pathlib import Path

from tinyCode.teams.models import MessageType, TeamMessage


_MEMBER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_member_name(member_name: str) -> str:
    if not _MEMBER_NAME_RE.fullmatch(member_name):
        raise ValueError(f"Invalid team member name: {member_name}")
    return member_name


class Mailbox:
    """Append-only JSONL mailbox for a single team member."""

    def __init__(self, team_dir: Path, member_name: str) -> None:
        self._dir = team_dir / "mailboxes"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._member_name = _validate_member_name(member_name)
        self._file = self._file_for(member_name)

    def send(self, msg: TeamMessage) -> None:
        target_name = msg.to_member or self._member_name
        with open(self._file_for(target_name), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": msg.id, "from": msg.from_member, "to": msg.to_member,
                "type": msg.msg_type.value, "content": msg.content,
                "summary": msg.summary, "timestamp": msg.timestamp,
            }, ensure_ascii=False) + "\n")

    def read_new(self, since_id: str = "") -> list[TeamMessage]:
        """Read messages since *since_id* (empty = all)."""
        if not self._file.exists():
            return []
        messages: list[TeamMessage] = []
        found_since = not since_id
        try:
            # A partially written or externally damaged JSONL file must not
            # terminate an otherwise healthy team run.  Replacement decoding
            # lets us skip only the broken row while preserving later rows.
            with open(self._file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    msg_id = data.get("id", "")
                    if not isinstance(msg_id, str) or not msg_id:
                        continue
                    if not found_since:
                        if msg_id == since_id:
                            found_since = True
                        continue
                    messages.append(TeamMessage(
                        id=msg_id,
                        from_member=self._text(data.get("from")),
                        to_member=self._text(data.get("to")),
                        msg_type=self._parse_message_type(data.get("type", "text")),
                        content=self._text(data.get("content")),
                        summary=self._text(data.get("summary")),
                        timestamp=self._text(data.get("timestamp")),
                    ))
        except OSError:
            return []
        return messages

    def broadcast(self, msg: TeamMessage, all_members: list[str]) -> None:
        """Send *msg* to every member's mailbox."""
        for name in all_members:
            if name == msg.from_member:
                continue
            self.send(TeamMessage(
                id=msg.id,
                from_member=msg.from_member,
                to_member=name,
                msg_type=msg.msg_type,
                content=msg.content,
                summary=msg.summary,
                timestamp=msg.timestamp,
            ))

    def _file_for(self, member_name: str) -> Path:
        return self._dir / f"{_validate_member_name(member_name)}.jsonl"

    @staticmethod
    def _parse_message_type(value: object) -> MessageType:
        try:
            return MessageType(value)
        except (ValueError, TypeError):
            return MessageType.TEXT

    @staticmethod
    def _text(value: object) -> str:
        return value if isinstance(value, str) else ""

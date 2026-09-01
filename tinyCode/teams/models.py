"""Team data models — team, member, task, message."""

import uuid
from dataclasses import dataclass, field
from enum import Enum

from tinyCode.time_utils import beijing_now_iso


class MemberStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    DONE = "done"
    FAILED = "failed"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageType(Enum):
    TEXT = "text"               # plain text with optional summary
    LIFECYCLE = "lifecycle"     # member started / finished / failed
    APPROVAL = "approval"       # approval request / reply
    BROADCAST = "broadcast"     # send to all members


@dataclass
class TeamTask:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    description: str = ""
    run_id: str = ""              # execution that owns this task
    preferred_member: str = ""    # scheduler hint; assigned_to is runtime state
    assigned_to: str = ""         # member name
    depends_on: list[str] = field(default_factory=list)  # task IDs
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    created_at: str = field(default_factory=beijing_now_iso)
    updated_at: str = ""


@dataclass
class TeamMessage:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    from_member: str = ""
    to_member: str = ""           # "" = broadcast
    msg_type: MessageType = MessageType.TEXT
    content: str = ""
    summary: str = ""             # optional one-line summary
    timestamp: str = field(default_factory=beijing_now_iso)


@dataclass
class MemberDef:
    name: str
    role: str = ""                # SubAgentRole name
    worktree: str = ""            # worktree name
    backend: str = "coro"         # "coro" | "terminal"
    needs_approval: bool = False
    model: str = ""               # override model


@dataclass
class TeamDef:
    name: str
    description: str = ""
    lead_role: str = ""           # role for the lead agent
    members: list[MemberDef] = field(default_factory=list)
    dispatch_mode: bool = False   # double-lock scheduling mode
    max_rounds_per_member: int = 10

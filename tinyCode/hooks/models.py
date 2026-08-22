"""Hook models — Event, Condition, Action, Rule."""

from dataclasses import dataclass, field
from enum import Enum


class HookEvent(Enum):
    # Session
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    # Round
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    # Message
    MESSAGE_PRE_SEND = "message_pre_send"
    MESSAGE_POST_RECEIVE = "message_post_receive"
    # Tool
    TOOL_PRE_EXEC = "tool_pre_exec"
    TOOL_POST_EXEC = "tool_post_exec"
    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_COMPRESS = "system_compress"


_INTERCEPT_EVENTS = {HookEvent.TOOL_PRE_EXEC}


class Operator(Enum):
    EXACT = "exact"
    NOT = "not"
    REGEX = "regex"
    GLOB = "glob"


class MatchMode(Enum):
    ALL = "ALL"
    ANY = "ANY"


class ActionType(Enum):
    SHELL = "shell"
    PROMPT_INJECT = "prompt_inject"
    HTTP = "http"
    SUB_AGENT = "sub_agent"


@dataclass
class ConditionRule:
    field: str          # dot-path into context: "tool_name", "params.path"
    operator: Operator
    value: str


@dataclass
class Condition:
    match: MatchMode = MatchMode.ALL
    rules: list[ConditionRule] = field(default_factory=list)


@dataclass
class Action:
    type: ActionType
    command: str = ""       # shell
    text: str = ""          # prompt_inject
    url: str = ""           # http
    method: str = "POST"    # http
    headers: dict = field(default_factory=dict)
    body: str = ""          # http
    task: str = ""          # sub_agent


@dataclass
class Control:
    once: bool = False       # execute only once, then auto-disable
    async_: bool = False     # run in background (forbidden if intercept)
    timeout: float = 30.0


@dataclass
class Rule:
    event: HookEvent
    condition: Condition | None = None
    actions: list[Action] = field(default_factory=list)
    control: Control = field(default_factory=Control)
    name: str = ""           # for logging / debugging

    @property
    def is_intercept(self) -> bool:
        return self.event in _INTERCEPT_EVENTS
